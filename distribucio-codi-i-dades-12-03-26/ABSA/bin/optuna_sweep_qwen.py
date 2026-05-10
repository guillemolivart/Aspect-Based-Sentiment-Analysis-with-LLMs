import argparse
import csv
import json
import sys
import time
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

from common import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PROMPT_PATH,
    OUTPUT_DIR,
    encode,
    extract_json,
    generate,
    get_prompts,
    load_dataset,
    prepare_messages,
)


def counts(prediction, gold):
    """Count predicted vs expected items and matches."""
    def flatten(value, prefix=""):
        if isinstance(value, (str, int, float, bool)):
            return [prefix + ":" + str(value)]
        if isinstance(value, list):
            flattened = []
            for item in value:
                flattened.extend(flatten(item, prefix))
            return flattened
        if isinstance(value, dict):
            flattened = []
            for key, item in value.items():
                flattened.extend(flatten(item, prefix + f".{key}"))
            return flattened
        return []

    pred = flatten(prediction)
    expected = flatten(gold)
    ok = sum(1 for item in pred if item in expected)
    return len(pred), len(expected), ok


def prf(predicted, expected, ok):
    """Calculate precision, recall, F1."""
    precision = 100.0 * ok / predicted if predicted else 0.0
    recall = 100.0 * ok / expected if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def load_model(model_path, load_in_4bit=False):
    """Load the Qwen model and tokenizer."""
    model_path = Path(model_path)
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(
            f"Model not found in {model_path}. Download Qwen/Qwen3.5-2B there first."
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    
    model_kwargs = {"device_map": "auto"}
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    else:
        model_kwargs["torch_dtype"] = dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    except TypeError:
        if "torch_dtype" in model_kwargs:
            model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    model.eval()
    return model, tokenizer


def objective(trial, model, tokenizer, prompts, examples, args):
    """
    Optuna objective function: evaluate a configuration and return the score.
    Score = 0.75 * f1_macro + 0.25 * f1_micro (to maximize).
    """
    
    # Suggest hyperparameters from narrow search space
    temperature = trial.suggest_categorical("temperature", [0.6, 0.65, 0.68, 0.7])
    top_p = trial.suggest_categorical("top_p", [0.7, 0.75, 0.78, 0.8])
    top_k = trial.suggest_categorical("top_k", [10, 15, 20, 25])
    presence_penalty = trial.suggest_categorical("presence_penalty", [1.8, 1.9, 2.0, 2.1])
    
    # Fixed parameters
    min_p = 0.0
    repetition_penalty = 1.0
    max_new_tokens = args.max_new_tokens
    
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "presence_penalty": presence_penalty,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
    }
    
    print(f"Trial {trial.number}: {generation_config}")
    
    # Evaluate on the dataset
    t0 = time.time()
    processed = []
    predicted_total = expected_total = ok_total = 0
    precision_macro = recall_macro = f1_macro = 0.0
    
    for index, example in enumerate(examples, start=args.start):
        messages = prepare_messages(prompts, example)
        model_inputs = encode(tokenizer, messages, enable_thinking=False)
        gen_text = generate(model, tokenizer, model_inputs, generation_config)
        prediction = extract_json(gen_text)
        
        predicted, expected, ok = counts(prediction, example["gold"])
        predicted_total += predicted
        expected_total += expected
        ok_total += ok
        
        processed.append({"id": example["id"], "prediction": prediction, "gold": example["gold"]})
    
    # Calculate overall metrics
    n = len(processed)
    
    if predicted_total > 0:
        precision_macro = 100.0 * ok_total / predicted_total
    if expected_total > 0:
        recall_macro = 100.0 * ok_total / expected_total
    if precision_macro + recall_macro > 0:
        f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro)
    
    # Micro F1 (per-sample average)
    precision_micro = recall_micro = f1_micro = 0.0
    if n > 0:
        for example in processed:
            pred, exp, ok = counts(example["prediction"], example["gold"])
            p, r, f = prf(pred, exp, ok)
            precision_micro += p
            recall_micro += r
            f1_micro += f
        
        precision_micro /= n
        recall_micro /= n
        f1_micro /= n
    
    elapsed = time.time() - t0
    sec_per_example = elapsed / len(processed) if processed else 0.0
    
    # Composite score: 75% macro F1 + 25% micro F1
    score = (0.75 * f1_macro) + (0.25 * f1_micro)
    
    # Store trial info
    trial.set_user_attr("generation_config", generation_config)
    trial.set_user_attr("f1_macro", f1_macro)
    trial.set_user_attr("f1_micro", f1_micro)
    trial.set_user_attr("precision_macro", precision_macro)
    trial.set_user_attr("recall_macro", recall_macro)
    trial.set_user_attr("precision_micro", precision_micro)
    trial.set_user_attr("recall_micro", recall_micro)
    trial.set_user_attr("time_sec", elapsed)
    trial.set_user_attr("sec_per_example", sec_per_example)
    
    print(f"  -> f1_macro={f1_macro:.2f}%, f1_micro={f1_micro:.2f}%, score={score:.2f}")
    
    return score


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for ABSA with Qwen")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--data", default="devel")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow running without CUDA (very slow)")
    parser.add_argument("--output-prefix", default="optuna_results")
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    out_dir = OUTPUT_DIR / "optuna"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{args.output_prefix}.run.log"

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, text):
            for stream in self.streams:
                stream.write(text)
                stream.flush()

        def flush(self):
            for stream in self.streams:
                stream.flush()

        def isatty(self):
            for stream in self.streams:
                checker = getattr(stream, "isatty", None)
                if callable(checker):
                    try:
                        if checker():
                            return True
                    except Exception:
                        continue
            return False

    log_fd = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _Tee(original_stdout, log_fd)
    sys.stderr = _Tee(original_stderr, log_fd)
    
    try:
        # Load data and model
        prompts = get_prompts(args.prompt_file)
        examples, data_path = load_dataset(args.data)

        if args.limit is not None:
            selected = examples[args.start : args.start + args.limit]
        else:
            selected = examples[args.start :]

        import torch
        model_path = Path(args.model_path)
        tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        cuda_available = torch.cuda.is_available()
        if not cuda_available and not args.allow_cpu:
            raise RuntimeError(
                "CUDA is not available in the current Python environment. "
                "Optuna would run on CPU and be extremely slow. "
                "Fix Torch/CUDA compatibility or pass --allow-cpu to force CPU execution."
            )

        dtype = torch.bfloat16 if cuda_available and torch.cuda.is_bf16_supported() else (
            torch.float16 if cuda_available else torch.float32
        )

        model_kwargs = {"device_map": "auto"}
        if args.load_in_4bit:
            model_kwargs["quantization_config"] = __import__("transformers").BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
            )
        else:
            model_kwargs["torch_dtype"] = dtype

        try:
            model = __import__("transformers").AutoModel.from_pretrained(
                model_path, trust_remote_code=True, **model_kwargs
            )
        except TypeError:
            if "torch_dtype" in model_kwargs:
                model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
            model = __import__("transformers").AutoModel.from_pretrained(
                model_path, trust_remote_code=True, **model_kwargs
            )

        model.eval()

        print("========= OPTUNA HYPERPARAMETER OPTIMIZATION =========")
        print(f"model_path={args.model_path}")
        print(f"prompt_file={prompts['path']}")
        print(f"data={data_path}")
        print(f"examples={len(selected)}")
        print(f"n_trials={args.n_trials}")
        print(f"torch_version={torch.__version__}")
        print(f"torch_cuda_version={torch.version.cuda}")
        print(f"cuda_available={cuda_available}")
        print(f"log_path={log_path}")
        print("=" * 60)

        # Create Optuna study with TPE sampler and seed
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            sampler=sampler,
            direction="maximize",
            study_name=args.output_prefix
        )

        # Inject baseline configuration as the very first trial
        baseline_params = {
            "temperature": 0.65,
            "top_p": 0.75,
            "top_k": 20,
            "presence_penalty": 2.0,
        }
        study.enqueue_trial(baseline_params)

        # Optimize
        study.optimize(
            lambda trial: objective(trial, model, tokenizer, prompts, selected, args),
            n_trials=args.n_trials,
            show_progress_bar=True
        )

        # Extract best trial
        best_trial = study.best_trial

        print("\n" + "=" * 60)
        print(f"Best trial: {best_trial.number}")
        print(f"Best score: {best_trial.value:.4f}")
        print(f"Best params: {best_trial.params}")
        print("=" * 60)

        # Define output paths BEFORE using them
        csv_path = out_dir / f"{args.output_prefix}.summary.csv"
        json_path = out_dir / f"{args.output_prefix}.chosen.json"

        # Save best config as JSON
        best_config = {
            "trial_number": best_trial.number,
            "score": best_trial.value,
            "generation_config": {
                "temperature": best_trial.params["temperature"],
                "top_p": best_trial.params["top_p"],
                "top_k": best_trial.params["top_k"],
                "presence_penalty": best_trial.params["presence_penalty"],
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "max_new_tokens": args.max_new_tokens,
            },
            "metrics": {
                "f1_macro": best_trial.user_attrs.get("f1_macro"),
                "f1_micro": best_trial.user_attrs.get("f1_micro"),
                "precision_macro": best_trial.user_attrs.get("precision_macro"),
                "recall_macro": best_trial.user_attrs.get("recall_macro"),
                "precision_micro": best_trial.user_attrs.get("precision_micro"),
                "recall_micro": best_trial.user_attrs.get("recall_micro"),
            }
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(best_config, f, indent=2, ensure_ascii=False)

        # Save all trials as CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "trial_number",
                "temperature",
                "top_p",
                "top_k",
                "presence_penalty",
                "score",
                "f1_macro",
                "f1_micro",
                "precision_macro",
                "recall_macro",
                "precision_micro",
                "recall_micro",
                "time_sec",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for trial in study.trials:
                writer.writerow({
                    "trial_number": trial.number,
                    "temperature": trial.params.get("temperature"),
                    "top_p": trial.params.get("top_p"),
                    "top_k": trial.params.get("top_k"),
                    "presence_penalty": trial.params.get("presence_penalty"),
                    "score": trial.value,
                    "f1_macro": trial.user_attrs.get("f1_macro"),
                    "f1_micro": trial.user_attrs.get("f1_micro"),
                    "precision_macro": trial.user_attrs.get("precision_macro"),
                    "recall_macro": trial.user_attrs.get("recall_macro"),
                    "precision_micro": trial.user_attrs.get("precision_micro"),
                    "recall_micro": trial.user_attrs.get("recall_micro"),
                    "time_sec": trial.user_attrs.get("time_sec"),
                })

        print(f"\nBest config saved: {json_path}")
        print(f"Summary CSV saved: {csv_path}")

        # Clean up model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_fd.close()


if __name__ == "__main__":
    main()
