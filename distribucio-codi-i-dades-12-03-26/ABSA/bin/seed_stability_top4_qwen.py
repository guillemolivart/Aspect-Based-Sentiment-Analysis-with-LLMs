#!/usr/bin/env python3
"""Run the top-4 ABSA configurations with 4 new seeds + original baseline.

Loads original results from AGGREGATED_RESULTS.csv, then runs 4 new experiments
with fixed seeds. Creates per-config tables with: Original + Runs 1-4 + Mean.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

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


TOP_CONFIGS = [
    {
        "name": "v6_F_refined_9_temp065_topp75",
        "temperature": 0.65,
        "top_p": 0.75,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 2.0,
        "repetition_penalty": 1.0,
    },
    {
        "name": "v6_F_refined_5_pp18",
        "temperature": 0.70,
        "top_p": 0.80,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.8,
        "repetition_penalty": 1.0,
    },
    {
        "name": "v6_F_cons_off_pres",
        "temperature": 0.70,
        "top_p": 0.80,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 2.0,
        "repetition_penalty": 1.0,
    },
    {
        "name": "v6_F_refined_7_topp75",
        "temperature": 0.70,
        "top_p": 0.75,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 2.0,
        "repetition_penalty": 1.0,
    },
]


DEFAULT_SEEDS = "22,33,44,55"


def counts(prediction, gold):
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
    precision = 100.0 * ok / predicted if predicted else 0.0
    recall = 100.0 * ok / expected if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_model(model_path, load_in_4bit=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    )

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


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


def evaluate_config(model, tokenizer, prompts, selected, config, seed, max_new_tokens):
    set_seed(seed)

    generation_config = {
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "top_k": config["top_k"],
        "min_p": config["min_p"],
        "presence_penalty": config["presence_penalty"],
        "repetition_penalty": config["repetition_penalty"],
        "max_new_tokens": max_new_tokens,
    }

    t0 = time.time()
    predicted_total = expected_total = ok_total = 0
    precision_macro = recall_macro = f1_macro = 0.0

    for example in selected:
        messages = prepare_messages(prompts, example)
        model_inputs = encode(tokenizer, messages, enable_thinking=False)
        gen_text = generate(model, tokenizer, model_inputs, generation_config)
        prediction = extract_json(gen_text)

        predicted, expected, ok = counts(prediction, example["gold"])
        predicted_total += predicted
        expected_total += expected
        ok_total += ok

        precision, recall, f1 = prf(predicted, expected, ok)
        precision_macro += precision
        recall_macro += recall
        f1_macro += f1

    n = len(selected)
    if n:
        precision_macro /= n
        recall_macro /= n
        f1_macro /= n

    precision_micro, recall_micro, f1_micro = prf(predicted_total, expected_total, ok_total)

    elapsed = time.time() - t0
    sec_per_example = elapsed / n if n else 0.0

    return {
        "seed": seed,
        "examples": n,
        "time_sec": elapsed,
        "sec_per_example": sec_per_example,
        "temperature": generation_config["temperature"],
        "top_p": generation_config["top_p"],
        "top_k": generation_config["top_k"],
        "min_p": generation_config["min_p"],
        "presence_penalty": generation_config["presence_penalty"],
        "repetition_penalty": generation_config["repetition_penalty"],
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
    }


def write_config_table(output_path, config_name, config, rows):
    numeric_keys = [
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
        "f1_micro",
        "f1_macro",
        "precision_macro",
        "recall_macro",
        "precision_micro",
        "recall_micro",
        "sec_per_example",
    ]

    summary_rows = []
    for index, row in enumerate(rows):
        # Run number: 0 for original, 1-4 for new runs
        run_label = "original" if row["seed"] == "original" else str(index)
        summary_rows.append(
            {
                "Config": config_name,
                "Run": run_label,
                "Seed": row["seed"],
                "Temp": row["temperature"],
                "Top-P": row["top_p"],
                "Top-K": row["top_k"],
                "PP": row["presence_penalty"],
                "F1-Micro": row["f1_micro"],
                "F1-Macro": row["f1_macro"],
                "Prec-Macro": row["precision_macro"],
                "Rec-Macro": row["recall_macro"],
                "Prec-Micro": row["precision_micro"],
                "Rec-Micro": row["recall_micro"],
                "Sec/Example": row["sec_per_example"],
            }
        )

    mean_row = {key: 0.0 for key in numeric_keys}
    for key in numeric_keys:
        mean_row[key] = sum(row[key] for row in rows) / len(rows)

    summary_rows.append(
        {
            "Config": config_name,
            "Run": "mean",
            "Seed": "",
            "Temp": config["temperature"],
            "Top-P": config["top_p"],
            "Top-K": config["top_k"],
            "PP": config["presence_penalty"],
            "F1-Micro": mean_row["f1_micro"],
            "F1-Macro": mean_row["f1_macro"],
            "Prec-Macro": mean_row["precision_macro"],
            "Rec-Macro": mean_row["recall_macro"],
            "Prec-Micro": mean_row["precision_micro"],
            "Rec-Micro": mean_row["recall_micro"],
            "Sec/Example": mean_row["sec_per_example"],
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Config",
        "Run",
        "Seed",
        "Temp",
        "Top-P",
        "Top-K",
        "PP",
        "F1-Micro",
        "F1-Macro",
        "Prec-Macro",
        "Rec-Macro",
        "Prec-Micro",
        "Rec-Micro",
        "Sec/Example",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: format_value(row.get(key)) for key in fieldnames})

    return summary_rows[-1]


def main():
    parser = argparse.ArgumentParser(description="Repeat top-4 ABSA configs with new seeds + original baseline")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--data", default="devel")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="comma-separated 4 seed values")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "hyperparam_sweep" / "seed_stability_top4"))
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if len(seeds) != 4:
        raise ValueError("This experiment expects exactly 4 new seeds (original loaded from AGGREGATED_RESULTS.csv).")

    prompts = get_prompts(args.prompt_file)
    examples, data_path = load_dataset(args.data)
    if args.limit is not None:
        selected = examples[args.start : args.start + args.limit]
    else:
        selected = examples[args.start :]

    model, tokenizer = load_model(args.model_path, args.load_in_4bit)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original results from AGGREGATED_RESULTS.csv
    aggregated_path = OUTPUT_DIR / "hyperparam_sweep" / "AGGREGATED_RESULTS.csv"
    original_results = {}
    if aggregated_path.exists():
        with open(aggregated_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                config_name = row["Config"].strip()
                original_results[config_name] = {
                    "f1_micro": float(row["F1-Micro"]),
                    "f1_macro": float(row["F1-Macro"]),
                    "precision_macro": float(row["Prec-Macro"]),
                    "recall_macro": float(row["Rec-Macro"]),
                    "precision_micro": float(row["Prec-Micro"]),
                    "recall_micro": float(row["Rec-Micro"]),
                }
    else:
        print(f"WARNING: {aggregated_path} not found. Running without original baseline.")

    print("========= SEED STABILITY EXPERIMENT =========")
    print(f"NOTE: Original results loaded from AGGREGATED_RESULTS.csv")
    print(f"      Running 4 new experiments with fixed seeds.")
    print(f"model_path={args.model_path}")
    print(f"prompt_file={prompts['path']}")
    print(f"data={data_path}")
    print(f"examples={len(selected)}")
    print(f"new_seeds={seeds}")
    print(f"output_dir={output_dir}")
    print("=" * 60)

    combined_rows = []
    for config in TOP_CONFIGS:
        print(f"\n=== {config['name']} ===")
        config_rows = []

        # Add original result as run 0
        if config["name"] in original_results:
            orig = original_results[config["name"]]
            original_row = {
                "seed": "original",
                "examples": len(selected),
                "time_sec": 0.0,
                "sec_per_example": 0.0,
                "temperature": config["temperature"],
                "top_p": config["top_p"],
                "top_k": config["top_k"],
                "min_p": config["min_p"],
                "presence_penalty": config["presence_penalty"],
                "repetition_penalty": config["repetition_penalty"],
                "f1_micro": orig["f1_micro"],
                "f1_macro": orig["f1_macro"],
                "precision_macro": orig["precision_macro"],
                "recall_macro": orig["recall_macro"],
                "precision_micro": orig["precision_micro"],
                "recall_micro": orig["recall_micro"],
            }
            config_rows.append(original_row)
            print(f"Run 0 (original): F1-Micro={orig['f1_micro']:.2f} | F1-Macro={orig['f1_macro']:.2f}")
        else:
            print(f"WARNING: Original result for {config['name']} not found in AGGREGATED_RESULTS.csv")

        # Run 4 new experiments with fixed seeds
        for run_num, seed in enumerate(seeds, start=1):
            print(f"Run {run_num}/4: seed {seed} ...")
            row = evaluate_config(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                selected=selected,
                config=config,
                seed=seed,
                max_new_tokens=args.max_new_tokens,
            )
            config_rows.append(row)
            print(
                f"  F1-Micro={row['f1_micro']:.2f} | F1-Macro={row['f1_macro']:.2f} | "
                f"Prec-Micro={row['precision_micro']:.2f} | Rec-Micro={row['recall_micro']:.2f}"
            )

        table_path = output_dir / f"{config['name']}.csv"
        mean_row = write_config_table(table_path, config["name"], config, config_rows)
        combined_rows.append(mean_row)
        print(f"Saved table to {table_path}")

    combined_path = output_dir / "seed_stability_overview.csv"
    with open(combined_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Config", "Temp", "Top-P", "Top-K", "PP",
            "F1-Micro-mean", "F1-Macro-mean", "Prec-Macro-mean", "Rec-Macro-mean", "Prec-Micro-mean", "Rec-Micro-mean",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in combined_rows:
            writer.writerow({
                "Config": row["Config"], "Temp": row["Temp"], "Top-P": row["Top-P"],
                "Top-K": row["Top-K"], "PP": row["PP"],
                "F1-Micro-mean": row["F1-Micro"], "F1-Macro-mean": row["F1-Macro"],
                "Prec-Macro-mean": row["Prec-Macro"], "Rec-Macro-mean": row["Rec-Macro"],
                "Prec-Micro-mean": row["Prec-Micro"], "Rec-Micro-mean": row["Rec-Micro"],
            })

    print("\n========= SUMMARY =========")
    for row in combined_rows:
        print(f"{row['Config']:32s} | F1-Micro={row['F1-Micro']:.2f} | F1-Macro={row['F1-Macro']:.2f}")
    print(f"Overview saved to: {combined_path}")


if __name__ == "__main__":
    main()