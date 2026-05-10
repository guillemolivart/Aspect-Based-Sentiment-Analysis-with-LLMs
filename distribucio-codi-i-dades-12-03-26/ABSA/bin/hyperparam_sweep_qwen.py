import argparse
import csv
import itertools
import json
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
    resolve_path,
)


def parse_list(arg, cast=float):
    if arg is None:
        return [None]
    parts = [p.strip() for p in arg.split(",") if p.strip() != ""]
    out = []
    for p in parts:
        if p.lower() == "none":
            out.append(None)
        else:
            out.append(cast(p))
    return out


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


def default_output_name(prefix):
    OUT = OUTPUT_DIR / "hyperparam_sweep"
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / f"{prefix}.json"


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for Qwen (no thinking)")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--data", default="devel")
    parser.add_argument("--temperatures", default="1.0", help="comma-separated temperatures or 'none'")
    parser.add_argument("--top-ps", default="1.0", help="comma-separated top_p values or 'none'")
    parser.add_argument("--top-ks", default="20", help="comma-separated top_k ints or 'none'")
    parser.add_argument("--min-ps", default="0.0", help="comma-separated min_p values or 'none'")
    parser.add_argument("--presence-penalties", default="2.0", help="comma-separated presence penalties or 'none'")
    parser.add_argument("--repetition-penalties", default="1.0", help="comma-separated repetition penalties or 'none'")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output-prefix", default="results")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--add-neighbors", action="store_true", help="also add neighbor values around recommended config")
    parser.add_argument("--min-improvement", type=float, default=0.5, help="minimum F1 (pp) improvement over baseline to accept new config")
    parser.add_argument("--drop-tolerance", type=float, default=0.5, help="if best config drops more than this vs baseline, keep baseline")
    args = parser.parse_args()

    temps = parse_list(args.temperatures, float)
    top_ps = parse_list(args.top_ps, float)
    top_ks = parse_list(args.top_ks, lambda x: int(float(x)))
    min_ps = parse_list(args.min_ps, float)
    pps = parse_list(args.presence_penalties, float)
    rps = parse_list(args.repetition_penalties, float)

    prompts = get_prompts(args.prompt_file)
    examples, data_path = load_dataset(args.data)
    if args.limit is not None:
        selected = examples[args.start : args.start + args.limit]
    else:
        selected = examples[args.start :]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_path = Path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )

    model_kwargs = {"device_map": "auto"}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
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

    results = []
    total_runs = (
        len(temps)
        * len(top_ps)
        * len(top_ks)
        * len(min_ps)
        * len(pps)
        * len(rps)
    )
    run_index = 0

    # recommended (zeroshot) defaults from zeroshot.py for no-thinking mode
    baseline_cfg = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 2.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": args.max_new_tokens,
    }

    # optionally add neighbor values around baseline to the sweep
    if args.add_neighbors:
        # simple neighbor deltas (can be tuned)
        temp_neighbors = [baseline_cfg["temperature"] + d for d in (-0.5, -0.2, 0.2, 0.5)]
        top_p_neighbors = [baseline_cfg["top_p"] + d for d in (-0.1, -0.05, 0.05, 0.1)]
        top_k_neighbors = [baseline_cfg["top_k"] + d for d in (-10, -5, 5, 10)]
        min_p_neighbors = [baseline_cfg["min_p"] + d for d in (0.0, 0.01, 0.05)]
        pp_neighbors = [baseline_cfg["presence_penalty"] + d for d in (-0.5, -0.2, 0.2, 0.5)]
        rp_neighbors = [baseline_cfg["repetition_penalty"] + d for d in (-0.5, -0.2, 0.2, 0.5)]

        # merge neighbors into lists
        for v in temp_neighbors:
            if v is not None and v not in temps:
                temps.append(v)
        for v in top_p_neighbors:
            if v is not None and v not in top_ps:
                top_ps.append(v)
        for v in top_k_neighbors:
            if v is not None and v not in top_ks:
                top_ks.append(int(v))
        for v in min_p_neighbors:
            if v is not None and v not in min_ps:
                min_ps.append(v)
        for v in pp_neighbors:
            if v is not None and v not in pps:
                pps.append(v)
        for v in rp_neighbors:
            if v is not None and v not in rps:
                rps.append(v)

        # recalc total runs
        total_runs = (
            len(temps)
            * len(top_ps)
            * len(top_ks)
            * len(min_ps)
            * len(pps)
            * len(rps)
        )

    for temp, top_p, top_k, min_p, pp, rp in itertools.product(
        temps, top_ps, top_ks, min_ps, pps, rps
    ):
        run_index += 1
        generation_config = {
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": pp,
            "repetition_penalty": rp,
            "max_new_tokens": args.max_new_tokens,
        }

        print(f"Run {run_index}/{total_runs}: {generation_config}")
        t0 = time.time()

        processed = []
        predicted_total = expected_total = ok_total = 0
        precision_macro = recall_macro = f1_macro = 0.0

        for index, example in enumerate(selected, start=args.start):
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

            out = dict(example)
            out["prediction"] = prediction
            out["raw_generation"] = gen_text
            processed.append(out)

        n = len(selected)
        if n:
            precision_macro /= n
            recall_macro /= n
            f1_macro /= n
        precision_micro, recall_micro, f1_micro = prf(predicted_total, expected_total, ok_total)

        elapsed = time.time() - t0
        sec_per_example = elapsed / len(processed) if processed else 0.0

        summary = {
            "generation_config": generation_config,
            "examples": len(processed),
            "time_sec": elapsed,
            "sec_per_example": sec_per_example,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_micro": precision_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
        }

        out_path = default_output_name(f"{args.output_prefix}.run{run_index}")
        with open(out_path, "w", encoding="utf-8") as fd:
            json.dump({"summary": summary, "examples": processed}, fd, ensure_ascii=False, indent=2)

        results.append(summary)

    # decide conservative best: compare to baseline
    baseline_summary = None
    for s in results:
        cfg = s["generation_config"]
        # match baseline by equality for main fields
        is_baseline = (
            cfg.get("temperature") == baseline_cfg["temperature"]
            and cfg.get("top_p") == baseline_cfg["top_p"]
            and cfg.get("top_k") == baseline_cfg["top_k"]
            and cfg.get("min_p") == baseline_cfg["min_p"]
            and cfg.get("presence_penalty") == baseline_cfg["presence_penalty"]
            and cfg.get("repetition_penalty") == baseline_cfg["repetition_penalty"]
        )
        if is_baseline:
            baseline_summary = s
            break

    # find best by micro F1
    best = max(results, key=lambda x: x.get("f1_micro", -1)) if results else None
    chosen = None
    if baseline_summary is None:
        chosen = best
    else:
        base_f1 = baseline_summary.get("f1_micro", 0.0)
        best_f1 = best.get("f1_micro", 0.0)
        if best_f1 < base_f1 - args.drop_tolerance:
            chosen = baseline_summary
        elif best_f1 >= base_f1 + args.min_improvement:
            chosen = best
        else:
            chosen = baseline_summary

    chosen_path = OUTPUT_DIR / "hyperparam_sweep" / f"{args.output_prefix}.chosen.json"
    with open(chosen_path, "w", encoding="utf-8") as fd:
        json.dump({"baseline": baseline_cfg, "chosen": chosen, "all": results}, fd, ensure_ascii=False, indent=2)

    print("Sweep complete")
    print(f"Summary CSV: {csv_path}")
    print(f"Chosen configuration saved: {chosen_path}")

    # write CSV summary
    csv_path = OUTPUT_DIR / "hyperparam_sweep" / f"{args.output_prefix}.summary.csv"
    with open(csv_path, "w", newline='', encoding="utf-8") as csvfd:
        writer = csv.writer(csvfd)
        header = [
            "run",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "presence_penalty",
            "repetition_penalty",
            "examples",
            "f1_micro",
            "f1_macro",
            "sec_per_example",
        ]
        writer.writerow(header)
        for i, s in enumerate(results, start=1):
            cfg = s["generation_config"]
            writer.writerow(
                [
                    i,
                    cfg.get("temperature"),
                    cfg.get("top_p"),
                    cfg.get("top_k"),
                    cfg.get("min_p"),
                    cfg.get("presence_penalty"),
                    cfg.get("repetition_penalty"),
                    s["examples"],
                    round(s["f1_micro"], 3),
                    round(s["f1_macro"], 3),
                    round(s["sec_per_example"], 3),
                ]
            )

    print("Sweep complete")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
