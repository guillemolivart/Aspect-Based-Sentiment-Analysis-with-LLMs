import argparse
import csv
import json
import time
from pathlib import Path

from common import (
    ABSA_DIR,
    DEFAULT_MODEL_PATH,
    OUTPUT_DIR,
    encode,
    extract_json,
    generate_with_metadata,
    get_prompts,
    load_dataset,
    prepare_messages,
)
from stats import counts, prf
from zeroshot import load_model


DEFAULT_PROMPT_PATH = ABSA_DIR / "prompts" / "absa_v2.json"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "thinking"


def parse_args():
    parser = argparse.ArgumentParser(description="Thinking-mode ABSA probe")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--prompt-file", default=str(DEFAULT_PROMPT_PATH))
    parser.add_argument("--data", default="devel")
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--analysis-output", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--selection", choices=["head", "spaced"], default="spaced")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def generation_config(args):
    return {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
    }


def select_examples(examples, args):
    indexed = list(enumerate(examples))
    indexed = indexed[args.start :]
    if args.limit is None or args.limit >= len(indexed):
        return indexed
    if args.selection == "head":
        return indexed[: args.limit]

    if args.limit == 1:
        return [indexed[0]]
    step = (len(indexed) - 1) / (args.limit - 1)
    positions = [round(i * step) for i in range(args.limit)]
    return [indexed[position] for position in positions]


def default_paths(data_path, prompt_name, config):
    name = (
        f"qwen35_2b_think_{data_path.stem}_{prompt_name}"
        f"_pp{config['presence_penalty']}_m{config['max_new_tokens']}.pilot"
    )
    return (
        DEFAULT_OUTPUT_DIR / f"{name}.json",
        DEFAULT_OUTPUT_DIR / f"{name}.summary.csv",
        DEFAULT_OUTPUT_DIR / f"{name}.analysis.md",
    )


def split_thinking_tokens(tokenizer, output_token_ids):
    end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    end_start = find_subsequence(output_token_ids, end_ids)
    if end_start == -1:
        return {
            "closed_thinking": False,
            "thinking_tokens": len(output_token_ids),
            "final_tokens": 0,
            "thinking_text": tokenizer.decode(output_token_ids, skip_special_tokens=True),
            "final_text": "",
        }

    thinking_ids = output_token_ids[:end_start]
    final_ids = output_token_ids[end_start + len(end_ids) :]
    return {
        "closed_thinking": True,
        "thinking_tokens": len(thinking_ids),
        "final_tokens": len(final_ids),
        "thinking_text": tokenizer.decode(thinking_ids, skip_special_tokens=True),
        "final_text": tokenizer.decode(final_ids, skip_special_tokens=True),
    }


def find_subsequence(values, pattern):
    if not pattern:
        return -1
    for index in range(0, len(values) - len(pattern) + 1):
        if values[index : index + len(pattern)] == pattern:
            return index
    return -1


def run_analysis(rows):
    totals = {
        "predicted": 0,
        "expected": 0,
        "ok": 0,
        "thinking_tokens": 0,
        "final_tokens": 0,
        "output_tokens": 0,
        "elapsed_sec": 0.0,
    }
    macro = [0.0, 0.0, 0.0]
    for row in rows:
        totals["predicted"] += row["predicted"]
        totals["expected"] += row["expected"]
        totals["ok"] += row["ok"]
        totals["thinking_tokens"] += row["thinking_tokens"]
        totals["final_tokens"] += row["final_tokens"]
        totals["output_tokens"] += row["output_tokens"]
        totals["elapsed_sec"] += row["elapsed_sec"]
        macro[0] += row["precision"]
        macro[1] += row["recall"]
        macro[2] += row["f1"]

    n = len(rows) or 1
    micro = prf(totals["predicted"], totals["expected"], totals["ok"])
    return {
        "n": len(rows),
        "macro": [value / n for value in macro],
        "micro": micro,
        **totals,
        "closed_thinking": sum(row["closed_thinking"] for row in rows),
        "hit_token_limit": sum(row["hit_token_limit"] for row in rows),
        "unfinished_at_limit": sum(row["unfinished_at_limit"] for row in rows),
    }


def write_summary(path, rows):
    fieldnames = [
        "index",
        "id",
        "language",
        "prompt_tokens",
        "output_tokens",
        "thinking_tokens",
        "final_tokens",
        "closed_thinking",
        "hit_token_limit",
        "unfinished_at_limit",
        "ended_with_eos",
        "elapsed_sec",
        "predicted",
        "expected",
        "ok",
        "precision",
        "recall",
        "f1",
    ]
    with open(path, "w", encoding="utf-8", newline="") as summary_fd:
        writer = csv.DictWriter(summary_fd, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_analysis(path, output_path, summary_path, config, selected_count, stats):
    lines = [
        "# Thinking Probe: Qwen3.5-2B",
        "",
        "## Run",
        "",
        f"- Output JSON: `{output_path}`",
        f"- Summary CSV: `{summary_path}`",
        f"- Examples: {selected_count}",
        f"- Generation config: `{config}`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| M.avg P/R/F1 | {stats['macro'][0]:.1f} / {stats['macro'][1]:.1f} / {stats['macro'][2]:.1f} |",
        f"| m.avg P/R/F1 | {stats['micro'][0]:.1f} / {stats['micro'][1]:.1f} / {stats['micro'][2]:.1f} |",
        f"| predicted / gold / ok | {stats['predicted']} / {stats['expected']} / {stats['ok']} |",
        f"| closed thinking | {stats['closed_thinking']} / {stats['n']} |",
        f"| hit max_new_tokens | {stats['hit_token_limit']} / {stats['n']} |",
        f"| unfinished at token limit | {stats['unfinished_at_limit']} / {stats['n']} |",
        f"| total output tokens | {stats['output_tokens']} |",
        f"| total thinking tokens | {stats['thinking_tokens']} |",
        f"| total final tokens | {stats['final_tokens']} |",
        f"| total runtime seconds | {stats['elapsed_sec']:.1f} |",
        "",
        "## Interpretation",
        "",
        "- `closed_thinking` means the generated output contained `</think>`.",
        "- `hit max_new_tokens` means generation consumed the full configured budget without EOS.",
        "- `unfinished at token limit` is the first simple loop-risk flag: it hit the token budget before closing `</think>`.",
        "- This is intentionally not a loop detector yet; it only measures whether thinking terminates under the chosen budget.",
    ]
    with open(path, "w", encoding="utf-8") as analysis_fd:
        analysis_fd.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    config = generation_config(args)
    prompts = get_prompts(args.prompt_file)
    examples, data_path = load_dataset(args.data)
    selected = select_examples(examples, args)

    output_path, summary_path, analysis_path = default_paths(
        data_path, prompts["name"], config
    )
    output_path = Path(args.output) if args.output else output_path
    summary_path = Path(args.summary_output) if args.summary_output else summary_path
    analysis_path = Path(args.analysis_output) if args.analysis_output else analysis_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)

    print("========= ABSA THINKING PROBE =========")
    print(f"model_path={args.model_path}")
    print(f"prompt_file={prompts['path']}")
    print(f"data={data_path}")
    print(f"generation={config}")
    print(f"selection={args.selection}")
    print(f"examples={len(selected)}")

    model, tokenizer = load_model(args.model_path, args.load_in_4bit)

    processed = []
    summary_rows = []
    for index, example in selected:
        print(f"Processing example {index}: {example['id']}", flush=True)
        messages = prepare_messages(prompts, example)
        model_inputs = encode(tokenizer, messages, enable_thinking=True)

        t0 = time.time()
        generation = generate_with_metadata(model, tokenizer, model_inputs, config)
        elapsed = time.time() - t0
        split = split_thinking_tokens(tokenizer, generation["output_token_ids"])

        prediction = (
            extract_json(generation["text_with_special_tokens"])
            if split["closed_thinking"]
            else {}
        )
        predicted, expected, ok = counts(prediction, example["gold"])
        precision, recall, f1 = prf(predicted, expected, ok)
        unfinished_at_limit = generation["hit_token_limit"] and not split[
            "closed_thinking"
        ]

        result = dict(example)
        result["prediction"] = prediction
        result["raw_generation"] = generation["text"]
        result["raw_generation_with_special_tokens"] = generation[
            "text_with_special_tokens"
        ]
        result["thinking_metadata"] = {
            "prompt_tokens": generation["prompt_tokens"],
            "output_tokens": generation["output_tokens"],
            "thinking_tokens": split["thinking_tokens"],
            "final_tokens": split["final_tokens"],
            "closed_thinking": split["closed_thinking"],
            "hit_token_limit": generation["hit_token_limit"],
            "unfinished_at_limit": unfinished_at_limit,
            "ended_with_eos": generation["ended_with_eos"],
            "elapsed_sec": elapsed,
        }
        processed.append(result)

        summary_rows.append(
            {
                "index": index,
                "id": example["id"],
                "language": example.get("language", "unknown"),
                "prompt_tokens": generation["prompt_tokens"],
                "output_tokens": generation["output_tokens"],
                "thinking_tokens": split["thinking_tokens"],
                "final_tokens": split["final_tokens"],
                "closed_thinking": split["closed_thinking"],
                "hit_token_limit": generation["hit_token_limit"],
                "unfinished_at_limit": unfinished_at_limit,
                "ended_with_eos": generation["ended_with_eos"],
                "elapsed_sec": elapsed,
                "predicted": predicted,
                "expected": expected,
                "ok": ok,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    with open(output_path, "w", encoding="utf-8") as output_fd:
        json.dump(processed, output_fd, indent=3, ensure_ascii=False)
    write_summary(summary_path, summary_rows)
    stats = run_analysis(summary_rows)
    write_analysis(analysis_path, output_path, summary_path, config, len(selected), stats)

    print("Done")
    print(f"Output: {output_path}")
    print(f"Summary: {summary_path}")
    print(f"Analysis: {analysis_path}")
    print(
        f"M.avg P/R/F1: {stats['macro'][0]:.1f} "
        f"{stats['macro'][1]:.1f} {stats['macro'][2]:.1f}"
    )
    print(
        f"m.avg P/R/F1: {stats['micro'][0]:.1f} "
        f"{stats['micro'][1]:.1f} {stats['micro'][2]:.1f}"
    )
    print(
        f"closed={stats['closed_thinking']}/{stats['n']} "
        f"hit_limit={stats['hit_token_limit']}/{stats['n']} "
        f"unfinished_at_limit={stats['unfinished_at_limit']}/{stats['n']}"
    )


if __name__ == "__main__":
    main()
