#!/usr/bin/env python3
"""Analyze ABSA inference JSON and optionally save report txt and table csv."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
ABSA_DIR = PARENT_DIR.parent
OUTPUT_DIR = ABSA_DIR / "outputs"

if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from common import ASPECTS


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze inference JSON metrics")
    parser.add_argument(
        "input_json",
        nargs="?",
        default=str(OUTPUT_DIR / "finetune" / "best_devel_predictions_qlora.json"),
        help="Path to predictions JSON",
    )
    parser.add_argument(
        "--save-txt",
        default=None,
        help="Optional path to save text report",
    )
    parser.add_argument(
        "--save-csv",
        default=None,
        help="Optional path to save aspect table csv",
    )
    return parser.parse_args()


def build_report(json_path):
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as in_fd:
        data = json.load(in_fd)

    lines = []
    rows = []

    lines.append(f"ANALYSIS: {json_path.name}")
    lines.append("=" * 70)

    total = len(data)
    lines.append(f"Total examples: {total}")
    if total == 0:
        return lines, rows

    has_pred = sum(1 for ex in data if "prediction_normalized" in ex)
    lines.append(f"Examples with predictions: {has_pred}/{total}")

    exact_matches = 0
    aspect_precision = defaultdict(lambda: {"correct": 0, "total": 0})
    aspect_recall = defaultdict(lambda: {"correct": 0, "total": 0})

    missing_aspects = 0
    hallucinated = 0
    wrong_polarity = 0

    for ex in data:
        gold = ex.get("gold", {})
        pred = ex.get("prediction_normalized", ex.get("prediction", {}))

        if pred == gold:
            exact_matches += 1

        if isinstance(gold, dict) and isinstance(pred, dict):
            for aspect in ASPECTS:
                if aspect in gold:
                    aspect_recall[aspect]["total"] += 1
                    if aspect in pred and pred[aspect] == gold[aspect]:
                        aspect_recall[aspect]["correct"] += 1

                if aspect in pred:
                    aspect_precision[aspect]["total"] += 1
                    if aspect in gold and pred[aspect] == gold[aspect]:
                        aspect_precision[aspect]["correct"] += 1

            if not pred and gold:
                missing_aspects += 1
            elif pred and not gold:
                hallucinated += 1
            else:
                for asp in set(gold.keys()) & set(pred.keys()):
                    if gold[asp] != pred[asp]:
                        wrong_polarity += 1
                        break

    lines.append(f"Exact matches (prediction == gold): {exact_matches}/{total}")
    if has_pred > 0:
        lines.append(f"Exact match accuracy: {100 * exact_matches / has_pred:.2f}%")

    lines.append("")
    lines.append(f"{'Aspect':<25} {'Precision':<18} {'Recall':<18}")
    lines.append("-" * 70)

    for aspect in sorted(ASPECTS):
        prec = aspect_precision[aspect]
        rec = aspect_recall[aspect]

        prec_pct = (100 * prec["correct"] / prec["total"]) if prec["total"] else 0.0
        rec_pct = (100 * rec["correct"] / rec["total"]) if rec["total"] else 0.0

        lines.append(
            f"{aspect:<25} {prec_pct:>6.2f}% ({prec['correct']}/{prec['total']:<4}) {rec_pct:>6.2f}% ({rec['correct']}/{rec['total']:<4})"
        )
        rows.append(
            {
                "aspect": aspect,
                "precision_pct": f"{prec_pct:.2f}",
                "precision_correct": prec["correct"],
                "precision_total": prec["total"],
                "recall_pct": f"{rec_pct:.2f}",
                "recall_correct": rec["correct"],
                "recall_total": rec["total"],
            }
        )

    lines.append("")
    lines.append("ERROR ANALYSIS")
    lines.append("-" * 70)
    lines.append(f"Empty predictions (gold != empty): {missing_aspects}")
    lines.append(f"Hallucinated predictions (pred != empty, gold empty): {hallucinated}")
    lines.append(f"Wrong polarity (right aspect, wrong sentiment): {wrong_polarity}")

    lines.append("")
    lines.append("SAMPLE ERRORS")
    lines.append("-" * 70)
    error_count = 0
    for ex in data:
        gold = ex.get("gold", {})
        pred = ex.get("prediction_normalized", ex.get("prediction", {}))
        if pred != gold and error_count < 3:
            text = ex.get("text", "")[:120]
            lines.append(f"[{error_count + 1}] id={ex.get('id', 'unknown')}")
            lines.append(f"Text: {text}...")
            lines.append(f"Gold: {gold}")
            lines.append(f"Pred: {pred}")
            lines.append("")
            error_count += 1

    return lines, rows


def main():
    args = parse_args()
    input_json = Path(args.input_json)

    if args.save_txt is None:
        txt_path = input_json.with_suffix(".analysis.txt")
    else:
        txt_path = Path(args.save_txt)

    if args.save_csv is None:
        csv_path = input_json.with_suffix(".analysis.csv")
    else:
        csv_path = Path(args.save_csv)

    lines, rows = build_report(input_json)

    report_text = "\n".join(lines)
    print(report_text)

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as txt_fd:
        txt_fd.write(report_text + "\n")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_fd:
        writer = csv.DictWriter(
            csv_fd,
            fieldnames=[
                "aspect",
                "precision_pct",
                "precision_correct",
                "precision_total",
                "recall_pct",
                "recall_correct",
                "recall_total",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved txt report to:", txt_path)
    print("Saved csv table to:", csv_path)


if __name__ == "__main__":
    main()
