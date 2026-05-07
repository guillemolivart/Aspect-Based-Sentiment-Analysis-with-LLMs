import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import OUTPUT_DIR, load_dataset
from stats import counts, prf


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ABSA frequency baselines")
    parser.add_argument("--train", default="train")
    parser.add_argument("--data", default="devel")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "baselines"))
    parser.add_argument("--top-k", default="1,2,3,4,5,8,11")
    return parser.parse_args()


def aspect_frequencies(examples):
    counter = Counter()
    for example in examples:
        counter.update(example["gold"].keys())
    return counter


def majority_polarity_by_aspect(examples):
    by_aspect = defaultdict(Counter)
    for example in examples:
        for aspect, polarity in example["gold"].items():
            by_aspect[aspect][polarity] += 1
    return {
        aspect: counts.most_common(1)[0][0]
        for aspect, counts in by_aspect.items()
    }


def evaluate(examples):
    predicted_total = expected_total = ok_total = 0
    macro_precision = macro_recall = macro_f1 = 0.0

    for example in examples:
        predicted, expected, ok = counts(example["prediction"], example["gold"])
        predicted_total += predicted
        expected_total += expected
        ok_total += ok

        precision, recall, f1 = prf(predicted, expected, ok)
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    n = len(examples)
    macro_precision /= n
    macro_recall /= n
    macro_f1 /= n
    micro_precision, micro_recall, micro_f1 = prf(
        predicted_total, expected_total, ok_total
    )

    return {
        "P_macro": macro_precision,
        "R_macro": macro_recall,
        "F1_macro": macro_f1,
        "P_micro": micro_precision,
        "R_micro": micro_recall,
        "F1_micro": micro_f1,
        "predicted": predicted_total,
        "expected": expected_total,
        "ok": ok_total,
    }


def with_prediction(examples, prediction_fn):
    output = []
    for example in examples:
        item = dict(example)
        item["prediction"] = prediction_fn(example)
        output.append(item)
    return output


def write_json(path, examples):
    with open(path, "w", encoding="utf-8") as output_fd:
        json.dump(examples, output_fd, indent=3, ensure_ascii=False)


def main():
    args = parse_args()
    train, train_path = load_dataset(args.train)
    data, data_path = load_dataset(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top_k_values = [int(value) for value in args.top_k.split(",") if value.strip()]
    top_aspects = [aspect for aspect, _ in aspect_frequencies(train).most_common()]
    majority_polarity = majority_polarity_by_aspect(train)

    baselines = {
        "empty": lambda _: {},
        "restaurant_general_positive": lambda _: {"restaurant_general": "positive"},
    }

    for k in top_k_values:
        aspects = top_aspects[:k]
        baselines[f"top{k}_positive"] = (
            lambda _, aspects=aspects: {aspect: "positive" for aspect in aspects}
        )
        baselines[f"top{k}_majority"] = (
            lambda _, aspects=aspects: {
                aspect: majority_polarity[aspect] for aspect in aspects
            }
        )

    # Diagnostic only: this uses gold aspect presence and is not a deployable baseline.
    baselines["oracle_aspects_majority_polarity"] = (
        lambda example: {
            aspect: majority_polarity.get(aspect, "positive")
            for aspect in example["gold"].keys()
        }
    )

    summary_rows = []
    for name, prediction_fn in baselines.items():
        examples = with_prediction(data, prediction_fn)
        output_path = output_dir / f"{name}.{data_path.stem}.json"
        write_json(output_path, examples)
        score = evaluate(examples)
        summary_rows.append(
            {
                "baseline": name,
                "data": str(data_path),
                "train": str(train_path),
                "output": str(output_path),
                **score,
            }
        )

    summary_rows.sort(key=lambda row: row["F1_macro"], reverse=True)
    summary_path = output_dir / f"summary.{data_path.stem}.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as output_fd:
        writer = csv.DictWriter(output_fd, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote baseline predictions to {output_dir}")
    print(f"Wrote summary to {summary_path}")
    print("baseline                             Pm    Rm    Fm    Fmicro")
    for row in summary_rows:
        print(
            f"{row['baseline']:35s} "
            f"{row['P_macro']:5.1f} {row['R_macro']:5.1f} "
            f"{row['F1_macro']:5.1f} {row['F1_micro']:7.1f}"
        )


if __name__ == "__main__":
    main()
