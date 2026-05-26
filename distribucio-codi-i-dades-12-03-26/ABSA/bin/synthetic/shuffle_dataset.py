#!/usr/bin/env python3
"""Shuffle a JSON dataset (top-level list) with a reproducible seed."""

import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Shuffle a JSON dataset list")
    parser.add_argument(
        "--input",
        default="dataset/train_plus_synthetic_v1.json",
        help="Path to input JSON file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output JSON file. If omitted, writes <input>.shuffled.json",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    if input_path.suffix == ".json":
        return input_path.with_name(f"{input_path.stem}.shuffled.json")
    return input_path.with_name(f"{input_path.name}.shuffled.json")


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.inplace:
        output_path = input_path
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = default_output_path(input_path)

    with input_path.open("r", encoding="utf-8") as fd:
        data = json.load(fd)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a top-level list")

    random.Random(args.seed).shuffle(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fd:
        json.dump(data, fd, ensure_ascii=False, indent=2)
        fd.write("\n")

    print(
        f"Shuffled {len(data)} rows with seed={args.seed}. "
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
