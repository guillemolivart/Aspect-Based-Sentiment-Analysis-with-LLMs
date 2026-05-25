#!/usr/bin/env python3
"""Generate the final targeted synthetic ABSA dataset with OpenAI API calls.

This script creates:
  - dataset/synthetic_v1.json
  - dataset/train_plus_synthetic_v1.json
  - outputs/synthetic/synthetic_v1.* diagnostic files

API key lookup order:
  1. --api-key
  2. OPENAI_API_KEY
  3. ABSA/.openai_api_key
"""

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from common import ABSA_DIR, ASPECTS, DATASET_DIR, OUTPUT_DIR, POLARITIES


BUCKET_QUOTAS = {
    "price_value_boundary": 56,
    "food_style_options_boundary": 36,
    "ambience_location_service_boundary": 30,
    "drinks_rare_aspects": 24,
    "neutral_conflict_polarity": 40,
    "hard_negatives_empty": 14,
    "natural_regularizers": 20,
}

BUCKET_SHORT = {
    "price_value_boundary": "price",
    "food_style_options_boundary": "foodstyle",
    "ambience_location_service_boundary": "ambience",
    "drinks_rare_aspects": "drinks",
    "neutral_conflict_polarity": "polarity",
    "hard_negatives_empty": "negative",
    "natural_regularizers": "regular",
}

ASPECT_DEFINITIONS = {
    "restaurant_general": "overall opinion about the restaurant as a whole, recommendation, satisfaction, disappointment, intention to return, or global experience",
    "restaurant_prices": "opinion about the restaurant's prices or value overall",
    "food_quality": "opinion about food taste, freshness, cooking, ingredients, or quality",
    "food_prices": "opinion about food/menu/dish prices or food value specifically",
    "food_style_options": "opinion about food presentation, originality, variety, menu options, portion size, or dietary options",
    "drinks_quality": "opinion about drink quality",
    "drinks_prices": "opinion about drink prices or drink value",
    "drinks_style_options": "opinion about drink variety, presentation, or options such as wine list, beer selection, cocktails, coffee options",
    "ambience": "opinion about atmosphere, decor, noise, music, comfort, cleanliness, crowding, or room feel",
    "service": "opinion about staff, waiters, attention, speed, friendliness, booking, seating, or treatment",
    "location": "opinion about location, views, parking, access, neighborhood, terrace, or surroundings",
}

PRIORITY_ASPECTS = {
    "restaurant_prices": 2.5,
    "food_prices": 2.0,
    "food_style_options": 2.0,
    "ambience": 1.8,
    "drinks_style_options": 2.5,
    "drinks_quality": 1.5,
    "location": 1.2,
    "restaurant_general": 1.0,
    "food_quality": 1.0,
    "service": 0.6,
}

FORBIDDEN_TEXT_MARKERS = [
    "aspect:",
    "sentiment:",
    "gold:",
    "json",
    "label",
    "annotation",
    "polarity",
    "restaurant_general",
    "food_quality",
    "food_style_options",
    "drinks_style_options",
]

OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


@dataclass
class GenerationPlan:
    plan_id: str
    bucket: str
    language: str
    target_gold: dict
    difficulty: str
    avoid_aspects: list
    style_seed_ids: list
    min_words: int = 15
    max_words: int = 140


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate targeted synthetic ABSA examples and train+synthetic dataset"
    )
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-key-file", default=str(ABSA_DIR / ".openai_api_key"))
    parser.add_argument("--endpoint", default=OPENAI_CHAT_COMPLETIONS_URL)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-completion-tokens", type=int, default=900)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-multiplier", type=float, default=1.45)
    parser.add_argument("--max-candidates", type=int, default=430)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true", help="Only build plans and diagnostics; do not call the API")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing synthetic output files")
    parser.add_argument("--resume", action="store_true", help="Resume from accepted metadata if present")
    parser.add_argument("--train-file", default=str(DATASET_DIR / "train.json"))
    parser.add_argument("--devel-file", default=str(DATASET_DIR / "devel.json"))
    parser.add_argument(
        "--devel-predictions",
        default=str(OUTPUT_DIR / "finetune" / "best_devel_predictions_qlora.json"),
        help="Current best model predictions on devel for aggregate error targeting",
    )
    parser.add_argument(
        "--train-predictions",
        default=str(OUTPUT_DIR / "finetune" / "best_train_predictions_qlora.json"),
        help="Optional current best model predictions on train. If missing, rare-pair train scoring is used.",
    )
    parser.add_argument(
        "--run-train-inference",
        action="store_true",
        help="If train predictions are missing, run the current QLoRA model on train before generating synthetic data.",
    )
    parser.add_argument(
        "--weights",
        default=str(
            OUTPUT_DIR
            / "finetune"
            / "FT.train.absa_v6.qlora4bit.simple.all-linear.r16.lr1e-4.generative_f1.weights"
        ),
        help="QLoRA weights used when --run-train-inference is enabled.",
    )
    parser.add_argument(
        "--prompt-file",
        default=str(ABSA_DIR / "prompts" / "absa_v6.json"),
        help="Prompt file used when --run-train-inference is enabled.",
    )
    parser.add_argument("--synthetic-output", default=str(DATASET_DIR / "synthetic_v1.json"))
    parser.add_argument("--combined-output", default=str(DATASET_DIR / "train_plus_synthetic_v1.json"))
    parser.add_argument(
        "--metadata-output",
        default=str(OUTPUT_DIR / "synthetic" / "synthetic_v1.accepted_with_metadata.json"),
    )
    parser.add_argument(
        "--rejected-output",
        default=str(OUTPUT_DIR / "synthetic" / "synthetic_v1.rejected.jsonl"),
    )
    parser.add_argument(
        "--plans-output",
        default=str(OUTPUT_DIR / "synthetic" / "synthetic_v1.plans.json"),
    )
    parser.add_argument(
        "--report-output",
        default=str(OUTPUT_DIR / "synthetic" / "synthetic_v1.report.json"),
    )
    return parser.parse_args()


def load_json(path):
    with open(path, encoding="utf-8") as fd:
        return json.load(fd)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fd:
        json.dump(data, fd, indent=2, ensure_ascii=False)


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fd:
        for row in rows:
            fd.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_api_key(args):
    if args.api_key:
        return args.api_key.strip()
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"].strip()
    key_path = Path(args.api_key_file)
    if key_path.exists():
        return key_path.read_text(encoding="utf-8").strip()
    raise RuntimeError(
        "OpenAI API key not found. Set OPENAI_API_KEY, pass --api-key, "
        f"or write it to {key_path}."
    )


def counter_to_jsonable(counter):
    return [
        {"key": list(key) if isinstance(key, tuple) else key, "count": count}
        for key, count in counter.most_common()
    ]


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def word_count(text):
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def ngrams(text, n=5):
    words = re.findall(r"\b[\w'-]+\b", normalize_text(text), flags=re.UNICODE)
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(xs, ys):
    if not xs or not ys:
        return 0.0
    return len(xs & ys) / len(xs | ys)


def span_supported(span, text):
    span_norm = normalize_text(span)
    text_norm = normalize_text(text)
    if not span_norm:
        return False
    if span_norm in text_norm:
        return True
    span_words = [w for w in re.findall(r"\b[\w'-]+\b", span_norm, flags=re.UNICODE) if len(w) > 2]
    if not span_words:
        return False
    return sum(1 for word in span_words if word in text_norm) / len(span_words) >= 0.8


def prediction_for_row(row):
    pred = row.get("prediction_normalized", row.get("prediction", {}))
    return pred if isinstance(pred, dict) else {}


def gold_for_row(row):
    gold = row.get("gold", {})
    return gold if isinstance(gold, dict) else {}


def pair_counter(examples):
    counter = Counter()
    for ex in examples:
        counter.update(gold_for_row(ex).items())
    return counter


def train_difficulty_scores(train_examples, train_predictions):
    score_by_id = defaultdict(float)
    pred_by_id = {str(row.get("id")): row for row in train_predictions}
    pair_counts = pair_counter(train_examples)

    for ex in train_examples:
        ex_id = str(ex.get("id"))
        gold = gold_for_row(ex)
        pred_row = pred_by_id.get(ex_id)
        pred = prediction_for_row(pred_row) if pred_row else {}

        missing = sum(1 for asp in gold if asp not in pred)
        wrong = sum(1 for asp, pol in gold.items() if asp in pred and pred[asp] != pol)
        extra = sum(1 for asp in pred if asp not in gold)
        rare_bonus = sum(max(0.0, 2.0 - math.log1p(pair_counts[(asp, pol)])) for asp, pol in gold.items())
        priority_bonus = sum(PRIORITY_ASPECTS.get(asp, 0.0) for asp in gold)
        score_by_id[ex_id] = 2.0 * missing + 1.5 * wrong + extra + rare_bonus + 0.3 * priority_bonus

    return score_by_id


def fallback_train_scores(train_examples):
    pair_counts = pair_counter(train_examples)
    score_by_id = defaultdict(float)
    for ex in train_examples:
        gold = gold_for_row(ex)
        rare_bonus = sum(max(0.0, 2.5 - math.log1p(pair_counts[(asp, pol)])) for asp, pol in gold.items())
        priority_bonus = sum(PRIORITY_ASPECTS.get(asp, 0.0) for asp in gold)
        neutral_conflict_bonus = sum(1.0 for pol in gold.values() if pol in {"neutral", "conflict"})
        score_by_id[str(ex.get("id"))] = rare_bonus + priority_bonus + neutral_conflict_bonus
    return score_by_id


def analyze_devel_errors(devel_predictions):
    missing = Counter()
    extra = Counter()
    wrong = Counter()
    wrong_triples = Counter()
    for row in devel_predictions:
        gold = gold_for_row(row)
        pred = prediction_for_row(row)
        for asp, pol in gold.items():
            if asp not in pred:
                missing[(asp, pol)] += 1
            elif pred[asp] != pol:
                wrong[(asp, pol)] += 1
                wrong_triples[(asp, pol, pred[asp])] += 1
        for asp, pol in pred.items():
            if asp not in gold:
                extra[(asp, pol)] += 1
    return {
        "missing": missing,
        "extra": extra,
        "wrong": wrong,
        "wrong_triples": wrong_triples,
    }


def prediction_error_profile(examples, predictions):
    pred_by_id = {str(row.get("id")): row for row in predictions}
    missing = Counter()
    extra = Counter()
    wrong = Counter()
    exact_matches = 0
    with_predictions = 0

    for ex in examples:
        pred_row = pred_by_id.get(str(ex.get("id")))
        if pred_row is None:
            continue
        with_predictions += 1
        gold = gold_for_row(ex)
        pred = prediction_for_row(pred_row)
        if gold == pred:
            exact_matches += 1
        for asp, pol in gold.items():
            if asp not in pred:
                missing[(asp, pol)] += 1
            elif pred[asp] != pol:
                wrong[(asp, pol, pred[asp])] += 1
        for asp, pol in pred.items():
            if asp not in gold:
                extra[(asp, pol)] += 1

    return {
        "examples": len(examples),
        "with_predictions": with_predictions,
        "exact_matches": exact_matches,
        "exact_match_accuracy": exact_matches / with_predictions if with_predictions else None,
        "missing": missing,
        "extra": extra,
        "wrong": wrong,
    }


def summarize_train_difficulty(train_examples, train_predictions, difficulty_scores, limit=30):
    pred_by_id = {str(row.get("id")): row for row in train_predictions}
    rows = []
    for ex in train_examples:
        ex_id = str(ex.get("id"))
        gold = gold_for_row(ex)
        pred_row = pred_by_id.get(ex_id)
        pred = prediction_for_row(pred_row) if pred_row else {}
        missing = {asp: pol for asp, pol in gold.items() if asp not in pred}
        wrong = {
            asp: {"gold": pol, "pred": pred[asp]}
            for asp, pol in gold.items()
            if asp in pred and pred[asp] != pol
        }
        extra = {asp: pol for asp, pol in pred.items() if asp not in gold}
        error_count = len(missing) + len(wrong) + len(extra)
        score = float(difficulty_scores[str(ex.get("id"))])
        if train_predictions and error_count == 0:
            continue
        rows.append(
            {
                "id": ex_id,
                "language": ex.get("language"),
                "difficulty_score": round(score, 4),
                "gold": gold,
                "prediction": pred if train_predictions else None,
                "errors": {
                    "missing": missing,
                    "wrong": wrong,
                    "extra": extra,
                    "total": error_count,
                },
                "text_preview": re.sub(r"\s+", " ", ex.get("text", "")).strip()[:260],
            }
        )

    rows.sort(
        key=lambda row: (
            row["errors"]["total"],
            row["difficulty_score"],
            len(row["gold"]),
        ),
        reverse=True,
    )
    return rows[:limit]


def template(target_gold, difficulty, avoid_aspects=None, min_words=15, max_words=140):
    return {
        "target_gold": target_gold,
        "difficulty": difficulty,
        "avoid_aspects": avoid_aspects or [],
        "min_words": min_words,
        "max_words": max_words,
    }


TEMPLATES = {
    "price_value_boundary": [
        template(
            {"food_quality": "positive", "restaurant_prices": "negative", "restaurant_general": "conflict"},
            "Good food, but the restaurant feels expensive overall for what it offers. Do not label food_prices unless a specific dish or menu price is evaluated.",
            ["food_prices"],
        ),
        template(
            {"food_quality": "positive", "food_prices": "negative", "restaurant_general": "conflict"},
            "The food tastes good, but a specific dish, menu, or portion is overpriced. This is food_prices, not restaurant_prices.",
            ["restaurant_prices"],
        ),
        template(
            {"food_quality": "positive", "restaurant_prices": "positive", "restaurant_general": "positive"},
            "Overall restaurant value is good even if the review does not discuss individual dish prices.",
            ["food_prices"],
        ),
        template(
            {"food_quality": "negative", "restaurant_prices": "negative", "restaurant_general": "negative"},
            "The restaurant is poor value overall because the experience is not worth the price.",
            ["food_prices"],
        ),
        template(
            {"food_quality": "positive", "food_prices": "positive", "food_style_options": "positive", "restaurant_general": "positive"},
            "A menu or dish price is praised together with portions or variety; keep price at food_prices.",
            ["restaurant_prices"],
        ),
        template(
            {"restaurant_prices": "neutral", "food_quality": "positive", "restaurant_general": "positive"},
            "The price is explicitly described as acceptable/fair/normal while the food is good.",
            ["food_prices"],
        ),
        template(
            {"restaurant_prices": "conflict", "food_quality": "positive", "service": "negative", "restaurant_general": "conflict"},
            "Mixed value judgment: good food but service makes the final value questionable.",
            ["food_prices"],
        ),
    ],
    "food_style_options_boundary": [
        template(
            {"food_quality": "positive", "food_style_options": "negative", "restaurant_general": "conflict"},
            "Food tastes good, but portions, menu variety, or presentation are criticized.",
        ),
        template(
            {"food_quality": "neutral", "food_style_options": "positive", "restaurant_general": "positive"},
            "Food is acceptable, while variety/presentation/originality is explicitly praised.",
        ),
        template(
            {"food_quality": "negative", "food_style_options": "positive", "restaurant_general": "conflict"},
            "Creative or varied menu but disappointing taste/cooking.",
        ),
        template(
            {"food_style_options": "negative", "restaurant_general": "negative"},
            "Limited menu/options or poor portions without a clear taste evaluation.",
            ["food_quality"],
        ),
        template(
            {"food_quality": "positive", "food_style_options": "positive", "food_prices": "negative", "restaurant_general": "conflict"},
            "Good food and generous/varied menu, but dish/menu price is too high.",
            ["restaurant_prices"],
        ),
    ],
    "ambience_location_service_boundary": [
        template(
            {"ambience": "positive", "service": "positive", "restaurant_general": "positive"},
            "Explicitly pleasant atmosphere/decor/comfort together with good staff.",
            ["location"],
        ),
        template(
            {"location": "positive", "restaurant_general": "positive"},
            "Central location or views are praised, but no interior atmosphere opinion should be inferred.",
            ["ambience"],
        ),
        template(
            {"service": "positive", "restaurant_general": "positive"},
            "Friendly or efficient staff are praised, but ambience is not evaluated.",
            ["ambience"],
        ),
        template(
            {"ambience": "negative", "food_quality": "positive", "restaurant_general": "conflict"},
            "Food is good but the place is noisy, cramped, uncomfortable, dirty, or badly decorated.",
        ),
        template(
            {"ambience": "neutral", "food_quality": "positive", "restaurant_general": "positive"},
            "Atmosphere is explicitly described as normal/acceptable/plain, while food is good.",
        ),
    ],
    "drinks_rare_aspects": [
        template(
            {"drinks_style_options": "positive", "food_quality": "positive", "restaurant_general": "positive"},
            "Wine list, beer selection, cocktails, or beverage variety are praised; this is drinks_style_options, not drinks_quality.",
            ["drinks_quality"],
        ),
        template(
            {"drinks_style_options": "negative", "restaurant_general": "negative"},
            "Drink selection is limited or disappointing without judging drink taste.",
            ["drinks_quality"],
        ),
        template(
            {"drinks_quality": "positive", "food_quality": "neutral", "restaurant_general": "positive"},
            "A specific drink tastes good or coffee/wine quality is praised; variety is not evaluated.",
            ["drinks_style_options"],
        ),
        template(
            {"drinks_quality": "negative", "food_quality": "positive", "restaurant_general": "conflict"},
            "Food is good but cocktails/wine/coffee quality is explicitly bad.",
            ["drinks_style_options"],
        ),
        template(
            {"drinks_prices": "negative", "drinks_quality": "positive", "restaurant_general": "conflict"},
            "A drink is good but explicitly too expensive.",
        ),
    ],
    "neutral_conflict_polarity": [
        template(
            {"food_quality": "neutral", "restaurant_general": "neutral"},
            "Food and overall experience are explicitly average, correct, acceptable, normal, or nothing special.",
        ),
        template(
            {"food_quality": "positive", "restaurant_prices": "negative", "restaurant_general": "conflict"},
            "Good food but poor overall value leads to mixed overall sentiment.",
            ["food_prices"],
        ),
        template(
            {"service": "neutral", "food_quality": "positive", "restaurant_general": "positive"},
            "Service is explicitly adequate/correct but not warm; food is positive.",
        ),
        template(
            {"food_quality": "conflict", "restaurant_general": "conflict"},
            "Some dishes are praised and others criticized; overall is mixed.",
        ),
        template(
            {"restaurant_general": "neutral", "food_quality": "positive", "ambience": "negative"},
            "Mixed signals but final overall statement is explicitly neutral rather than positive or negative.",
        ),
        template(
            {"restaurant_prices": "neutral", "food_quality": "neutral", "restaurant_general": "neutral"},
            "Explicitly fair price and average food, with no strong recommendation.",
        ),
    ],
    "hard_negatives_empty": [
        template(
            {},
            "Factual restaurant visit description with no explicit opinion about any aspect.",
            min_words=10,
            max_words=80,
        ),
        template(
            {},
            "Mentions dishes, location, or staff factually, but avoids evaluative language.",
            min_words=10,
            max_words=80,
        ),
        template(
            {"restaurant_general": "neutral"},
            "Very minimal explicit overall neutral opinion, with no other aspect sentiment.",
            min_words=10,
            max_words=80,
        ),
    ],
    "natural_regularizers": [
        template(
            {"food_quality": "positive", "service": "positive", "restaurant_general": "positive"},
            "Ordinary positive review with common aspects only.",
        ),
        template(
            {"food_quality": "negative", "service": "negative", "restaurant_general": "negative"},
            "Ordinary negative review with common aspects only.",
        ),
        template(
            {"food_quality": "positive", "service": "positive", "ambience": "positive", "restaurant_general": "positive"},
            "Natural positive review with 3-4 common labels.",
        ),
        template(
            {"food_quality": "positive", "service": "negative", "restaurant_general": "conflict"},
            "Natural mixed review: food positive, service negative, overall mixed.",
        ),
        template(
            {"food_quality": "positive", "ambience": "negative", "restaurant_general": "conflict"},
            "Natural mixed review: food positive, ambience negative, overall mixed.",
        ),
    ],
}


def weighted_choice_without_replacement(items, weights, k, rng):
    chosen = []
    pool = list(zip(items, weights))
    for _ in range(min(k, len(pool))):
        total = sum(max(0.001, weight) for _, weight in pool)
        pick = rng.random() * total
        upto = 0.0
        selected_idx = 0
        for idx, (_, weight) in enumerate(pool):
            upto += max(0.001, weight)
            if upto >= pick:
                selected_idx = idx
                break
        item, _ = pool.pop(selected_idx)
        chosen.append(item)
    return chosen


def seed_relevance(example, plan, difficulty_score, seed_usage):
    gold = gold_for_row(example)
    score = difficulty_score
    for asp, pol in plan.target_gold.items():
        if asp in gold:
            score += 3.0
        if gold.get(asp) == pol:
            score += 2.0
    if any(pol in {"neutral", "conflict"} for pol in gold.values()):
        score += 1.0
    score -= 0.4 * seed_usage[str(example.get("id"))]
    score -= 0.2 * abs(len(gold) - len(plan.target_gold))
    return max(0.01, score)


def choose_style_seeds(plan, train_examples, difficulty_scores, seed_usage, rng, k=3):
    same_language = [ex for ex in train_examples if ex.get("language") == plan.language]
    candidates = same_language if len(same_language) >= k else train_examples
    scored = [
        (ex, seed_relevance(ex, plan, difficulty_scores[str(ex.get("id"))], seed_usage))
        for ex in candidates
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    top = scored[:80]
    selected = weighted_choice_without_replacement(
        [ex for ex, _ in top],
        [score for _, score in top],
        k,
        rng,
    )
    for ex in selected:
        seed_usage[str(ex.get("id"))] += 1
    return selected


def make_language_sequence(count, rng):
    es_count = round(count * 0.675)
    languages = ["es"] * es_count + ["en"] * (count - es_count)
    rng.shuffle(languages)
    return languages


def build_plan(bucket, idx, language, train_examples, difficulty_scores, seed_usage, rng):
    tmpl = rng.choice(TEMPLATES[bucket])
    temp_plan = GenerationPlan(
        plan_id=f"{BUCKET_SHORT[bucket]}_{idx:04d}",
        bucket=bucket,
        language=language,
        target_gold=dict(tmpl["target_gold"]),
        difficulty=tmpl["difficulty"],
        avoid_aspects=list(tmpl["avoid_aspects"]),
        style_seed_ids=[],
        min_words=tmpl["min_words"],
        max_words=tmpl["max_words"],
    )
    seeds = choose_style_seeds(temp_plan, train_examples, difficulty_scores, seed_usage, rng)
    temp_plan.style_seed_ids = [str(ex.get("id")) for ex in seeds]
    return temp_plan, seeds


def plan_to_json(plan):
    return {
        "plan_id": plan.plan_id,
        "bucket": plan.bucket,
        "language": plan.language,
        "target_gold": plan.target_gold,
        "difficulty": plan.difficulty,
        "avoid_aspects": plan.avoid_aspects,
        "style_seed_ids": plan.style_seed_ids,
        "min_words": plan.min_words,
        "max_words": plan.max_words,
    }


def build_schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string"},
            "language": {"type": "string", "enum": ["en", "es"]},
            "gold_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "aspect": {"type": "string", "enum": ASPECTS},
                        "polarity": {"type": "string", "enum": POLARITIES},
                        "support_span": {"type": "string"},
                    },
                    "required": ["aspect", "polarity", "support_span"],
                },
            },
            "notes": {"type": "string"},
        },
        "required": ["text", "language", "gold_items", "notes"],
    }


def aspect_reference_text():
    return "\n".join(f"- {aspect}: {definition}" for aspect, definition in ASPECT_DEFINITIONS.items())


def build_messages(plan, seed_examples):
    target_json = json.dumps(plan.target_gold, ensure_ascii=False, sort_keys=True)
    avoid = ", ".join(plan.avoid_aspects) if plan.avoid_aspects else "none"
    style_blocks = []
    for idx, ex in enumerate(seed_examples, start=1):
        style_blocks.append(
            "\n".join(
                [
                    f"Example {idx} id={ex.get('id')} language={ex.get('language')}",
                    f"Text: {ex.get('text')}",
                    "Gold: " + json.dumps(gold_for_row(ex), ensure_ascii=False, sort_keys=True),
                ]
            )
        )

    system = (
        "You generate realistic restaurant reviews for ABSA synthetic data augmentation. "
        "Return exactly one JSON object following the provided schema. "
        "Do not copy or paraphrase the style examples. "
        "The generated review must be natural, human-like, and plausible for the requested language."
    )
    user = f"""
Generate one synthetic restaurant review for an ABSA dataset.

Language: {plan.language}
Bucket: {plan.bucket}
Target gold labels, exact and complete:
{target_json}

Difficulty to express:
{plan.difficulty}

Aspects to avoid unless explicitly required by the target gold: {avoid}

Official aspect definitions:
{aspect_reference_text()}

Polarity rules:
- positive: explicit favorable opinion.
- negative: explicit unfavorable opinion.
- neutral: explicitly evaluated as average, acceptable, normal, correct, or neither good nor bad.
- conflict: explicit mixed opinion about the same aspect or clear positive and negative evidence.
- Absent means not included. Never use neutral for an absent aspect.

Output requirements:
- Return JSON only.
- `gold_items` must contain exactly the target aspects and polarities, no more and no fewer.
- Each `support_span` must be a short exact phrase copied from the generated text that justifies that aspect.
- The review should be {plan.min_words}-{plan.max_words} words.
- Do not mention labels, aspects, sentiment, JSON, annotation, or this task in the review text.
- Do not use bullet points or markdown.
- Do not copy or lightly paraphrase the examples below.

Style examples from train.json:
{chr(10).join(style_blocks)}
""".strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def post_openai_chat_completion(args, api_key, messages):
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_completion_tokens": args.max_completion_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "absa_synthetic_example",
                "strict": True,
                "schema": build_schema(),
            },
        },
    }
    return post_json_with_fallback(args.endpoint, api_key, payload, args.max_retries, args.retry_base_seconds)


def post_json_with_fallback(endpoint, api_key, payload, max_retries, retry_base_seconds):
    last_error = None
    for attempt in range(max_retries):
        try:
            return post_json(endpoint, api_key, payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {exc.code}: {body}"
            if exc.code == 400 and "max_completion_tokens" in payload:
                fallback = dict(payload)
                fallback["max_tokens"] = fallback.pop("max_completion_tokens")
                try:
                    return post_json(endpoint, api_key, fallback)
                except urllib.error.HTTPError as inner_exc:
                    body = inner_exc.read().decode("utf-8", errors="replace")
                    last_error = f"HTTP {inner_exc.code}: {body}"
            if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError(last_error) from exc
        except urllib.error.URLError as exc:
            last_error = str(exc)

        time.sleep(retry_base_seconds * (2**attempt) + random.random())

    raise RuntimeError(last_error or "OpenAI request failed")


def post_json(endpoint, api_key, payload):
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_api_response(response):
    try:
        message = response["choices"][0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected API response shape: {response}") from exc

    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Empty API content: {response}")
    return json.loads(content)


def gold_from_items(items):
    if not isinstance(items, list):
        raise ValueError("gold_items must be a list")
    gold = {}
    supports = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("gold item must be an object")
        aspect = item.get("aspect")
        polarity = item.get("polarity")
        support_span = item.get("support_span", "")
        if aspect not in ASPECTS:
            raise ValueError(f"invalid aspect: {aspect}")
        if polarity not in POLARITIES:
            raise ValueError(f"invalid polarity: {polarity}")
        if aspect in gold:
            raise ValueError(f"duplicate aspect: {aspect}")
        gold[aspect] = polarity
        supports[aspect] = str(support_span)
    return gold, supports


def max_reference_similarity(text, reference_ngrams):
    grams = ngrams(text)
    if not grams:
        return 0.0
    return max((jaccard(grams, ref) for ref in reference_ngrams if ref), default=0.0)


def validate_candidate(payload, plan, reference_texts, reference_ngrams, accepted_texts, accepted_ngrams):
    if not isinstance(payload, dict):
        return None, "payload is not an object"

    text = re.sub(r"\s+", " ", str(payload.get("text", "")).strip())
    language = payload.get("language")
    if language != plan.language:
        return None, f"language mismatch: expected {plan.language}, got {language}"
    if not text:
        return None, "empty text"

    lowered = text.lower()
    if any(marker in lowered for marker in FORBIDDEN_TEXT_MARKERS):
        return None, "annotation/task language leaked into review text"
    if any(token in text for token in ["```", "\n-", "\n*", "{", "}"]):
        return None, "markdown or JSON-like text leaked into review"

    wc = word_count(text)
    if wc < plan.min_words or wc > plan.max_words:
        return None, f"word count {wc} outside [{plan.min_words}, {plan.max_words}]"

    gold, supports = gold_from_items(payload.get("gold_items", []))
    if gold != plan.target_gold:
        return None, f"gold mismatch: expected {plan.target_gold}, got {gold}"

    for aspect in gold:
        if not span_supported(supports.get(aspect, ""), text):
            return None, f"support span missing for {aspect}: {supports.get(aspect)}"

    norm = normalize_text(text)
    if norm in reference_texts or norm in accepted_texts:
        return None, "exact duplicate text"

    max_ref_sim = max_reference_similarity(text, reference_ngrams)
    if max_ref_sim > 0.55:
        return None, f"too similar to train/devel text: {max_ref_sim:.3f}"

    max_accept_sim = max_reference_similarity(text, accepted_ngrams)
    if max_accept_sim > 0.62:
        return None, f"too similar to accepted synthetic text: {max_accept_sim:.3f}"

    candidate = {
        "language": language,
        "text": text,
        "gold": gold,
        "_meta": {
            "bucket": plan.bucket,
            "plan_id": plan.plan_id,
            "difficulty": plan.difficulty,
            "avoid_aspects": plan.avoid_aspects,
            "style_seed_ids": plan.style_seed_ids,
            "support_spans": supports,
            "word_count": wc,
            "max_reference_5gram_jaccard": max_ref_sim,
            "max_synthetic_5gram_jaccard": max_accept_sim,
        },
    }
    return candidate, None


def call_one_plan(args, api_key, plan, seed_examples):
    messages = build_messages(plan, seed_examples)
    response = post_openai_chat_completion(args, api_key, messages)
    parsed = parse_api_response(response)
    usage = response.get("usage", {})
    return {
        "plan": plan_to_json(plan),
        "payload": parsed,
        "usage": usage,
        "model": response.get("model", args.model),
    }


def safe_output_paths(args):
    paths = [
        Path(args.synthetic_output),
        Path(args.combined_output),
        Path(args.metadata_output),
        Path(args.rejected_output),
        Path(args.plans_output),
        Path(args.report_output),
    ]
    existing = [path for path in paths if path.exists()]
    if existing and not args.overwrite and not args.resume and not args.dry_run:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist. Use --overwrite or --resume:\n  {joined}")
    if args.overwrite:
        for path in existing:
            path.unlink()


def save_progress(args, accepted, rejected, plans, report_extra=None):
    synthetic = [
        {
            "id": f"synth_v1_{idx:04d}",
            "language": row["language"],
            "text": row["text"],
            "gold": row["gold"],
        }
        for idx, row in enumerate(accepted, start=1)
    ]
    metadata = []
    for idx, row in enumerate(accepted, start=1):
        meta_row = dict(synthetic[idx - 1])
        meta_row["_meta"] = row["_meta"]
        metadata.append(meta_row)

    train = load_json(args.train_file)
    combined = train + synthetic

    write_json(args.synthetic_output, synthetic)
    write_json(args.combined_output, combined)
    write_json(args.metadata_output, metadata)
    write_json(args.plans_output, plans)

    report = build_report(synthetic, accepted, rejected, plans, report_extra or {})
    write_json(args.report_output, report)


def build_report(synthetic, accepted, rejected, plans, extra):
    aspect_counts = Counter()
    polarity_counts = Counter()
    bucket_counts = Counter()
    language_counts = Counter()
    label_count_dist = Counter()
    for row in synthetic:
        gold = gold_for_row(row)
        aspect_counts.update(gold.keys())
        polarity_counts.update(gold.values())
        label_count_dist[len(gold)] += 1
        language_counts[row.get("language", "unknown")] += 1
    for row in accepted:
        bucket_counts[row["_meta"]["bucket"]] += 1

    reject_reasons = Counter(row.get("reason", "unknown") for row in rejected)
    return {
        "accepted": len(synthetic),
        "rejected": len(rejected),
        "planned": len(plans),
        "aspect_counts": dict(aspect_counts),
        "polarity_counts": dict(polarity_counts),
        "bucket_counts": dict(bucket_counts),
        "language_counts": dict(language_counts),
        "label_count_distribution": dict(sorted(label_count_dist.items())),
        "top_reject_reasons": reject_reasons.most_common(30),
        **extra,
    }


def load_resume(args):
    metadata_path = Path(args.metadata_output)
    if not args.resume or not metadata_path.exists():
        return []
    rows = load_json(metadata_path)
    accepted = []
    for row in rows:
        accepted.append(
            {
                "language": row["language"],
                "text": row["text"],
                "gold": row["gold"],
                "_meta": row.get("_meta", {}),
            }
        )
    return accepted


def maybe_generate_train_predictions(args):
    train_predictions = Path(args.train_predictions)
    if train_predictions.exists() or not args.run_train_inference or args.dry_run:
        return

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Cannot run train inference because weights were not found: {weights}"
        )

    inference_script = ABSA_DIR / "bin" / "finetune" / "finetune-inference.py"
    train_predictions.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(inference_script),
        "--weights",
        str(weights),
        "--prompt-file",
        str(Path(args.prompt_file)),
        "--data",
        "train",
        "--output",
        str(train_predictions),
        "--temperature",
        "0",
        "--top-p",
        "1.0",
        "--top-k",
        "20",
        "--presence-penalty",
        "0.0",
        "--max-new-tokens",
        "512",
    ]
    print("Running train inference for difficulty scoring:")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(inference_script.parent), check=True)


def prepare_reference_sets(train, devel):
    texts = {normalize_text(ex.get("text", "")) for ex in train + devel}
    grams = [ngrams(ex.get("text", "")) for ex in train + devel]
    return texts, grams


def build_needed_plans(args, bucket_counts, train, difficulty_scores, seed_usage, rng, plan_start):
    plans_with_seeds = []
    plan_idx = plan_start
    for bucket, quota in BUCKET_QUOTAS.items():
        missing = quota - bucket_counts[bucket]
        if missing <= 0:
            continue
        attempts = max(missing, math.ceil(missing * args.candidate_multiplier))
        languages = make_language_sequence(attempts, rng)
        for language in languages:
            plan_idx += 1
            plan, seeds = build_plan(bucket, plan_idx, language, train, difficulty_scores, seed_usage, rng)
            plans_with_seeds.append((plan, seeds))
    return plans_with_seeds, plan_idx


def generation_complete(bucket_counts):
    return all(bucket_counts[bucket] >= quota for bucket, quota in BUCKET_QUOTAS.items())


def trim_to_quotas(accepted):
    selected = []
    counts = Counter()
    for row in accepted:
        bucket = row["_meta"]["bucket"]
        if counts[bucket] < BUCKET_QUOTAS[bucket]:
            selected.append(row)
            counts[bucket] += 1
    return selected


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    safe_output_paths(args)
    maybe_generate_train_predictions(args)

    train = load_json(args.train_file)
    devel = load_json(args.devel_file)
    devel_predictions = load_json(args.devel_predictions)

    train_predictions_path = Path(args.train_predictions)
    if train_predictions_path.exists():
        train_predictions = load_json(train_predictions_path)
        difficulty_scores = train_difficulty_scores(train, train_predictions)
        train_score_source = str(train_predictions_path)
    else:
        train_predictions = []
        difficulty_scores = fallback_train_scores(train)
        train_score_source = "fallback rare-pair and priority-aspect train scoring"

    devel_errors = analyze_devel_errors(devel_predictions)
    train_profile = prediction_error_profile(train, train_predictions) if train_predictions else None
    top_train_difficult = summarize_train_difficulty(
        train,
        train_predictions,
        difficulty_scores,
    )
    seed_usage = Counter()
    reference_texts, reference_ngrams = prepare_reference_sets(train, devel)
    accepted = load_resume(args)
    rejected = []
    plans_json = []
    plan_counter = 0
    total_candidates = 0

    if args.dry_run:
        bucket_counts = Counter()
        plans_with_seeds, _ = build_needed_plans(
            args,
            bucket_counts,
            train,
            difficulty_scores,
            seed_usage,
            rng,
            plan_counter,
        )
        plans_json = [plan_to_json(plan) for plan, _ in plans_with_seeds]
        planned_bucket_counts = Counter(plan["bucket"] for plan in plans_json)
        planned_language_counts = Counter(plan["language"] for plan in plans_json)
        write_json(args.plans_output, plans_json)
        write_json(
            args.report_output,
            {
                "dry_run": True,
                "planned": len(plans_json),
                "planned_bucket_counts": dict(planned_bucket_counts),
                "planned_language_counts": dict(planned_language_counts),
                "bucket_quotas": BUCKET_QUOTAS,
                "train_score_source": train_score_source,
                "train_error_profile": (
                    {
                        "examples": train_profile["examples"],
                        "with_predictions": train_profile["with_predictions"],
                        "exact_matches": train_profile["exact_matches"],
                        "exact_match_accuracy": train_profile["exact_match_accuracy"],
                        "missing": counter_to_jsonable(train_profile["missing"]),
                        "extra": counter_to_jsonable(train_profile["extra"]),
                        "wrong": counter_to_jsonable(train_profile["wrong"]),
                    }
                    if train_profile
                    else None
                ),
                "top_train_difficult_examples": top_train_difficult,
                "devel_missing": counter_to_jsonable(devel_errors["missing"]),
                "devel_extra": counter_to_jsonable(devel_errors["extra"]),
                "devel_wrong": counter_to_jsonable(devel_errors["wrong"]),
                "devel_wrong_triples": counter_to_jsonable(devel_errors["wrong_triples"]),
            },
        )
        print(f"Dry run complete. Plans written to {args.plans_output}")
        return

    api_key = read_api_key(args)
    accepted_texts = {normalize_text(row["text"]) for row in accepted}
    accepted_ngrams = [ngrams(row["text"]) for row in accepted]
    bucket_counts = Counter(row["_meta"]["bucket"] for row in accepted)
    all_rejected = []

    while not generation_complete(bucket_counts) and total_candidates < args.max_candidates:
        remaining_slots = args.max_candidates - total_candidates
        plans_with_seeds, plan_counter = build_needed_plans(
            args,
            bucket_counts,
            train,
            difficulty_scores,
            seed_usage,
            rng,
            plan_counter,
        )
        if len(plans_with_seeds) > remaining_slots:
            plans_with_seeds = plans_with_seeds[:remaining_slots]
        if not plans_with_seeds:
            break

        print(
            f"Starting wave with {len(plans_with_seeds)} candidates. "
            f"Accepted so far: {len(accepted)} / {sum(BUCKET_QUOTAS.values())}",
            flush=True,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_plan = {
                executor.submit(call_one_plan, args, api_key, plan, seeds): plan
                for plan, seeds in plans_with_seeds
            }
            for future in concurrent.futures.as_completed(future_to_plan):
                plan = future_to_plan[future]
                plans_json.append(plan_to_json(plan))
                total_candidates += 1
                try:
                    result = future.result()
                    payload = result["payload"]
                    candidate, reason = validate_candidate(
                        payload,
                        plan,
                        reference_texts,
                        reference_ngrams,
                        accepted_texts,
                        accepted_ngrams,
                    )
                    if candidate is None:
                        rejected.append(
                            {
                                "plan": plan_to_json(plan),
                                "reason": reason,
                                "payload": payload,
                            }
                        )
                        continue

                    bucket = candidate["_meta"]["bucket"]
                    if bucket_counts[bucket] >= BUCKET_QUOTAS[bucket]:
                        rejected.append(
                            {
                                "plan": plan_to_json(plan),
                                "reason": f"bucket quota already filled: {bucket}",
                                "payload": payload,
                            }
                        )
                        continue

                    candidate["_meta"]["api_model"] = result["model"]
                    candidate["_meta"]["usage"] = result.get("usage", {})
                    accepted.append(candidate)
                    bucket_counts[bucket] += 1
                    accepted_texts.add(normalize_text(candidate["text"]))
                    accepted_ngrams.append(ngrams(candidate["text"]))

                except Exception as exc:
                    rejected.append(
                        {
                            "plan": plan_to_json(plan),
                            "reason": f"api_or_parse_error: {exc}",
                        }
                    )

        accepted = trim_to_quotas(accepted)
        bucket_counts = Counter(row["_meta"]["bucket"] for row in accepted)
        append_jsonl(args.rejected_output, rejected)
        all_rejected.extend(rejected)
        rejected = []
        save_progress(
            args,
            accepted,
            all_rejected,
            plans_json,
            {
                "complete": generation_complete(bucket_counts),
                "total_candidates": total_candidates,
                "bucket_quotas": BUCKET_QUOTAS,
                "train_score_source": train_score_source,
                "top_train_difficult_examples": top_train_difficult,
            },
        )
        print(f"Progress saved. Bucket counts: {dict(bucket_counts)}", flush=True)

    if not generation_complete(bucket_counts):
        print("WARNING: generation ended before all quotas were filled.", file=sys.stderr)

    save_progress(
        args,
        accepted,
        all_rejected,
        plans_json,
        {
            "complete": generation_complete(bucket_counts),
            "total_candidates": total_candidates,
            "bucket_quotas": BUCKET_QUOTAS,
            "train_score_source": train_score_source,
            "train_error_profile": (
                {
                    "examples": train_profile["examples"],
                    "with_predictions": train_profile["with_predictions"],
                    "exact_matches": train_profile["exact_matches"],
                    "exact_match_accuracy": train_profile["exact_match_accuracy"],
                    "missing": counter_to_jsonable(train_profile["missing"]),
                    "extra": counter_to_jsonable(train_profile["extra"]),
                    "wrong": counter_to_jsonable(train_profile["wrong"]),
                }
                if train_profile
                else None
            ),
            "top_train_difficult_examples": top_train_difficult,
            "devel_missing": counter_to_jsonable(devel_errors["missing"]),
            "devel_extra": counter_to_jsonable(devel_errors["extra"]),
            "devel_wrong": counter_to_jsonable(devel_errors["wrong"]),
            "devel_wrong_triples": counter_to_jsonable(devel_errors["wrong_triples"]),
        },
    )
    print(f"Synthetic dataset written to {args.synthetic_output}")
    print(f"Combined train dataset written to {args.combined_output}")
    print(f"Accepted {len(accepted)} synthetic examples from {total_candidates} candidates.")


if __name__ == "__main__":
    main()
