#!/usr/bin/env python3
"""
Quick stats analyzer for FT.devel.json outputs.
Executes inference if needed, then computes accuracy and error analysis.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = PARENT_DIR.parent / "outputs"

if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from common import ASPECTS, POLARITIES


def analyze_json(json_path):
    """Load and analyze inference results."""
    if not json_path.exists():
        print(f"❌ File not found: {json_path}")
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n📊 ANALYSIS: {json_path.name}")
    print("=" * 70)
    
    # Basic stats
    total = len(data)
    print(f"Total examples: {total}")
    
    if total == 0:
        return data
    
    # Check which examples have predictions
    has_pred = sum(1 for ex in data if 'prediction_normalized' in ex)
    print(f"Examples with predictions: {has_pred}/{total}")
    
    # Accuracy: exact match on prediction_normalized vs gold
    exact_matches = 0
    aspect_precision = defaultdict(lambda: {"correct": 0, "total": 0})
    aspect_recall = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for ex in data:
        gold = ex.get('gold', {})
        pred = ex.get('prediction_normalized', {})
        
        if pred == gold:
            exact_matches += 1
        
        # Per-aspect stats (if gold and pred are dicts)
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
    
    print(f"Exact matches (prediction == gold): {exact_matches}/{total}")
    if has_pred > 0:
        print(f"Exact match accuracy: {100*exact_matches/has_pred:.2f}%")
    
    # Per-aspect metrics
    print(f"\n{'Aspect':<25} {'Precision':<12} {'Recall':<12}")
    print("-" * 50)
    for aspect in sorted(ASPECTS):
        prec = aspect_precision.get(aspect, {})
        rec = aspect_recall.get(aspect, {})
        
        prec_pct = (100 * prec["correct"] / prec["total"]) if prec["total"] > 0 else 0
        rec_pct = (100 * rec["correct"] / rec["total"]) if rec["total"] > 0 else 0
        
        print(f"{aspect:<25} {prec_pct:>6.2f}% ({prec['correct']}/{prec['total']:<3}) {rec_pct:>6.2f}% ({rec['correct']}/{rec['total']:<3})")
    
    # Error distribution
    print(f"\n--- ERROR ANALYSIS ---")
    missing_aspects = 0  # pred is empty but gold is not
    hallucinated = 0     # pred has aspects gold doesn't have
    wrong_polarity = 0   # same aspects but wrong polarity
    
    for ex in data:
        gold = ex.get('gold', {})
        pred = ex.get('prediction_normalized', {})
        
        if isinstance(gold, dict) and isinstance(pred, dict):
            if not pred and gold:
                missing_aspects += 1
            elif pred and not gold:
                hallucinated += 1
            else:
                gold_aspects = set(gold.keys())
                pred_aspects = set(pred.keys())
                
                # Check if same aspects but wrong polarity
                common = gold_aspects & pred_aspects
                if common:
                    for asp in common:
                        if gold[asp] != pred[asp]:
                            wrong_polarity += 1
                            break
    
    print(f"Empty predictions (gold != empty): {missing_aspects}")
    print(f"Hallucinated predictions (pred != empty, gold empty): {hallucinated}")
    print(f"Wrong polarity (right aspect, wrong sentiment): {wrong_polarity}")
    
    # Show sample errors
    print(f"\n--- SAMPLE ERRORS ---")
    error_count = 0
    for ex in data[:10]:  # First 10 examples
        gold = ex.get('gold', {})
        pred = ex.get('prediction_normalized', {})
        if pred != gold and error_count < 3:
            text = ex.get('text', '')[:80]
            print(f"\n✗ Text: {text}...")
            print(f"  Gold: {gold}")
            print(f"  Pred: {pred}")
            error_count += 1
    
    return data


if __name__ == "__main__":
    # Try to find FT.devel.json
    json_path = OUTPUT_DIR / "FT.devel.json"
    
    if not json_path.exists():
        # Try alternative name
        json_path = OUTPUT_DIR / "FT.train.devel.json"
    
    if not json_path.exists():
        # Try ABSA-specific path
        json_path = Path(__file__).parent.parent.parent / "outputs" / "FT.devel.json"
    
    print(f"Looking for inference output at: {json_path}")
    data = analyze_json(json_path)
