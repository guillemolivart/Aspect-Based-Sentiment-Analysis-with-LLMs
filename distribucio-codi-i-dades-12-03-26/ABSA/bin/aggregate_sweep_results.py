#!/usr/bin/env python3
"""
Aggregate all .summary.csv files from outputs/hyperparam_sweep into a single ranked table.
"""

import csv
import json
from pathlib import Path
from tabulate import tabulate

SWEEP_DIR = Path(__file__).parent.parent / "outputs" / "hyperparam_sweep"

def parse_summary_csv(csv_path):
    """Read a .summary.csv file and extract the row."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if rows:
            return rows[0]  # Single row per config
    return None

def parse_chosen_json(json_path):
    """Read a .chosen.json file to extract generation config and metrics."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        chosen = data.get("chosen", {})
        config = chosen.get("generation_config", {})
        return {
            "config": config,
            "metrics": {
                "f1_micro": chosen.get("f1_micro"),
                "f1_macro": chosen.get("f1_macro"),
                "precision_macro": chosen.get("precision_macro"),
                "recall_macro": chosen.get("recall_macro"),
                "precision_micro": chosen.get("precision_micro"),
                "recall_micro": chosen.get("recall_micro"),
            }
        }

def main():
    # Find all .chosen.json files
    json_files = sorted(SWEEP_DIR.glob("*.chosen.json"))
    
    if not json_files:
        print(f"No .chosen.json files found in {SWEEP_DIR}")
        return
    
    results = []
    
    for json_path in json_files:
        config_name = json_path.stem.replace(".chosen", "")
        
        try:
            data = parse_chosen_json(json_path)
            config = data["config"]
            metrics = data["metrics"]
            
            results.append({
                "Config": config_name,
                "Temp": config.get("temperature"),
                "Top-P": config.get("top_p"),
                "Top-K": config.get("top_k"),
                "PP": config.get("presence_penalty"),
                "F1-Micro": round(metrics.get("f1_micro", 0), 2),
                "F1-Macro": round(metrics.get("f1_macro", 0), 2),
                "Prec-Macro": round(metrics.get("precision_macro", 0), 2),
                "Rec-Macro": round(metrics.get("recall_macro", 0), 2),
                "Prec-Micro": round(metrics.get("precision_micro", 0), 2),
                "Rec-Micro": round(metrics.get("recall_micro", 0), 2),
            })
        except Exception as e:
            print(f"Error processing {json_path}: {e}")
            continue
    
    if not results:
        print("No results to display")
        return
    
    # Sort by F1-Micro descending
    results.sort(key=lambda x: x["F1-Micro"], reverse=True)
    
    # Print table
    print("\n" + "="*150)
    print("HYPERPARAMETER SWEEP RESULTS - Ranked by F1-Micro")
    print("="*150)
    print(tabulate(results, headers="keys", tablefmt="grid", floatfmt=".2f"))
    print("="*150)
    
    # Save to CSV
    output_csv = SWEEP_DIR / "AGGREGATED_RESULTS.csv"
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {output_csv}")
    print(f"\nTop 3 configurations:")
    for i, r in enumerate(results[:3], 1):
        print(f"  {i}. {r['Config']:40s} | F1-Micro: {r['F1-Micro']:6.2f} | F1-Macro: {r['F1-Macro']:6.2f}")

if __name__ == "__main__":
    main()
