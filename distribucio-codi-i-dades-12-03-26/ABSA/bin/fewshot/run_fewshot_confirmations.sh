#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ABSA_DIR="$ROOT_DIR/distribucio-codi-i-dades-12-03-26/ABSA"
PYTHON="$ROOT_DIR/.venv/bin/python"
LOG_DIR="$ROOT_DIR/logs/fewshot"
RUN_LOG="$LOG_DIR/fewshot_confirmations.$(date +%Y%m%d_%H%M%S).log"

MODEL_DIR="$ABSA_DIR/model"
EMBEDDING_DIR="$ABSA_DIR/embedding_model/qwen3_embedding_0_6b"
PROMPT_FILE="prompts/absa_v6.json"
SEEDS=(101 202 303)

CONFIGS=(
  "absa_mmr 8"
  "dense_topk 12"
  "absa_mmr 12"
  "dense_topk 10"
)

mkdir -p "$LOG_DIR" "$ABSA_DIR/outputs/fewshot"

log() {
  echo "[$(date -Is)] $*" | tee -a "$RUN_LOG"
}

repair_venv_python() {
  if [[ ! -x "$PYTHON" && -x /usr/bin/python3 ]]; then
    log "Repairing .venv/bin/python3 symlink"
    ln -sf /usr/bin/python3 "$ROOT_DIR/.venv/bin/python3"
  fi
}

check_environment() {
  repair_venv_python

  if [[ ! -x "$PYTHON" ]]; then
    log "ERROR: $PYTHON is not executable"
    exit 1
  fi
  if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    log "ERROR: generator model not found at $MODEL_DIR"
    exit 1
  fi
  if [[ ! -f "$EMBEDDING_DIR/config.json" ]]; then
    log "ERROR: embedding model not found at $EMBEDDING_DIR"
    exit 1
  fi
  if [[ ! -f "$ABSA_DIR/$PROMPT_FILE" ]]; then
    log "ERROR: prompt not found at $ABSA_DIR/$PROMPT_FILE"
    exit 1
  fi

  "$PYTHON" - <<'PY' | tee -a "$RUN_LOG"
import importlib
for package in ["torch", "transformers", "accelerate", "numpy"]:
    module = importlib.import_module(package)
    print(f"{package} {getattr(module, '__version__', 'ok')}")
PY
}

run_case() {
  local method="$1"
  local k="$2"
  local seed="$3"
  local name="fs_confirm_${method}_k${k}_seed${seed}"
  local log_file="$LOG_DIR/${name}.log"

  log "START $name"
  "$PYTHON" "$ABSA_DIR/bin/fewshot.py" \
    --model-path "$MODEL_DIR" \
    --embedding-model-path "$EMBEDDING_DIR" \
    --prompt-file "$PROMPT_FILE" \
    --method "$method" \
    --k "$k" \
    --seed "$seed" \
    --keep-raw \
    --resume \
    --checkpoint-every 1 \
    --output-prefix "$name" \
    2>&1 | tee "$log_file"

  local status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    log "FAILED $name status=$status"
    return "$status"
  fi
  log "DONE $name"
  return 0
}

print_confirmation_ranking() {
  "$PYTHON" - <<'PY'
import json
from collections import defaultdict
from pathlib import Path

base = Path("distribucio-codi-i-dades-12-03-26/ABSA/outputs/fewshot")
rows = []
for path in base.glob("fs_confirm_*_seed*.summary.json"):
    summary = json.loads(path.read_text())
    config = summary["config"]
    rows.append(
        {
            "method": config["method"],
            "k": config["k"],
            "seed": config["seed"],
            "f1_micro": summary["f1_micro"],
            "f1_macro": summary["f1_macro"],
            "precision_micro": summary["precision_micro"],
            "recall_micro": summary["recall_micro"],
            "predicted_total": summary["predicted_total"],
            "ok_total": summary["ok_total"],
            "file": path.name,
        }
    )

print("Per-run confirmation ranking")
print("F1_micro,F1_macro,P_micro,R_micro,method,k,seed,pred,ok,file")
for row in sorted(rows, key=lambda x: x["f1_micro"], reverse=True):
    print(
        f"{row['f1_micro']:.2f},{row['f1_macro']:.2f},"
        f"{row['precision_micro']:.2f},{row['recall_micro']:.2f},"
        f"{row['method']},{row['k']},{row['seed']},"
        f"{row['predicted_total']},{row['ok_total']},{row['file']}"
    )

grouped = defaultdict(list)
for row in rows:
    grouped[(row["method"], row["k"])].append(row)

print()
print("Grouped confirmation means")
print("method,k,n,F1_micro_mean,F1_micro_min,F1_micro_max,F1_macro_mean,P_micro_mean,R_micro_mean")
for (method, k), items in sorted(grouped.items()):
    def mean(key):
        return sum(item[key] for item in items) / len(items)

    f1s = [item["f1_micro"] for item in items]
    print(
        f"{method},{k},{len(items)},"
        f"{mean('f1_micro'):.2f},{min(f1s):.2f},{max(f1s):.2f},"
        f"{mean('f1_macro'):.2f},{mean('precision_micro'):.2f},"
        f"{mean('recall_micro'):.2f}"
    )
PY
}

main() {
  cd "$ROOT_DIR"
  log "Few-shot confirmations started"
  log "ROOT_DIR=$ROOT_DIR"
  log "PYTHON=$PYTHON"
  check_environment

  local failures=0
  for config in "${CONFIGS[@]}"; do
    read -r method k <<< "$config"
    for seed in "${SEEDS[@]}"; do
      run_case "$method" "$k" "$seed" || failures=$((failures + 1))
    done
  done

  log "Confirmations completed with failures=$failures"
  print_confirmation_ranking | tee "$LOG_DIR/fewshot_confirmation_ranking.csv"

  if [[ "$failures" -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
