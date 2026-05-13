#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ABSA_DIR="$ROOT_DIR/distribucio-codi-i-dades-12-03-26/ABSA"
PYTHON="$ROOT_DIR/.venv/bin/python"
LOG_DIR="$ROOT_DIR/logs/fewshot"
RUN_LOG="$LOG_DIR/fewshot_exploration.$(date +%Y%m%d_%H%M%S).log"

MODEL_DIR="$ABSA_DIR/model"
EMBEDDING_DIR="$ABSA_DIR/embedding_model/qwen3_embedding_0_6b"
PROMPT_FILE="prompts/absa_v6.json"
SEED=42

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
  local name="fs_${method}_k${k}_seed${seed}"
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

run_smoke_test() {
  local name="smoke_fs_dense_topk_k5"
  log "START $name"
  "$PYTHON" "$ABSA_DIR/bin/fewshot.py" \
    --model-path "$MODEL_DIR" \
    --embedding-model-path "$EMBEDDING_DIR" \
    --prompt-file "$PROMPT_FILE" \
    --method dense_topk \
    --k 5 \
    --seed "$SEED" \
    --limit 3 \
    --keep-raw \
    --resume \
    --checkpoint-every 1 \
    --output-prefix "$name" \
    2>&1 | tee "$LOG_DIR/${name}.log"

  local status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    log "Smoke test failed status=$status"
    exit "$status"
  fi
  log "DONE $name"
}

print_ranking() {
  "$PYTHON" - <<'PY'
import json
from pathlib import Path

base = Path("distribucio-codi-i-dades-12-03-26/ABSA/outputs/fewshot")
rows = []
for path in base.glob("fs_*_seed42.summary.json"):
    summary = json.loads(path.read_text())
    config = summary["config"]
    rows.append(
        (
            summary["f1_micro"],
            summary["f1_macro"],
            summary["precision_micro"],
            summary["recall_micro"],
            config["method"],
            config["k"],
            path.name,
        )
    )

print("F1_micro,F1_macro,P_micro,R_micro,method,k,file")
for row in sorted(rows, reverse=True):
    print(
        f"{row[0]:.2f},{row[1]:.2f},{row[2]:.2f},{row[3]:.2f},"
        f"{row[4]},{row[5]},{row[6]}"
    )
PY
}

main() {
  cd "$ROOT_DIR"
  log "Few-shot exploration started"
  log "ROOT_DIR=$ROOT_DIR"
  log "PYTHON=$PYTHON"
  check_environment

  if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
    run_smoke_test
  else
    log "Skipping smoke test because SKIP_SMOKE=1"
  fi

  local failures=0

  for k in 1 2 4 6 8 10 12; do
    run_case random_fixed "$k" "$SEED" || failures=$((failures + 1))
  done

  for k in 2 4 5 6 8 10 12; do
    run_case dense_topk "$k" "$SEED" || failures=$((failures + 1))
  done

  for k in 1 2 4 5 6 8 10 12; do
    run_case absa_mmr "$k" "$SEED" || failures=$((failures + 1))
  done

  for k in 4 5 6 8; do
    run_case hard_mix "$k" "$SEED" || failures=$((failures + 1))
  done

  for k in 5 8; do
    run_case manual_fixed_hard "$k" "$SEED" || failures=$((failures + 1))
  done

  log "Exploration completed with failures=$failures"
  print_ranking | tee "$LOG_DIR/fewshot_exploration_ranking.csv"

  if [[ "$failures" -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
