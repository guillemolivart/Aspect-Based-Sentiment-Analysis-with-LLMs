#!/bin/bash

set -u

# Run a set of hyperparam sweep commands sequentially and log each run.
# Usage: bash run_hyperparam_sweeps_v6.sh

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE_DIR"

OUTDIR=outputs/hyperparam_sweep
mkdir -p "$OUTDIR"

run() {
  prefix="$1"
  shift
  LOG="$OUTDIR/${prefix}.run.log"
  echo "===== RUN: ${prefix} =====" | tee -a "$LOG"
  echo "Start: $(date -u +'%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOG"
  python bin/hyperparam_sweep_qwen.py "$@" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  echo "Exit code: $rc" | tee -a "$LOG"
  echo "End: $(date -u +'%Y-%m-%d %H:%M:%S UTC')" | tee -a "$LOG"
  echo "" >> "$LOG"
}

# A official
run "v6_A_official" --output-prefix "v6_A_official" --prompt-file "prompts/absa_v6.json" --temperatures 1.0 --top-ps 1.0 --top-ks 20 --min-ps 0.0 --presence-penalties 2.0 --repetition-penalties 1.0

# B lower presence
run "v6_B_lower_pres" --output-prefix "v6_B_lower_pres" --prompt-file "prompts/absa_v6.json" --temperatures 1.0 --top-ps 1.0 --top-ks 20 --min-ps 0.0 --presence-penalties 1.0 --repetition-penalties 1.0

# C medium presence
run "v6_C_med_pres" --output-prefix "v6_C_med_pres" --prompt-file "prompts/absa_v6.json" --temperatures 1.0 --top-ps 1.0 --top-ks 20 --min-ps 0.0 --presence-penalties 1.5 --repetition-penalties 1.0

# D no presence sanity
run "v6_D_no_pres" --output-prefix "v6_D_no_pres" --prompt-file "prompts/absa_v6.json" --temperatures 1.0 --top-ps 1.0 --top-ks 20 --min-ps 0.0 --presence-penalties 0.0 --repetition-penalties 1.0

# E conservative
run "v6_E_cons" --output-prefix "v6_E_cons" --prompt-file "prompts/absa_v6.json" --temperatures 0.7 --top-ps 0.8 --top-ks 20 --min-ps 0.0 --presence-penalties 1.5 --repetition-penalties 1.0

# F conservative official presence
run "v6_F_cons_off_pres" --output-prefix "v6_F_cons_off_pres" --prompt-file "prompts/absa_v6.json" --temperatures 0.7 --top-ps 0.8 --top-ks 20 --min-ps 0.0 --presence-penalties 2.0 --repetition-penalties 1.0

# G balanced
run "v6_G_bal" --output-prefix "v6_G_bal" --prompt-file "prompts/absa_v6.json" --temperatures 0.8 --top-ps 0.95 --top-ks 20 --min-ps 0.0 --presence-penalties 1.5 --repetition-penalties 1.0

# H slightly narrower official
run "v6_H_narrow_off" --output-prefix "v6_H_narrow_off" --prompt-file "prompts/absa_v6.json" --temperatures 1.0 --top-ps 0.95 --top-ks 20 --min-ps 0.0 --presence-penalties 2.0 --repetition-penalties 1.0

echo "All runs queued/completed. Logs: $OUTDIR"
