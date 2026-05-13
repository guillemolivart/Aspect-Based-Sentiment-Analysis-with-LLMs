#!/bin/bash

set -u

# Refined hyperparameter sweep: explore around F (best config so far: 67.71%)
# Focus on small variations and promising new combinations
# Usage: bash run_hyperparam_refined.sh

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

# Current best: F (temp=0.7, top_p=0.8, pp=2.0) -> f1_micro=67.71%

# Variations around best F
run "v6_F_refined_1_temp06" --output-prefix "v6_F_refined_1_temp06" --prompt-file "prompts/absa_v6.json" --temperatures 0.6 --top-ps 0.8 --top-ks 20 --min-ps 0.0 --presence-penalties 2.0 --repetition-penalties 1.0

run "v6_F_refined_2_temp08" --output-prefix "v6_F_refined_2_temp08" --prompt-file "prompts/absa_v6.json" --temperatures 0.8 --top-ps 0.8 --top-ks 20 --min-ps 0.0 --presence-penalties 2.0 --repetition-penalties 1.0

run "v6_F_refined_3_topk30" --output-prefix "v6_F_refined_3_topk30" --prompt-file "prompts/absa_v6.json" --temperatures 0.7 --top-ps 0.8 --top-ks 30 --min-ps 0.0 --presence-penalties 2.0 --repetition-penalties 1.0

run "v6_F_refined_4_topk10" --output-prefix "v6_F_refined_4_topk10" --prompt-file "prompts/absa_v6.json" --temperatures 0.7 --top-ps 0.8 --top-ks 10 --min-ps 0.0 --presence-penalties 2.0 --repetition-penalties 1.0

run "v6_F_refined_5_pp18" --output-prefix "v6_F_refined_5_pp18" --prompt-file "prompts/absa_v6.json" --temperatures 0.7 --top-ps 0.8 --top-ks 20 --min-ps 0.0 --presence-penalties 1.8 --repetition-penalties 1.0

run "v6_F_refined_6_pp22" --output-prefix "v6_F_refined_6_pp22" --prompt-file "prompts/absa_v6.json" --temperatures 0.7 --top-ps 0.8 --top-ks 20 --min-ps 0.0 --presence-penalties 2.2 --repetition-penalties 1.0

run "v6_F_refined_7_topp75" --output-prefix "v6_F_refined_7_topp75" --prompt-file "prompts/absa_v6.json" --temperatures 0.7 --top-ps 0.75 --top-ks 20 --min-ps 0.0 --presence-penalties 2.0 --repetition-penalties 1.0

run "v6_F_refined_8_topp85" --output-prefix "v6_F_refined_8_topp85" --prompt-file "prompts/absa_v6.json" --temperatures 0.7 --top-ps 0.85 --top-ks 20 --min-ps 0.0 --presence-penalties 2.0 --repetition-penalties 1.0

run "v6_F_refined_9_temp065_topp75" --output-prefix "v6_F_refined_9_temp065_topp75" --prompt-file "prompts/absa_v6.json" --temperatures 0.65 --top-ps 0.75 --top-ks 20 --min-ps 0.0 --presence-penalties 2.0 --repetition-penalties 1.0

run "v6_F_refined_10_temp06_pp22" --output-prefix "v6_F_refined_10_temp06_pp22" --prompt-file "prompts/absa_v6.json" --temperatures 0.6 --top-ps 0.8 --top-ks 20 --min-ps 0.0 --presence-penalties 2.2 --repetition-penalties 1.0

run "v6_F_refined_11_pp25" --output-prefix "v6_F_refined_11_pp25" --prompt-file "prompts/absa_v6.json" --temperatures 0.7 --top-ps 0.8 --top-ks 20 --min-ps 0.0 --presence-penalties 2.5 --repetition-penalties 1.0

run "v6_F_refined_12_pp27" --output-prefix "v6_F_refined_12_pp27" --prompt-file "prompts/absa_v6.json" --temperatures 0.7 --top-ps 0.8 --top-ks 20 --min-ps 0.0 --presence-penalties 2.7 --repetition-penalties 1.0

echo "All refined runs queued/completed. Logs: $OUTDIR"
