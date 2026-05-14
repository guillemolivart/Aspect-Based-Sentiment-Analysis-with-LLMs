#!/bin/bash
#SBATCH -p cuda
#SBATCH -A cuda
#SBATCH --qos=cuda3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH -c 2
#SBATCH --mem=48Gb

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /scratch/nas/1/PDI/mml0/MML.venv/bin/activate

python3 finetune-train.py "$@"

deactivate

# Usage examples and notes:
#
# 1) Run interactively on the node (default: uses few-shot selection)
#    ./finetune-train.sh ../prompts/absa_v8.json train --learning-rate 2e-4 --num-epochs 3 --per-device-train-batch 2 --batch-size 4
#
# 2) Run without few-shot (explicit flag):
#    ./finetune-train.sh ../prompts/absa_v8.json train --learning-rate 2e-4 --num-epochs 3 --per-device-train-batch 2 --batch-size 4 --no-fewshot
#
# 3) Submit via Slurm (example):
#    sbatch finetune-train.sh ../prompts/absa_v8.json train --learning-rate 2e-4 --num-epochs 3 --per-device-train-batch 2 --batch-size 4
#    sbatch finetune-train.sh ../prompts/absa_v8.json train --learning-rate 2e-4 --num-epochs 3 --per-device-train-batch 2 --batch-size 4 --no-fewshot
#
# Output location:
# The script uses the configured `OUTPUT_DIR` (from common.py) and creates a folder named
# `FT.<dataset_stem>.<suffix>.weights`, e.g. `FT.train.fewshot.weights` or `FT.train.simple.weights`.
# Check that directory after the run for checkpoints, `training_args.bin`, `trainer_state.json` and
# the final saved model.