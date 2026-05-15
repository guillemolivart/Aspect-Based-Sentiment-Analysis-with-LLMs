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
# 1) Strong default LoRA SFT run (no few-shot in training; output-only loss)
#    ./finetune-train.sh ../prompts/absa_v6.json train \
#      --learning-rate 1e-4 --num-epochs 5 --max-length 3072 \
#      --per-device-train-batch 1 --gradient-accumulation-steps 8
#
# 2) QLoRA 4-bit version requested by the project statement
#    ./finetune-train.sh ../prompts/absa_v6.json train \
#      --load-in-4bit --learning-rate 1e-4 --num-epochs 5 --max-length 3072 \
#      --per-device-train-batch 1 --gradient-accumulation-steps 8
#
# 3) Higher-capacity LoRA ablation
#    ./finetune-train.sh ../prompts/absa_v6.json train \
#      --lora-r 32 --lora-alpha 64 --learning-rate 5e-5 --num-epochs 5 \
#      --max-length 3072 --per-device-train-batch 1 --gradient-accumulation-steps 8
#
# Output location:
# The script uses the configured `OUTPUT_DIR` (from common.py) and creates a folder named
# `outputs/finetune/FT.<dataset>.<prompt>.<mode>.<shots>.<targets>.r<rank>.lr<lr>.weights`.
# Check that directory after the run for checkpoints, `training_args.bin`, `trainer_state.json` and
# the final saved model.
