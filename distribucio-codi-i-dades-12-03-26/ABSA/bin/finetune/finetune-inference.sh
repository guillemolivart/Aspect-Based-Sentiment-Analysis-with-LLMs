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

PROMPT_FILE="${1:-../prompts/absa_v6.json}"
TEMP="${2:-0.68}"
TOP_P="${3:-0.72}"
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))

python3 finetune-inference.py --prompt-file "$PROMPT_FILE" --temperature "$TEMP" --top-p "$TOP_P" "$@"

deactivate