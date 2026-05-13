#! /bin/bash
#SBATCH -p cuda
#SBATCH -A cuda
#SBATCH --qos=cuda3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH -c 2
#SBATCH --mem=48Gb 

source /scratch/nas/1/PDI/mml0/MML.venv/bin/activate

python3 zeroshot.py "$@"

deactivate
