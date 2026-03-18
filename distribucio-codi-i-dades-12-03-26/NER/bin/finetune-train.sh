#! /bin/bash
#SBATCH -p cuda
#SBATCH -A cudabig
#SBATCH --qos=cudabig3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH -c 4
#SBATCH --mem=64Gb 


## Use this script to send a job to the cluster with: 
##   sbatch fewshot.sh 10

# activate virtual environment with needed python modules
source /scratch/nas/1/PDI/mml0/MML.venv/bin/activate

# run the few-shot extractor
python3 finetune-train.py $*

# deactivate the virtual environment
deactivate

