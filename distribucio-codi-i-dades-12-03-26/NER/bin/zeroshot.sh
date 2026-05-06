#! /bin/bash
#SBATCH -p cuda
#SBATCH -A cuda
#SBATCH --qos=cuda3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH -c 2
#SBATCH --mem=48Gb 


## Use this script to send a job to the cluster with: 
##   sbatch fewshot.sh 10

# activate virtual environment with needed python modules
source /scratch/nas/1/PDI/mgl0/MLL.venv/bin/activate

# run the few-shot extractor
python3 zeroshot.py 

# deactivate the virtual environment
deactivate

