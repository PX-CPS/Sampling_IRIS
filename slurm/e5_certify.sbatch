#!/bin/bash
#SBATCH --job-name=e5-certify
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-8
#SBATCH --output=%x-%A_%a.out

source ~/nobackup/miniforge3/etc/profile.d/conda.sh
conda activate ipgcs
cd ~/Sampling_IRIS/src
export PYTHONPATH=${GCS_SR:-~/gcs-science-robotics}:$PYTHONPATH

python3 e5_certify_panda55.py --idx $SLURM_ARRAY_TASK_ID
