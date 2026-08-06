#!/bin/bash
#SBATCH --job-name=e9-verify
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
source ~/nobackup/miniforge3/etc/profile.d/conda.sh
conda activate ipgcs
cd ~/Sampling_IRIS/src
export PYTHONPATH=${GCS_SR:-~/gcs-science-robotics}:$PYTHONPATH
python3 e9_verify_gcsstar.py
