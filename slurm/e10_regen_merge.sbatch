#!/bin/bash
#SBATCH --job-name=e10-merge
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
source ~/nobackup/miniforge3/etc/profile.d/conda.sh
conda activate ipgcs
cd ~/Sampling_IRIS/src
export PYTHONPATH=${GCS_SR:-~/gcs-science-robotics}:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 e10_regen219.py --merge
