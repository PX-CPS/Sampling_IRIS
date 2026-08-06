#!/bin/bash
#SBATCH --job-name=e10-regenA
#SBATCH --partition=compute
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=%x-%A_%a.out
source ~/nobackup/miniforge3/etc/profile.d/conda.sh
conda activate ipgcs
cd ~/Sampling_IRIS/src
export PYTHONPATH=${GCS_SR:-~/gcs-science-robotics}:$PYTHONPATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 e10_regen219.py --stride 8 --stride-id $SLURM_ARRAY_TASK_ID
