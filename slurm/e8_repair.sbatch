#!/bin/bash
#SBATCH --job-name=e8-repair
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --array=0-1
#SBATCH --output=%x-%A_%a.out
source ~/nobackup/miniforge3/etc/profile.d/conda.sh
conda activate ipgcs
cd ~/Sampling_IRIS/src
export PYTHONPATH=${GCS_SR:-~/gcs-science-robotics}:$PYTHONPATH
PRESETS=(panda55 biman219)
python3 e8_repair_lib.py --preset ${PRESETS[$SLURM_ARRAY_TASK_ID]}
