#!/bin/bash
#SBATCH --job-name=e8-e7R
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-1
#SBATCH --output=%x-%A_%a.out
source ~/nobackup/miniforge3/etc/profile.d/conda.sh
conda activate ipgcs
cd ~/Sampling_IRIS/src
export PYTHONPATH=${GCS_SR:-~/gcs-science-robotics}:$PYTHONPATH
PRESETS=(panda55R biman219R)
python3 e7_highd_audit.py --preset ${PRESETS[$SLURM_ARRAY_TASK_ID]} \
  --out out/e7_${PRESETS[$SLURM_ARRAY_TASK_ID]}.json
