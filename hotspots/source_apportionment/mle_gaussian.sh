#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=15
#SBATCH --time=18:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=mle_gaussian
#SBATCH --output=slurm_mle_%j.out

conda activate /scratch/ab9738/cctv_pollution/env/;
export PATH=/scratch/ab9738/cctv_pollution/env/bin:$PATH;
cd /scratch/ab9738/pollution_with_sensors/hotspots/source_apportionment/
python mle_gaussian.py