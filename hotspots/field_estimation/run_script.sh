#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --time=18:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=st_krig
#SBATCH --output=st_krig_%j.out

conda activate /scratch/ab9738/cctv_pollution/env/;
export PATH=/scratch/ab9738/cctv_pollution/env/bin:$PATH;
cd /scratch/ab9738/pollution_with_sensors/hotspots/field_estimation/
python neural_network_spinterpolation.py
