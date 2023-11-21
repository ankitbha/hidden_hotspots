#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --mem=32GB

conda activate /scratch/ab9738/cctv_pollution/env/;
export PATH=/scratch/ab9738/cctv_pollution/env/bin:$PATH;
cd /scratch/ab9738/pollution_with_sensors/hotspots/field_estimation/
python interpolation_ext.py 0.1
python interpolation_ext.py 0.2
python interpolation_ext.py 0.3
python interpolation_ext.py 0.4
python interpolation_ext.py 0.5