#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --mem=32GB

conda activate /scratch/ab9738/hidden_hotspots/env/;
export PATH=/scratch/ab9738/hidden_hotspots/env/bin:$PATH;
cd /scratch/ab9738/hidden_hotspots/code/field_estimation/
for i in {1..9}; do
    for j in {40..44}; do
        k=$(echo "$i * 0.1" | bc)
        python random_sensors_removed.py $k $j
        python random_sensor_failure.py $k $j
    done
done