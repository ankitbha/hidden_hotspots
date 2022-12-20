#!/bin/bash

#SBATCH --job-name=jupyter_cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16GB
#SBATCH --time=12:00:00

#source ~/.bashrc
conda activate /scratch/ab9738/epod-nyu-delhi-pollution/env/;
export PATH=/scratch/ab9738/epod-nyu-delhi-pollution/env/bin:$PATH;
cd /scratch/ab9738/

#port=$(shuf -i 10000-65500 -n 1)
port=8881
/usr/bin/ssh -N -f -R $port:localhost:$port log-2
/usr/bin/ssh -N -f -R $port:localhost:$port log-1

jupyter lab --no-browser --port $port --notebook-dir /scratch/ab9738 --NotebookApp.token='' --NotebookApp.password=''
