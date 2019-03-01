#!/bin/bash

for kval in {1..10}
do
    # run training for each value of K
    python train_knn.py $1 $2 $kval v1 --batch 96 --epochs 20 --max-batches 30 --yes
    python train_knn.py $1 $2 $kval v2 --batch 96 --epochs 20 --max-batches 30 --yes
done

