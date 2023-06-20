#!/bin/bash

#testdaysfilepath=data/testdays_kaiterra_set1_seed9.csv
testdaysfilepath=data/testdays_govdata_set1_seed6.csv

for kval in {1..10}
do
    # run training for each value of K
    # python train_knn.py $1 $2 $kval v1 --batch 96 --epochs 20 --max-batches 30 --split 0.9 --yes
    python train_knn.py $1 $2 $kval v1 --batch 96 --epochs 10 --max-batches 60 --testdays $testdaysfilepath --yes
    # python train_knn.py $1 $2 $kval v2 --batch 96 --epochs 20 --max-batches 30 --split 0.9 --yes
    python train_knn.py $1 $2 $kval v2 --batch 96 --epochs 10 --max-batches 60 --testdays $testdaysfilepath --yes
done

