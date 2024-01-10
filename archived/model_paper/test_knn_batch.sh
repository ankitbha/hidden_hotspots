#!/bin/bash

for kval in {1..10}
do
    # run testing and plotting script
    python test_knn.py $1 $2 $kval v1 --split 0.9
    python test_knn.py $1 $2 $kval v2 --split 0.9
done

