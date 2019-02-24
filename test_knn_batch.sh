#!/bin/bash

for kval in {1..10}
do
    # run testing and plotting script
    python test_knn.py $kval v1
    python test_knn.py $kval v2
done

