#!/bin/bash

for targ in {0..5}
do
	for fold in {0..8}
	do
		python3 train_folds.py --mode $1 --fold $fold --target $targ \
			--lr 0.000005 --history 32 --stride 2                  \
			--epochs 120 --batch 32 --hidden 256
	done
done
