#!/bin/bash

for targ in {0..8}
do
	for fold in {0..8}
	do
		python3 train_folds.py --mode $1 --fold $fold --target $targ \
			--lr 0.000005 --history 96                  \
			--epochs 100 --batch 64 --hidden 256
	done
done
