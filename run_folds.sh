#!/bin/bash

for targ in {0..8}
do
	python3 train_folds.py --mode $1 --fold $2 --target $targ \
		--lr 0.00005 --history 360                  \
		--epochs 400 --batch 32 --hidden 256
done
