#!/bin/bash

# Trains P_k | P_i U Q_i
# segment -2 hardcoded to select segment 0 for both ours and gov

for targ in {0..9}
do
	# echo $targ
	# python3 train_series.py --segment -1 \
	# 	--target $targ --lr 0.00001 --history 32 --epochs 200 --batch 96 --hidden 512
	python3 train_series.py --segment -2 --target $targ --lr 0.00005 --history 16 \
		 --epochs 400 --batch 32 --hidden 256
done
