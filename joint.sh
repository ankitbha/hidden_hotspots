#!/bin/bash


for targ in {0..10}
do
	# echo $targ
	# python3 train_series.py --segment -1 \
	# 	--target $targ --lr 0.00001 --history 32 --epochs 200 --batch 96 --hidden 512
	python3 train_series.py --segment -2 --target $targ --lr 0.00005 --history 64 --epochs 200 --batch 64 --hidden 128
done
