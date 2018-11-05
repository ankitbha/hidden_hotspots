#!/bin/bash


for targ in {0..10}
do
	# echo $targ
	python3 train_series.py --segment -1 \
		--target $targ --lr 0.00001 --history 32 --epochs 200 --batch 96 --hidden 512
done
