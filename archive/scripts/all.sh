
#!/bin/bash

targs='0 1 2 3 4 5 6 7 8'

seg='0'
# targs='4 8'
# targs='5 6'

for tii in $targs;
do
    cmd="python3 -u train_series.py --segment $seg --target $tii --lr 0.0001 --history 64 --epochs 200 --batch 96 --hidden 128"
    echo $cmd
    eval $cmd
done
