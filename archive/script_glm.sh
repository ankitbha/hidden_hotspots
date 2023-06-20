#!/bin/bash

SOURCE=kaiterra
#MONITOR_ID=113E

for mid in CBC7 A9BE C0A7 72CA BC46 20CA EAC8 113E E8E4 603A; do
    
    for K in 1 2 3 4 5 6 7 8 9 10; do
	
	for hist in 1 2 3 4; do
	    
	    for quant in 0 1; do
		
		for model in 'glm' 'elastic'; do
		    
		    echo K = $K hist = $hist quant = $quant model = $model
		    
		    python train_models.py $SOURCE pm25 $K v2 $quant --model $model --test-monitor-id $mid --history $hist --yes
		    
		done
		
	    done
	    
	done
	
    done
    
done
