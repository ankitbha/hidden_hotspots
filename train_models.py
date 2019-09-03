# ********************************************************************
# 
# Fit other models to the pollution data, and not just using
# nearest neighbors as input, but also "most influential"
# neighbors, based on correlation and clustering.
#
# Author: Shiva R. Iyer
#
# Date: Aug 13, 2019
#
# ********************************************************************

import os
import sys
import argparse
import random
import math
import numpy as np
import matplotlib.pyplot as plt


def prettyprint_args(ns, outfile=sys.stdout):
    print('\nInput argument --', file=outfile)

    for k,v in ns.__dict__.items():
        print('{}: {}'.format(k,v), file=outfile)

    print(file=outfile)
    return


def frac_type(arg):
    try:
        val = float(arg)
        if val <= 0 or val >= 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError('train-test split should be in (0,1)')
    return val


def estimate_glm(monitorid, K, logfile=None):
    
    # set random seed to 0 for now
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    return




if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Machine learning models for air quality prediction')
    parser.add_argument('source', choices=('kaiterra', 'govdata'), help='Source of the data')
    parser.add_argument('sensor', choices=('pm25', 'pm10'), help='Type of sensory data')
    parser.add_argument('numneighbors', type=int, help='Number of nearest neighbors to use (K)')
    parser.add_argument('knn_version', choices=('v1', 'v2'), help='Version 1 or 2')
    parser.add_argument('--history', type=int, default=32, dest='histlen', help='Length of history')
    parser.add_argument('--stride', type=int, default=1, help='Length of stride through data')

    megroup = parser.add_mutually_exclusive_group()
    megroup.add_argument('--split', type=frac_type, default=0.8, help='Train-test split fraction')
    megroup.add_argument('--testdays', help='File containing test days (\'split\' is ignored)')
    
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')
    args = parser.parse_args()
    
    # confirm before beginning execution
    prettyprint_args(args)

    if not args.yes:
        confirm = input('Proceed? (y/n) ')
        if confirm.strip().lower() != 'y':
            print('Confirm by typing in \'y\' or \'Y\' only.')
            raise SystemExit()
    
    # begin logging
    fout = open('output/model_{}_{}_K{:02d}_{}.out'.format(args.source, args.sensor, args.numneighbors, args.knn_version), 'w')
    prettyprint_args(args, outfile=fout)
    
    fout.close()

