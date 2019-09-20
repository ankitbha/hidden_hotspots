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
import itertools
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model

from glob import glob
from tqdm import tqdm

from datasets import create_dataset_knn, create_dataset_knn_sensor, create_dataset_knn_testdays


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


def estimate_glm_reg(model, args, logfile=None):

    K = args.numneighbors
        
    # set random seed to 0 for now
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    if args.testdays == None:
        if args.monitorid is None:
            dataset_df, trainrefs, testrefs = create_dataset_knn(args.source, args.sensor, K, args.knn_version, 
                                                                 args.stride, args.histlen, args.split)
        else:
            dataset_df, trainrefs, testrefs = create_dataset_knn_sensor(args.source, args.monitorid, args.sensor, K, args.knn_version, 
                                                                        args.stride, args.histlen, args.split)
    else:
        dataset_df, trainrefs, testrefs = create_dataset_knn_testdays(args.source, args.sensor, K, args.knn_version, 
                                                                      args.testdays, args.stride, args.histlen)
    
    if len(testrefs) == 0:
        return np.nan, np.nan
    
    n_features = K*args.histlen if args.knn_version == 'v1' else 3*K*args.histlen
    
    N = len(trainrefs) + len(testrefs)

    print('Total number of samples:', N)
    print('Number of features:', n_features)
    print('No of train refs: {}, no of test refs: {}'.format(len(trainrefs), len(testrefs)))
    
    X = np.empty((N, n_features))
    y = np.empty(N)
    for rii, ref in tqdm(enumerate(itertools.chain(trainrefs, testrefs)), total=N, desc='Collecting all samples from refs'):
        _, start = ref
        batch = dataset_df.iloc[start:start+args.histlen,:]
        X[rii,:] = batch.iloc[:,1:].values.ravel()
        y[rii] = batch.iloc[-1,0]

    print('All valid?', np.isfinite(X).all() & np.isfinite(y).all())
    X_train = X[:len(trainrefs), :]
    y_train = y[:len(trainrefs)]

    print('Number of training samples:', len(y_train))
    
    X_test = X[len(trainrefs):, :]
    y_test = y[len(trainrefs):]

    print('Number of test samples:', len(y_test))

    #from sklearn import linear_model

    #clf = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(np.mean((y_pred - y_test)**2)) * 100
    mape = np.mean(np.abs((y_pred - y_test) / y_test))
    print('RMSE = {:.2f}, MAPE = {:.2f}'.format(rmse, mape))
    
    return rmse, mape



def estimate_glm_excLF(model, args, logfile=None):
    '''
    Estimate the excess over a baseline.
    '''
    
    K = args.numneighbors
    
    # set random seed to 0 for now
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    if args.testdays == None:
        if args.monitorid is None:
            dataset_df, trainrefs, testrefs = create_dataset_knn(args.source, args.sensor, K, args.knn_version, 
                                                                 args.stride, args.histlen, args.split)
            raise Exception('provide a monitorid for now!')
        else:
            dataset_df, trainrefs, testrefs = create_dataset_knn_sensor(args.source, args.monitorid, args.sensor, K, args.knn_version, 
                                                                        args.stride, args.histlen, args.split)
            filterflist = glob('figures/freq_components/location_wise/{}_{}*_T_filter_thres03CPD.csv'.format(args.monitorid, args.sensor))
            dataset_df['target_lowpass'] = np.nan
            for fpath in filterflist:
                filter_df = pd.read_csv(fpath, index_col=[0], usecols=[0,2])
                values_lp = filter_df[args.sensor + '_lowpass'].values / 100.0
                dataset_df.loc[filter_df.index, 'target_lowpass'] = values_lp
                dataset_df.loc[filter_df.index, 'target'] -= values_lp
            #dataset_df.loc[:, 'target'] -= dataset_df['target_lowpass'].values
            #print(dataset_df.columns)
            #print(sum(np.isfinite(dataset_df.target.values)), sum(np.isfinite(dataset_df.target_lowpass.values)))
    else:
        dataset_df, trainrefs, testrefs = create_dataset_knn_testdays(args.source, args.sensor, K, args.knn_version, 
                                                                      args.testdays, args.stride, args.histlen)
        raise Exception('\"testdays\" not supported right now!')

    if len(testrefs) == 0:
        return np.nan, np.nan
    
    n_features = K*args.histlen if args.knn_version == 'v1' else 3*K*args.histlen
    
    N = len(trainrefs) + len(testrefs)
    
    print('Total number of samples:', N)
    print('Number of features:', n_features)
    print('No of train refs: {}, no of test refs: {}'.format(len(trainrefs), len(testrefs)))
    
    X = np.empty((N, n_features))
    y = np.empty(N)
    y_lp = np.empty(N)
    for rii, ref in tqdm(enumerate(itertools.chain(trainrefs, testrefs)), total=N, desc='Collecting all samples from refs'):
        _, start = ref
        batch = dataset_df.iloc[start:start+args.histlen,:-1]
        X[rii,:] = batch.iloc[:,1:].values.ravel()
        y[rii] = batch.iloc[-1,0]
        y_lp[rii] = batch.iloc[-1,-1]

    #print(y[~np.isfinite(y)])

    print('All valid?', np.isfinite(X).all() & np.isfinite(y).all())
    X_train = X[:len(trainrefs), :]
    y_train = y[:len(trainrefs)]
    y_lp_train = y_lp[:len(trainrefs)]

    print('Number of training samples:', len(y_train))
    
    X_test = X[len(trainrefs):, :]
    y_test = y[len(trainrefs):]
    y_lp_test = y[len(trainrefs):]

    print('Number of test samples:', len(y_test))

    #from sklearn import linear_model

    #clf = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_test_true = y_test + y_lp_test
    y_pred_true = y_pred + y_lp_test

    print('All positive?', np.all(y_test_true > 0))
    
    #rmse = np.sqrt(np.mean((y_pred - y_test)**2)) * 100
    #mape = np.mean(np.abs((y_pred - y_test) / y_test))
    rmse = np.sqrt(np.mean((y_pred_true - y_test_true)**2)) * 100
    mape = np.mean(np.abs((y_pred_true - y_test_true) / y_test_true))
    print('RMSE = {:.2f}, MAPE = {:.2f}'.format(rmse, mape))
    
    return rmse, mape




if __name__=='__main__':

    quantity_dict = {0: 'reg',
                     1: 'excLF'}
    
    func_dict = {0: estimate_glm_reg, 1: estimate_glm_excLF}
    
    parser = argparse.ArgumentParser(description='Machine learning models for air quality prediction')
    parser.add_argument('source', choices=('kaiterra', 'govdata'), help='Source of the data')
    parser.add_argument('sensor', choices=('pm25', 'pm10'), help='Type of sensory data')
    parser.add_argument('numneighbors', type=int, help='Number of nearest neighbors to use (K)')
    parser.add_argument('knn_version', choices=('v1', 'v2'), help='Version 1 or 2')
    parser.add_argument('quantity', type=int, choices=(0,1), help='0: regular, 1: excess over LF baseline')
    parser.add_argument('--history', type=int, default=1, dest='histlen', help='Length of history')
    parser.add_argument('--stride', type=int, default=1, help='Length of stride through data')
    parser.add_argument('--test-monitor-id', dest='monitorid', help='Test only at one location')
    parser.add_argument('--model', choices=('glm', 'lasso', 'ridge', 'elastic'), default='glm')
    parser.add_argument('--append-results', action='store_true', default=True, help='Append results to file if it exists')

    megroup = parser.add_mutually_exclusive_group()
    megroup.add_argument('--split', type=frac_type, default=0.8, help='Train-test split fraction')
    megroup.add_argument('--testdays', help='File containing test days (\'split\' is ignored)')
    
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')
    args = parser.parse_args()

    if args.histlen <= 0:
        raise argparse.ArgumentTypeError('history length should >= 1')
    
    # confirm before beginning execution
    prettyprint_args(args)
    
    if not args.yes:
        confirm = input('Proceed? (y/n) ')
        if confirm.strip().lower() != 'y':
            print('Confirm by typing in \'y\' or \'Y\' only.')
            raise SystemExit()
    
    # begin logging
    savedir = 'output/output_models/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    if args.monitorid is not None:
        savepath = savedir + 'model_{}_{}_{}_{}_knn_{}_{}.csv'.format(args.model, quantity_dict[args.quantity], args.source, args.sensor, args.knn_version, args.monitorid)
    else:
        savepath = savedir + 'model_{}_{}_{}_{}_knn_{}.csv'.format(args.model, quantity_dict[args.quantity], args.source, args.sensor, args.knn_version)
        raise Exception('please provide monitorid for now!')

    fout = None
    
    if os.path.exists(savepath):
        if not args.append_results:
            if not args.yes:
                confirm = input('Output savepath {} already exists. Overwrite? (y/n) '.format(savepath))
                if confirm.strip().lower() != 'y':
                    print('Aborting program. Please rename existing file or provide alternate name for saving.')
                    raise SystemExit()
            fout = open(savepath, 'w')
            fout.write('K,histlen,rmse,mape' + os.linesep)
        else:
            fout = open(savepath, 'a')
    else:
        fout = open(savepath, 'w')
        fout.write('K,histlen,rmse,mape' + os.linesep)
    
    #prettyprint_args(args, outfile=fout)

    if args.model == 'glm':
        model = linear_model.LinearRegression()
    elif args.model == 'lasso':
        model = linear_model.LassoCV(cv=5)
    elif args.model == 'ridge':
        model = linear_model.RidgeCV(cv=5)
    elif args.model == 'elastic':
        l1_ratio_grid = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
        model = linear_model.ElasticNetCV(l1_ratio=l1_ratio_grid, cv=5)
    
    rmse, mape = func_dict[args.quantity](model, args)
    
    fout.write('{},{},{:.3f},{:.3f}'.format(args.numneighbors, args.histlen, rmse, mape) + os.linesep)
    
    fout.close()
