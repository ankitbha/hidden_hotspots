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


# if quantity == 0
def estimate_glm_reg(model, source, sensor, numneighbors, knn_version, histlen=0, stride=1, split=0.8, testdays=None, monitorid=None):
    
    # only one of them has to be provided (XOR)
    assert ((split is None) ^ (testdays is None))
    
    # set random seed to 0 for now
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    if testdays is None:
        if monitorid is None:
            dataset_df, trainrefs, testrefs = create_dataset_knn(source, sensor, numneighbors, knn_version, 
                                                                 stride, histlen+1, split)
        else:
            dataset_df, trainrefs, testrefs = create_dataset_knn_sensor(source, monitorid, sensor, numneighbors, knn_version, 
                                                                        stride, histlen+1, split)
    else:
        dataset_df, trainrefs, testrefs = create_dataset_knn_testdays(source, sensor, numneighbors, knn_version, 
                                                                      testdays, stride, histlen+1)
        raise Exception('\"testdays\" not supported right now!')
    
    if len(testrefs) == 0:
        return np.nan, np.nan
    
    n_features = numneighbors*(histlen+1) if knn_version == 'v1' else 3*numneighbors*(histlen+1)
    
    N = len(trainrefs) + len(testrefs)
    
    print('Total number of samples:', N)
    print('Number of features:', n_features)
    print('No of train refs: {}, no of test refs: {}'.format(len(trainrefs), len(testrefs)))
    
    X = np.empty((N, n_features))
    y = np.empty(N)
    for rii, ref in tqdm(enumerate(itertools.chain(trainrefs, testrefs)), total=N, desc='Collecting all samples from refs'):
        _, start = ref
        batch = dataset_df.iloc[start:start+histlen+1,:]
        X[rii,:] = batch.iloc[:,1:].values.ravel()
        y[rii] = batch.iloc[-1,0]
    
    print('All valid?', np.isfinite(X).all() & np.isfinite(y).all())
    X_train = X[:len(trainrefs), :]
    y_train = y[:len(trainrefs)]
    
    print('Number of training samples:', len(y_train))
    
    X_test = X[len(trainrefs):, :]
    y_test = y[len(trainrefs):]
    
    print('Number of test samples:', len(y_test))
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(np.mean((y_pred - y_test)**2)) * 100
    mape = np.mean(np.abs((y_pred - y_test) / y_test))
    print('RMSE = {:.2f}, MAPE = {:.2f}'.format(rmse, mape))
    
    return rmse, mape


# if quantity == 1
def estimate_glm_excess(model, source, sensor, numneighbors, knn_version, histlen=0, stride=1, split=0.8, testdays=None, monitorid=None):
    '''
    Estimate the excess over a baseline.
    '''
    
    # only one of them has to be provided (XOR)
    assert ((split is None) ^ (testdays is None))
    
    # set random seed to 0 for now
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    if testdays is None:
        if monitorid is None:
            dataset_df, trainrefs, testrefs = create_dataset_knn(source, sensor, numneighbors, knn_version, 
                                                                 stride, histlen+1, split)
            raise Exception('provide a monitorid for now!')
        else:
            dataset_df, trainrefs, testrefs = create_dataset_knn_sensor(source, monitorid, sensor, numneighbors, knn_version, 
                                                                        stride, histlen+1, split)
            filterflist = glob('figures/freq_components/filter/thres03CPD/{0}/{0}_{1}*_T_filter_thres03CPD.csv'.format(monitorid, sensor))
            dataset_df['target_lowpass'] = np.nan
            for fpath in filterflist:
                filter_df = pd.read_csv(fpath, index_col=[0], usecols=[0,2])
                values_lp = filter_df[sensor + '_lowpass'].values / 100.0
                dataset_df.loc[filter_df.index, 'target_lowpass'] = values_lp
                dataset_df.loc[filter_df.index, 'target'] -= values_lp
            #dataset_df.loc[:, 'target'] -= dataset_df['target_lowpass'].values
            #print(dataset_df.columns)
            #print(sum(np.isfinite(dataset_df.target.values)), sum(np.isfinite(dataset_df.target_lowpass.values)))
    else:
        dataset_df, trainrefs, testrefs = create_dataset_knn_testdays(source, sensor, numneighbors, knn_version, 
                                                                      testdays, stride, histlen+1)
        raise Exception('\"testdays\" not supported right now!')

    print(dataset_df.columns)
    
    if len(testrefs) == 0:
        return np.nan, np.nan
    
    n_features = numneighbors*(histlen+1) if knn_version == 'v1' else 3*numneighbors*(histlen+1)
    
    N = len(trainrefs) + len(testrefs)
    
    print('Total number of samples:', N)
    print('Number of features:', n_features)
    print('No of train refs: {}, no of test refs: {}'.format(len(trainrefs), len(testrefs)))
    
    X = np.empty((N, n_features))
    y = np.empty(N)
    y_lp = np.empty(N)
    for rii, ref in tqdm(enumerate(itertools.chain(trainrefs, testrefs)), total=N, desc='Collecting all samples from refs'):
        _, start = ref
        batch = dataset_df.iloc[start:start+histlen+1,:-1]
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
    y_lp_test = y_lp[len(trainrefs):]
    
    print('Number of test samples:', len(y_test))
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    y_test_true = y_test + y_lp_test
    y_pred_true = y_pred + y_lp_test
    
    print('All positive?', np.all(y_test_true > 0))
    
    rmse = np.sqrt(np.mean((y_pred_true - y_test_true)**2)) * 100
    mape = np.mean(np.abs((y_pred_true - y_test_true) / y_test_true))
    print('RMSE = {:.2f}, MAPE = {:.2f}'.format(rmse, mape))
    
    return rmse, mape



if __name__=='__main__':
    
    quantity_dict = {0: 'reg',
                     1: 'excess'}
    
    func_dict = {0: estimate_glm_reg, 1: estimate_glm_excess}
    
    parser = argparse.ArgumentParser(description='Machine learning models for air quality prediction')
    parser.add_argument('source', choices=('kaiterra', 'govdata'), help='Source of the data')
    parser.add_argument('sensor', choices=('pm25', 'pm10'), help='Type of sensory data')
    parser.add_argument('numneighbors', type=int, help='Number of nearest neighbors to use (K)')
    parser.add_argument('knn_version', choices=('v1', 'v2'), help='Version 1 or 2')
    parser.add_argument('quantity', type=int, choices=(0,1), help='0: regular, 1: excess over LF baseline')
    parser.add_argument('--history', type=int, default=0, dest='histlen', help='Length of history')
    parser.add_argument('--stride', type=int, default=1, help='Length of stride through data')
    parser.add_argument('--test-monitor-id', dest='monitorid', help='Test only at one location')
    parser.add_argument('--model', choices=('glm', 'lasso', 'ridge', 'elastic'), default='glm')
    parser.add_argument('--append-results', action='store_true', default=True, help='Append results to results file if it exists')
    parser.add_argument('--save-dir', default=os.path.join('results', 'linear_models'), help='Directory where results will be stored')
    
    megroup = parser.add_mutually_exclusive_group()
    megroup.add_argument('--split', type=frac_type, default=0.8, help='Train-test split fraction')
    megroup.add_argument('--testdays', help='File containing test days (\'split\' is ignored)')
    
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')
    args = parser.parse_args()
    
    if args.histlen < 0:
        raise argparse.ArgumentTypeError('history length should >= 0')
    
    # confirm before beginning execution
    prettyprint_args(args)
    
    if not args.yes:
        confirm = input('Proceed? (y/n) ')
        if confirm.strip().lower() != 'y':
            print('Confirm by typing in \'y\' or \'Y\' only.')
            raise SystemExit()
    
    # begin logging
    #savedir = 'output/output_models/'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    if args.monitorid is not None:
        savepath = os.path.join(args.save_dir, 'model_{}_{}_{}_{}_knn_{}_{}.csv'.format(args.model, quantity_dict[args.quantity], args.source, args.sensor, args.knn_version, args.monitorid))
    else:
        savepath = os.path.join(args.save_dir, 'model_{}_{}_{}_{}_knn_{}.csv'.format(args.model, quantity_dict[args.quantity], args.source, args.sensor, args.knn_version))
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

    # def estimate_glm_reg(model, source, sensor, numneighbors, knn_version, histlen=0, stride=1, split=0.8, testdays=None, monitorid=None, logfile=None):
    rmse, mape = func_dict[args.quantity](model,
                                          args.source, args.sensor, args.numneighbors, args.knn_version,
                                          args.histlen, args.stride, args.split, args.testdays, args.monitorid)
    
    fout.write('{},{},{:.3f},{:.3f}'.format(args.numneighbors, args.histlen, rmse, mape) + os.linesep)
    
    fout.close()
