# Train MPRNN

import os
import sys
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from geopy import distance
from torch.utils import data
from torch.autograd import Variable

class PMDataset(data.Dataset):
    """ Dataset generator for PM data. """
    def __init__(self, data, source, thres=None, n_max=None, batch_size=1):
        # prepare the dataset now
        self.source = source
        self.thres = thres
        self.n_max = n_max
        self.bsize = batch_size
        self.adj_matrix_df = self.get_adjacency_matrix()
        self.index_mid = data.index.levels[0]
        self.index_ts = data.index.levels[1]
        self.data = data.values.reshape((self.index_ts.size, self.index_mid.size), order='F')

    def get_adjacency_matrix(self):
        """'source': None/'combined', 'kaiterra' or 'govdata' (default: None)

        'thres': threshold distance in meters (default: None)

        'n_max': max number of neighbors (default: None)

        TODO: Should the graph be undirected or directed?  That is, is it
        ok for sensor A to be in the influence ilst of B, but not
        vice-versa? I think it should directed, esp if we take into
        account wind direction. If it directed, then the adj matrix is no
        longer symmetric.

        """

        if self.source is None:
            self.source = 'combined'
        savepath = os.path.join('data', '{}_distances.csv'.format(self.source))

        if os.path.exists(savepath):
            adj_matrix_df = pd.read_csv(savepath, index_col=[0])
        else:
            locs_df = get_locations(self.source)
            adj_matrix = np.empty((locs_df.index.size, locs_df.index.size), dtype=np.float64)

            for ii, mid in enumerate(locs_df.index):
                coords = (locs_df.loc[mid].Latitude, locs_df.loc[mid].Longitude)
                distances = [distance.distance(coords, (locs_df.loc[m].Latitude, locs_df.loc[m].Longitude)).meters for m in locs_df.index]
                adj_matrix[ii,:] = distances

            adj_matrix_df = pd.DataFrame(adj_matrix, index=locs_df.index, columns=locs_df.index)
            adj_matrix_df.to_csv(savepath, float_format='%.0f')

        # drop invalid sensor
        if self.source is None or self.source in ('combined', 'kaiterra'):
            adj_matrix_df.drop('D385', axis=0, inplace=True)
            adj_matrix_df.drop('D385', axis=1, inplace=True)

        # 'inf' or 'nan' should be used for 'no edges' rather than 0
        # (except for diagonal entries)
        adj_matrix_df[adj_matrix_df == 0] = np.inf
        for ii in range(adj_matrix_df.shape[0]):
            adj_matrix_df.iloc[ii,ii] = 0

        # apply distance threshold for edges
        if self.thres is not None:
            adj_matrix_df[adj_matrix_df >= self.thres] = np.inf

        # max number of neighbors we want to allow (we are keeping the
        # graph undirected for now, so we are cutting off edges both in
        # rows and columns i.e. if we remove e_ij, then we remove e_ji as
        # well).
        if self.n_max is not None:
            if 1 <= self.n_max <= adj_matrix_df.shape[0] - 1:
                idx = adj_matrix_df.values.argsort(axis=1)
                for ii in range(adj_matrix_df.shape[0]):
                    adj_matrix_df.iloc[ii, idx[ii, self.n_max+1:]] = np.inf
                    adj_matrix_df.iloc[idx[ii, self.n_max+1:], ii] = np.inf

        adj_matrix_df[np.isinf(adj_matrix_df.values)] = 0

        return adj_matrix_df

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index,:]

    def generator(self):
        return data.DataLoader(self, batch_size=self.bsize, num_workers=2)

    pass


def get_locations(source=None):

    fpath_kai = os.path.join('data', 'kaiterra', 'kaiterra_locations.csv')
    fpath_gov = os.path.join('data', 'govdata', 'govdata_locations.csv')
    
    if source == 'combined' or source is None:
        locs_df_kai = pd.read_csv(fpath_kai, usecols=[0,2,3,4], index_col=[0])
        locs_df_gov = pd.read_csv(fpath_gov, index_col=[0])
        locs_df = pd.concat([locs_df_kai, locs_df_gov], axis=0, sort=False)
    elif source == 'kaiterra':
        locs_df = pd.read_csv(fpath_kai, usecols=[0,2,3,4], index_col=[0])
    elif source == 'govdata':
        locs_df = pd.read_csv(fpath_gov, index_col=[0])

    locs_df.sort_index(inplace=True)

    return locs_df


def load_data(fpath, sensor, split=0.8, fpath_test=None):

    # load the data
    df = pd.read_csv(fpath, index_col=[0,1], parse_dates=True)
    
    # data is a pd.Series now
    data = df[sensor]
    data.sort_index(inplace=True)
    
    if fpath_test is None:
        data_train, data_test = [], []
        grouped = data.groupby(level=0)
        for name, group in grouped:
            N = int(split * group.shape[0])
            group_train, group_test = group.iloc[:N], group.iloc[N:]
            data_train.append(group_train)
            data_test.append(group_test)
        data_train = pd.concat(data_train, axis=0, sort=False)
        data_test = pd.concat(data_test, axis=0, sort=False)
    else:
        data_train = data
        data_test = pd.read_csv(fpath_test, index_col=[0,1], parse_dates=True)
        data_test = data_test[sensor]

    return data_train, data_test


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


def train(args, logfile=None):

    DENSE = False
    EPS = 120
    LAG = 24 + 1
    HSIZE = 128

    
    
    # use cuda if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # set random seed to 0 for now
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Spatio-temporal LSTM for air quality prediction')
    parser.add_argument('fpath', help='Input data file')
    parser.add_argument('sensor', choices=('pm25', 'pm10'), help='Type of sensory data')

    megroup = parser.add_mutually_exclusive_group()
    megroup.add_argument('--split', type=frac_type, default=0.8, help='Train-test split fraction')
    megroup.add_argument('--test', help='File containing test data')

    parser.add_argument('--history', type=int, default=32, dest='histlen', help='Length of history')
    parser.add_argument('--stride', type=int, default=1, help='Length of stride through data')
    parser.add_argument('--batchsize', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=120, help='Number of epochs for training')
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
    parts = os.path.basename(args.fpath).rsplit('_', 4)
    source = parts[0].split('_')[0]
    resolution = parts[1]
    start_date, end_date = parts[2], parts[3]
    
    savepath = os.path.join('output', 'mprnn', '{}_{}.log'.format(source, args.sensor))
    if not os.path.exists(savepath):
        os.makedirs(os.path.dirname(savepath))
    else:
        if not args.yes:
            confirm = input('Output savepath {} already exists. Overwrite? (y/n) ')
            if confirm.strip().lower() != 'y':
                print('Aborting program. Please rename existing file or provide alternate name for saving.')
                raise SystemExit()

    data_train, data_test = load_data(args.fpath, args.sensor, args.split, args.test)

    dataset_train

    # run the training
    # with open(savepath, 'w') as fout:
    #     prettyprint_args(args, outfile=fout)
    #     train(args, logfile=fout)
