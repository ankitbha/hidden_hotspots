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

from utils import frac_type, prettyprint_args
from datasets import get_adjacency_matrix, get_locations
from models import MPRNN
from models.temporal import RNN
from geopy import distance
from torch.utils import data
from torch.autograd import Variable


def generate_batch(data, histlen=8):
    """Data is an unstacked dataframe -- index is timestamp, columns are
    monitor locations.

    """
    n2t = lambda arr: torch.from_numpy(np.array(arr)).float()
    for ii in range(data.shape[0]-histlen):
        batch = data.iloc[ii:ii+histlen+1,:].values
        if np.isfinite(batch).all():
            X = batch[:histlen,:].T
            Xt = [[n2t(step).unsqueeze(0).unsqueeze(0) for step in node] for node in X]
            Y = batch[1:histlen+1,:]
            Yt = n2t(np.array([Y]))
            yield Xt, Yt


def train(nodes, data, adj, rnn_model=RNN.RNN_MIN,
          mpn_model=MPRNN.MP_THIN, single_mpn=False, val_split=0.8,
          nepochs=10, hiddensize=128, histlen=8, lr=0.001, seed=0,
          savepath_prefix=None, logfile=None):
    """Provide training data and adj info and other training params."""

    # TODO: handle missing data

    # TODO: do cross-validation

    # select only the nodes requested
    data = data[nodes] / 100.0
    adj = adj.loc[nodes, nodes]

    # training and validation sets
    val_start_ind = int(val_split * data.shape[0])
    data_train = data.iloc[:val_start_ind,:]
    data_val = data.iloc[val_start_ind:,:]

    # set random seed to 0 for now
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    mprnn = MPRNN.MPRNN(nodes, adj, hiddensize, rnn_model, mpn_model, single_mpn, verbose=True)

    # use cuda if available
    if torch.cuda.is_available():
        mprnn = mprnn.cuda()
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    mprnn.device = device
    criterion, opt, sch = mprnn.params(lr=lr)

    eval_best = {'rmse' : np.inf, 'mape' : np.inf, 'epoch' : 0}
    savepath_best = None

    max_batches = None

    for epoch in range(nepochs):

        disp_string = '== EPOCH {}/{} =='.format(epoch+1, nepochs)
        print(disp_string)
        if logfile is not None:
            print(disp_string, file=logfile)

        # training
        mprnn.train()
        train_loss_total = 0
        train_count_total = 0
        train_ape_total = 0
        for ii, (Xt, Yt) in enumerate(generate_batch(data_train, histlen)):

            if ii == max_batches:
                break

            preds = mprnn(Xt)
            loss = criterion(preds, Yt)
            train_loss_total += loss.item()
            train_count_total += Yt.numel()

            preds = preds.detach().squeeze().numpy()
            Yt = Yt.detach().squeeze().numpy()
            train_ape_total += np.abs((preds - Yt) / Yt).sum()

            opt.zero_grad()
            loss.backward()
            opt.step()

            sys.stdout.write('\rTraining batch {}/{}'.format(ii+1, data_train.shape[0]-histlen))

        rmse_train = np.sqrt(train_loss_total / train_count_total)
        mape_train = train_ape_total * 100.0 / train_count_total

        disp_string = 'RMSE train: {:.4f}, MAPE train: {:.4f} %'.format(rmse_train, mape_train)
        print(os.linesep + disp_string)

        if logfile is not None:
            print(disp_string, file=logfile)
        
        # evaluation
        mprnn.eval()
        eval_loss_total = 0
        eval_count_total = 0
        eval_ape_total = 0
        with torch.no_grad():
            for ii, (Xt, Yt) in enumerate(generate_batch(data_val), histlen):

                if ii == max_batches:
                    break

                preds = mprnn(Xt)
                loss = criterion(preds, Yt)
                eval_loss_total += loss.item()
                eval_count_total += Yt.numel()

                preds = preds.detach().squeeze().numpy()
                Yt = Yt.detach().squeeze().numpy()
                eval_ape_total += np.abs((preds - Yt) / Yt).sum()

                sys.stdout.write('\rValidation batch {}/{}'.format(ii+1, data_val.shape[0]-histlen))

            rmse_eval = np.sqrt(eval_loss_total / eval_count_total)
            mape_eval = eval_ape_total * 100.0 / eval_count_total

            disp_string = 'RMSE eval: {:.4f}, MAPE eval: {:.4f} %'.format(rmse_eval, mape_eval)
            print(os.linesep + disp_string)

            if logfile is not None:
                print(disp_string, file=logfile)

            if rmse_eval < eval_best['rmse']:
                eval_best['epoch'] = epoch
                eval_best['rmse'] = rmse_eval
                eval_best['mape'] = mape_eval

                if savepath_prefix is not None:
                    if savepath_best is not None:
                        os.unlink(savepath_best + '.pth')
                        os.unlink(savepath_best + '.txt')
                    savepath_best = os.path.join(savepath_prefix + '_E{:03d}'.format(epoch))
                    torch.save(mprnn.state_dict(), savepath_best + '.pth')
                    with open(savepath_best + '.txt', 'w') as fout:
                        fout.write('Eval best: EPOCH={}, RMSE={:.4f}, MAPE={:.4f} %'.format(epoch, rmse_eval, mape_eval))

    return eval_best


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Spatio-temporal LSTM for air quality prediction')
    parser.add_argument('fpath', help='Input data file')
    parser.add_argument('sensor', choices=('pm25', 'pm10'), help='Type of sensory data')
    parser.add_argument('--adj-thres', default=5000, help='Threshold dist (metres) for neighborhood (default: 5000 m)')
    parser.add_argument('--adj-nmax', help='Upper limit on number of neighbors (default: None)')

    megroup = parser.add_mutually_exclusive_group()
    megroup.add_argument('--split', type=frac_type, default=0.8, help='Train-test split fraction')
    megroup.add_argument('--train-end-dt', type=pd.Timestamp, help='End datetime to mark training period')
    megroup.add_argument('--test', help='File containing test data')

    parser.add_argument('--val-split', type=float, default=0.8, help='Split for validation')
    parser.add_argument('--cross-validate', '-cv', action='store_true', default=False, help='Do cross-validation')
    parser.add_argument('--history', type=int, default=24, dest='histlen', help='Length of history (hours)')
    parser.add_argument('--stride', type=int, default=1, help='Length of stride through data')
    parser.add_argument('--batchsize', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')
    args = parser.parse_args()

    # show all options and confirm before beginning execution
    prettyprint_args(args)

    if not args.yes:
        confirm = input('Proceed? (y/n) ')
        if confirm.strip().lower() != 'y':
            print('Confirm by typing in \'y\' or \'Y\' only.')
            raise SystemExit()

    # begin logging
    # parts = os.path.basename(args.fpath).rsplit('_', 4)
    # source = parts[0].split('_')[0]
    # resolution = parts[1]
    # start_date, end_date = parts[2], parts[3]
    savedir = os.path.join('output', 'mprnn')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    logfilepath_prefix = os.path.join(savedir, 'mprnn_{}_{}'.format(args.sensor, pd.Timestamp.now().strftime('%Y%m%d%H%M')))

    # read in the data
    data = pd.read_csv(args.fpath, index_col=[0,1], parse_dates=True)[args.sensor]
    data = data.unstack(level=0)
    data.sort_index(axis=1, inplace=True)
    data.drop('EastArjunNagar_CPCB', axis=1, inplace=True, errors='ignore')
    
    nodes = data.columns
    res = data.index[1] - data.index[0]

    # get the adjacency matrix
    adj_matrix = get_adjacency_matrix('data', thres=args.adj_thres, n_max=args.adj_nmax)

    # split into train and test
    if args.train_end_dt is not None:
        data_train = data.loc[:(args.train_end_dt - res),:]
        data_test = data.loc[args.train_end_dt:,:]
    elif args.test is not None:
        data_train = data
        data_test = pd.read_csv(args.test, index_col=[0,1], parse_dates=True)[args.sensor]
        data_test = data_test.unstack(level=0)
        data_test.sort_index(axis=1, inplace=True)
        data_test.drop('EastArjunNagar_CPCB', axis=1, inplace=True, errors='ignore')
        assert data_test.index[0] > data_train.index[-1]
        assert data_test.shape[1] == data_train.shape[1]
        assert (data_test.columns == data_train.columns).all()
    else:
        train_end_ind = int(args.split*data.shape[0])
        data_train = data.iloc[:train_end_ind,:]
        data_test = data.iloc[train_end_ind:,:]

    # run the training
    with open(logfilepath_prefix + '.log', 'w') as fout:
        prettyprint_args(args, outfile=fout)
        eval_best = train(nodes, data_train, adj_matrix, val_split=args.val_split, savepath_prefix=logfilepath_prefix, logfile=fout)

    print(eval_best)
