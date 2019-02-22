
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import threading

from datasets import create_dataset_knn, batch_knn
from nets import Series
from torch.autograd import Variable

def prettyprint_args(ns):
    print(os.linesep + 'Input argument --')

    for k,v in ns.__dict__.items():
        print('{}: {}'.format(k,v))

    print()
    return

def frac(arg):
    try:
        val = float(arg)
        if val <= 0 or val >= 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError('train-test split should be in (0,1)')
    return


def train(K, args):
    ''' Training code to be run inside a thread. '''
    
    # use cuda if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # set random seed to 0 for now
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_df, trainrefs, testrefs = create_dataset_knn(K, args.knn_version, args.datesuffix, args.sensor,
                                                         args.stride, args.histlen, args.split)

    if args.knn_version == 'v1':
        model = Series(batchsize=args.batch,
                       historylen=args.histlen,
                       numsegments=K + 1,
                       hiddensize=args.hidden).to(device).double()
    else:
        model = Series(batchsize=args.batch,
                       historylen=args.histlen,
                       numsegments=3*K + 1,
                       hiddensize=args.hidden).to(device).double()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    for eii in range(args.epochs):
        
        np.random.shuffle(trainrefs)
        losses = []
        nBatches = len(trainrefs)//args.batch
        for bii in range(0, len(trainrefs) - args.batch, args.batch):
            model.train()
            
            batchrefs = trainrefs[bii:bii+args.batch]
            # batch_seq, batch_lbls = knodes_batch(dataset, batchrefs, pad=10)
            batch_seq, batch_lbls = batch_knn(dataset_df, batchrefs, args.histlen)

            batch_seq = np.transpose(batch_seq, [2, 0, 1])
            batch_seq = Variable(torch.from_numpy(batch_seq), requires_grad=True).to(device).double()
            batch_lbls = torch.from_numpy(batch_lbls).unsqueeze(2).to(device)

            lstm_state = model.init_lstms(device=device)

            preds, _ = model(batch_seq, lstm_state)

            loss = criterion(preds, batch_lbls)
            losses.append(loss.item())
            n_iter = eii*(nBatches)+bii
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write('[E:%d/%d] B:%d/%d  loss: %.2f          \r' % (
                eii+1, args.epochs,
                bii//args.batch+1, nBatches,
                loss.item() * 100.0**2,
            ))
            sys.stdout.flush()

        nBatches = len(testrefs)//args.batch
        tloss = 0
        tmape = 0
        print()
        for bii in range(0, len(testrefs) - args.batch, args.batch):
            model.eval()

            batchrefs = testrefs[bii:bii+args.batch]
            # batch_seq, batch_lbls = knodes_batch(dataset, batchrefs, mode='test', pad=10)
            batch_seq, batch_lbls = batch_knn(dataset_df, batchrefs, args.histlen)

            batch_seq = np.transpose(batch_seq, [2, 0, 1])
            batch_seq = Variable(torch.from_numpy(batch_seq), requires_grad=True).to(device).double()
            batch_lbls = torch.from_numpy(batch_lbls).unsqueeze(2).to(device)

            lstm_state = model.init_lstms(device=device)

            preds, _ = model(batch_seq, lstm_state)
            loss = criterion(preds, batch_lbls)
            tloss += loss.detach().cpu().numpy()

            batch_lbls = batch_lbls.detach().squeeze().cpu().numpy()
            preds = preds.detach().squeeze().cpu().numpy()
            batch_lbls[batch_lbls == 0] = 0.01
            tmape += np.mean(np.abs((preds - batch_lbls) / batch_lbls))

            sys.stdout.write('Testing %d/%d       \r' % (
                bii//args.batch+1, nBatches
            ))
    #         break
        tmape /= nBatches
        tmape *= 100.0
        tloss /= nBatches
        print('   Testing Loss:  %.3f      Testing MAPE: %.1f%%' % (
            tloss * 100.0**2, tmape))
    pass




if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Spatio-temporal LSTM for air quality prediction')
    parser.add_argument('maxneighbors', type=int, help='Max number of nearest neighbors to use (K)')
    parser.add_argument('knn_version', choices=('v1', 'v2'), help='Version 1 or 2')
    parser.add_argument('--data-version', default='2019_Feb_05', dest='datesuffix', help='Version of raw data to use')
    parser.add_argument('--sensor', choices=('pm25', 'pm10'), default='pm25', help='Type of sensory data')
    parser.add_argument('--history', type=int, default=32, dest='histlen', help='Length of history')
    parser.add_argument('--stride', type=int, default=1, help='Length of stride through data')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=120, help='Number of epochs for training')
    parser.add_argument('--split', type=frac, default=0.8, help='Train-test split fraction')
    args = parser.parse_args()
    
    # confirm before beginning execution
    prettyprint_args(args)
    
    confirm = input('Proceed? (y/n) ')
    if confirm.strip().lower() != 'y':
        print('Confirm by typing in \'y\' or \'Y\' only.')
        raise SystemExit()
    
    # create threads for training
    threads_list = []
    for K in range(1, args.maxneighbors+1):
        thread = threading.Thread(target=train, args=(K, args), name='{}-NN trainer'.format(K))
        thread.start()
        threads_list.append(thread)

    for thread in threads_list:
        thread.join()
    


    # for ii in range(5, 10):
    #     knearest(ii)



def knearest(kk):
    _, (test_series, test_labels) = create_dataset_knodes_sensorid(
        LIDS[3], kk, split=0.8)

    model.eval()
    series_loss = 0
    series_mape = 0
    continuous = []
    missing = 0
    lstm_state = model.init_lstms(device=device, batch=1)
    for ind in range(test_series.shape[1]):
        raw_batch_seq = np.array([test_series[:, ind:ind+1]])
        batch_seq = np.zeros((1, 10, 1))
        batch_seq[:, :raw_batch_seq.shape[1], :] = raw_batch_seq
        batch_lbls = np.array([[test_labels[ind]]])

    #     print(batch_seq.shape, batch_lbls.shape)

        batch_seq = np.transpose(batch_seq, [2, 0, 1])
        batch_seq = Variable(torch.from_numpy(batch_seq), requires_grad=False)             .to(device).double()
        batch_lbls = torch.from_numpy(batch_lbls).unsqueeze(2).to(device)

        preds, lstm_state = model(batch_seq, lstm_state)

        predval = preds.detach().squeeze().cpu().numpy()
        continuous.append(predval)

        if np.isnan(batch_lbls.item()):
            continue

        loss = criterion(preds, batch_lbls)
        series_loss += loss.detach().cpu().numpy()
        batch_lbls = batch_lbls.detach().squeeze().cpu().numpy()
        batch_lbls[batch_lbls == 0] = 0.01
        preds = preds.detach().squeeze().cpu().numpy()
        ape = np.abs((preds - batch_lbls) / batch_lbls)
    #     print(ape, preds, batch_lbls)
        series_mape += ape
    series_loss /= (test_series.shape[1] - missing)
    series_mape /= (test_series.shape[1] - missing)
    series_mape *= 100.0
    series_loss *= 100**2

    plt.figure(figsize=(14, 5))
    plt.gca().set_title('K: %d  MSE: %.2f   MAPE: %.2f' % (kk+1, series_loss, series_mape))
    for ii in range(kk):
        plt.plot(test_series[ii, :] * 1000, color='#AAAAAA')
    plt.plot(np.array(continuous) * 100)
    plt.plot(np.array(test_labels) * 100)
    plt.show();
    plt.close()
