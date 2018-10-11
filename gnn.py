# ********************************************************************
# 
# Implement a neural consisting of CNN and LSTM elements for
# spatiotemporal pollution forecasting.
#
# Input: 2-D array MxN, where M is number of locations, and N is
# length of history
#
# Layers: 1-D Conv -> 1-D Pooling -> 1-D Conv -> 1-D Pooling -> LSTM
#
# Author: Shiva R. Iyer
#
# Date: Aug 15, 2018
#
# ********************************************************************

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse


if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    
    hiddensize = 25

    parser = argparse.ArgumentParser(description='LSTM for traffic congestion prediction')
    parser.add_argument('segment', help='Name of segment')
    parser.add_argument('--history', type=int, required=True, help='Length of history')
    parser.add_argument('--hops', type=int, choices=(0,1,2), default=0, help='Number of hops for neighborhood')
    parser.add_argument('--ahead', type=int, default=0, help='Number of timestamps to predict in the future')
    parser.add_argument('--gc', action='store_true', help='Whether to perform graph convolution (valid only if hops > 0)')
    args = parser.parse_args()

    # printing arguments
    print(os.linesep + 'Program arguments: ')
    print('Segment: {}, history: {}, hops: {}, ahead: {}, gc: {}'.format(args.segment, args.history, args.hops, args.ahead, args.gc) + os.linesep)
    
    # segname = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    segname = args.segment

    # data file
    if args.hops > 0:
        infilepath = 'train-data/{}-{}hop.pt'.format(segname, args.hops)
    else:
        infilepath = 'train-data/{}.pt'.format(segname)

    if args.gc:
        # *** HARDCODING THIS FOR NOW!! THIS IS NOT CORRECT FOR GENERIC INPUT. ***
        # 
        # |   |   |
        # V   V   V
        # 
        # We assume that if hop is 1, then given segment has 1 successor
        # and 1 predecessor. If hop is 2, then 2 successors and 2
        # predecessors.
        
        adjmat = None
        weightmat = None
        
        if args.hops == 1:
            adjmat = np.array([[1, 1, 1],
                               [1, 1, 0],
                               [1, 0, 1]])
            weightmat = np.array([[1, 1, 1],
                                  [1, 1, 2],
                                  [1, 2, 1]])
        elif args.hops == 2:
            adjmat = np.array([[1, 1, 0, 1, 0],
                               [1, 1, 1, 0, 0],
                               [0, 1, 1, 0, 0],
                               [1, 0, 0, 1, 1],
                               [0, 0, 0, 1, 1]])
            weightmat = np.array([[1, 1, 2, 1, 2],
                                  [1, 1, 1, 2, 3],
                                  [2, 1, 1, 3, 4],
                                  [1, 2, 3, 1, 1],
                                  [2, 3, 4, 1, 1]])
        else:
            raise Exception("With GC, values supported for \"hops\" is only 1 or 2.")

    
    # load data and split into training and testing
    arr = torch.load(infilepath)

    if len(arr.shape) == 2:
        arr = np.expand_dims(arr, axis=2)
    
    # 'input' is data from first 60 days and 'target' is predicting
    # next point given previous two points

    numsegments = arr.shape[2]
    
    # training
    input = torch.from_numpy(arr[:60, :-1, :])
    target = torch.from_numpy(arr[:60, args.history:, 0])
    
    # test
    test_input = torch.from_numpy(arr[60:, :-1, :])
    test_target = torch.from_numpy(arr[60:, args.history:, 0])

    # build the model
    if args.gc:
        convmat = adjmat * weightmat
        seq = SequenceGC(args.history, numsegments, hiddensize, convmat)
        nnname = 'lstmgc'
    else:
        seq = Sequence(args.history, numsegments, hiddensize)
        nnname = 'lstm'
    
    seq.double()

    # mean-squared loss criterion works reasonably for estimating real
    # values
    criterion = nn.MSELoss()
    
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    # optimizer = optim.RMSprop(seq.parameters())
    
    numiter = 20
    loss_mat = np.empty((numiter, 2))

    saverootdir = 'results/{}/'.format(segname)
    # savedir = os.path.join(saverootdir, 'lstm-{}hop/{}-history-{}-ahead/'.format(args.hops, args.history, args.ahead))
    os.makedirs(saverootdir, exist_ok=True)

    ys_iter = []
    
    # begin to train
    for i in range(numiter):
        print('STEP: ', i)
        losslist = []
        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            losslist.append(loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        train_loss = min(losslist)
        loss_mat[i,0] = train_loss

        # begin to predict, no need to track gradient here
        with torch.no_grad():
            pred = seq(test_input, future=args.ahead)
            loss = criterion(pred, test_target)
            test_loss = loss.item()
            print('test loss:', test_loss)
            loss_mat[i,1] = test_loss
            y = pred.detach().numpy()
            ys_iter.append(y)
        
    min_train_loss, min_test_loss = loss_mat.min(axis=0)
    min_train_loss_iter, min_test_loss_iter = loss_mat.argmin(axis=0)
    stat1 = 'min_train_loss = {}, min_train_loss_iter = {}'.format(min_train_loss, min_train_loss_iter + 1)
    stat2 = 'min_test_loss = {}, min_test_loss_iter = {}'.format(min_test_loss, min_test_loss_iter + 1)
    print(stat1)
    print(stat2)

    # save the predictions with min test loss
    np.savetxt(os.path.join(saverootdir, '{}-predict-alldays-{}-{}hop-h{}-a{}-stopiter{:02d}.csv'.format(segname, nnname, args.hops, args.history, args.ahead, min_test_loss_iter)), ys_iter[min_test_loss_iter][:,1:], fmt='%f', delimiter=',')
    np.savetxt(os.path.join(saverootdir, '{}-actual-alldays-h{}.csv'.format(segname, args.history)), test_target.numpy()[:,:-1], fmt='%f', delimiter=',')
    
    with open(os.path.join(saverootdir, '{}-losses-{}-{}hop-h{}-a{}.txt'.format(segname, nnname, args.hops, args.history, args.ahead)), 'w') as fout:
        fout.write(stat1 + '\n')
        fout.write(stat2 + '\n')
        fout.write('Min train loss,test loss\n')
        np.savetxt(fout, loss_mat, delimiter=',', fmt='%f')
