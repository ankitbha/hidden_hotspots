# ********************************************************************
#
# Author: Shiva R. Iyer, Ulzee An
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
from generate_nndatasets import *
from nets import *
from configs import *
from random import shuffle
from torch.autograd import Variable
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser(description='LSTM for traffic congestion prediction')
    parser.add_argument('--history', type=int, required=True, help='Length of history')
    parser.add_argument('--hidden', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--segment', type=int, required=True, help='Segment to train on; check configs...')
    parser.add_argument('--target', type=int, required=True, help='Target to train for; check configs...')
    parser.add_argument('--stride', type=int, default=2, help='Stride factor')
    args = parser.parse_args()

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    log = SummaryWriter()

    # printing arguments
    print(os.linesep + 'Program arguments: ')
    print('Segment: {}, history: {}, lr: {}, hidden: {}'.format(
        args.segment, args.history, args.lr, args.hidden) + os.linesep)

    use_segment = SEGMENTS[args.segment]
    numsegments = len(use_segment['locations'])
    (train_data, test_data), metadata = create_dataset(use_segment)

    seq = Sequence(
        batchsize=args.batch,
        historylen=args.history,
        numsegments=numsegments,
        hiddensize=args.hidden).to(device).double()

    # mean-squared loss criterion works reasonably for estimating real
    # values
    criterion = nn.MSELoss()
    optimizer = optim.Adam(seq.parameters(), lr=args.lr)

    for eii in range(args.epochs):

        seq.train()
        # TODO: strided indices for more batches
        seqlen = args.history + 1
        stride = args.history // args.stride
        assert stride > 2
        numseqs = 0
        seqinds = []
        for ind in range(0, train_data.shape[1] - seqlen, stride):
            seqinds.append(ind)
            assert ind < train_data.shape[1]
            numseqs += 1
        shuffle(seqinds)
        numbatches = numseqs // args.batch

        losses = []
        for bii in range(numbatches):
            bii *= args.batch
            batch_inds = seqinds[bii:bii+args.batch]
            batch_seq, batch_lbls = nextval_batch(
                train_data,
                args.target,
                batch_inds,
                history=args.history)

            batch_seq = np.transpose(batch_seq, [2, 0, 1])
            batch_seq = Variable(torch.from_numpy(batch_seq), requires_grad=True).to(device)
            batch_lbls = torch.from_numpy(batch_lbls).unsqueeze(1).to(device)

            lstm_state = seq.init_lstms(device=device)

            preds, _ = seq(batch_seq, lstm_state)

            loss = criterion(preds, batch_lbls)
            losses.append(loss.item())
            n_iter = eii*numseqs+bii
            log.add_scalar('train/loss', loss, n_iter)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            sys.stdout.write('[E:%d/%d] B:%d/%d  loss: %.2f          \r' % (
                eii+1, args.epochs,
                bii//args.batch+1, numbatches,
                loss.item(),
            ))
            sys.stdout.flush()

        seq.eval()
        lossavg = 0

        predictions = []
        for ind in range(test_data.shape[1]-seqlen+1):
            batch_seq, batch_lbls = nextval_batch(
                test_data,
                args.target,
                [ind],
                history=args.history)
            batch_seq = np.transpose(batch_seq, [2, 0, 1])
            batch_seq = Variable(torch.from_numpy(batch_seq), requires_grad=False).to(device)
            batch_lbls = torch.from_numpy(batch_lbls).unsqueeze(1).to(device)

            lstm_state = seq.init_lstms(device=device, batch=1)
            preds, _ = seq(batch_seq, lstm_state)

            predval = preds.detach().squeeze().cpu().numpy()
            predictions.append(predval*100)

            loss = criterion(preds, batch_lbls)
            lossavg += loss.detach().cpu().numpy()
        lossavg /= (test_data.shape[1] - seqlen)
        log.add_scalar('test/loss', lossavg, n_iter)
        print('\n   Testing Loss:  %.3f' % lossavg)

        # plot_next('preview/pred_next_%d.png' % n_iter, args.target, test_data, predictions)

        if eii % 10 == 0:
            torch.save(seq, 'checkpoints/%s.pth' % seq.name)
    print()

    log.close()
