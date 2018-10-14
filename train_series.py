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
from datasets import *
from nets import *
from configs import *
from random import shuffle
from torch.autograd import Variable
from tensorboardX import SummaryWriter

def plot_series(imgname, target, segments, predictions, loss, histlen):
    plt.figure(figsize=(14, 7))
    for sii, seg in enumerate(segments):
        if sii != target:
            plt.plot(seg*100, color='gray')
    for sii, seg in enumerate(segments):
        if sii == target:
            plt.plot(seg*100, color='C0')

    # mean of all other measurements at time tt
    avgseries = np.zeros(segments.shape[1])
    for sii, seg in enumerate(segments):
        if sii == target: continue
        avgseries += seg
    avgseries /= (segments.shape[0]-1)
    plt.plot(avgseries*100, color='C2')

    # continuous lstm predictions
    xs = list(range(len(predictions)))
    assert len(xs) == len(predictions)
    plt.plot(predictions, color='C1')

    plt.gca().set_title('Loss: %.3f' % loss)
    plt.savefig(imgname, bbox_inches='tight')
    plt.close()

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
    parser.add_argument('--segment', type=int, required=True, help='Segment to train on; (-1) for all; check configs...')
    parser.add_argument('--target', type=int, required=True, help='Target to train for; check configs...')
    parser.add_argument('--stride', type=int, default=2, help='Stride factor')
    parser.add_argument('--load', type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    log = SummaryWriter()

    # TODO: ability to train on all segments
    use_segment = SEGMENTS[args.segment]
    numsegments = len(use_segment['locations'])
    (train_data, test_data), metadata = create_dataset(use_segment)

    model = Series(
        batchsize=args.batch,
        historylen=args.history,
        numsegments=numsegments,
        hiddensize=args.hidden).to(device).double()

    # mean-squared loss criterion works reasonably for estimating real
    # values
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for eii in range(args.epochs):

        model.train()
        seqlen = args.history
        numseqs = 0
        seqinds = []
        for ind in range(0, train_data.shape[1] - seqlen, args.stride):
            seqinds.append(ind)
            assert ind < train_data.shape[1]
            numseqs += 1
        shuffle(seqinds)
        numbatches = numseqs // args.batch

        losses = []
        for bii in range(numbatches):
            bii *= args.batch
            batch_inds = seqinds[bii:bii+args.batch]
            batch_seq, batch_lbls = series_batch(
                train_data,
                args.target,
                batch_inds,
                history=args.history)

            batch_seq = np.transpose(batch_seq, [2, 0, 1])
            batch_seq = Variable(torch.from_numpy(batch_seq), requires_grad=True) \
                .to(device).double()
            batch_lbls = torch.from_numpy(batch_lbls).unsqueeze(2).to(device)

            lstm_state = model.init_lstms(device=device)

            preds, _ = model(batch_seq, lstm_state)

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

        model.eval()
        lossavg = 0

        continuous = []
        lstm_state = model.init_lstms(device=device, batch=1)
        for ind in range(test_data.shape[1]):
            batch_seq, batch_lbls = series_batch(
                test_data,
                args.target,
                [ind],
                history=1) # feeding 1 at a time
            batch_seq = np.transpose(batch_seq, [2, 0, 1])
            batch_seq = Variable(torch.from_numpy(batch_seq), requires_grad=False) \
                .to(device).double()
            batch_lbls = torch.from_numpy(batch_lbls).unsqueeze(2).to(device)

            preds, lstm_state = model(batch_seq, lstm_state)

            predval = preds.detach().squeeze().cpu().numpy()
            continuous.append(predval*100)

            loss = criterion(preds, batch_lbls)
            lossavg += loss.detach().cpu().numpy()
        lossavg /= (test_data.shape[1] - seqlen)
        log.add_scalar('test/loss', lossavg, n_iter)
        print('\n   Testing Loss:  %.3f' % lossavg)

        plot_series('preview/pred_seriescont_%d.png' % n_iter, args.target, test_data, continuous, lossavg, args.history)
        # break

        if eii % 10 == 0:
            torch.save(model, 'checkpoints/%s.pth' % model.name)
    print()

    log.close()
