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
    # parser.add_argument('segment', help='Name of segment')
    parser.add_argument('--history', type=int, required=True, help='Length of history')
    parser.add_argument('--hidden', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--segment', type=int, required=True, help='Segment to train on; check configs...')
    parser.add_argument('--target', type=int, required=True, help='Target to train for; check configs...')
    # FIXME: batchsize = # segments right now...
    # parser.add_argument('--hops', type=int, choices=(0,1,2), default=0, help='Number of hops for neighborhood')
    # parser.add_argument('--ahead', type=int, default=0, help='Number of timestamps to predict in the future')
    # parser.add_argument('--gc', action='store_true', help='Whether to perform graph convolution (valid only if hops > 0)')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log = SummaryWriter()

    # printing arguments
    print(os.linesep + 'Program arguments: ')
    print('Segment: {}, history: {}, lr: {}, hidden: {}'.format(
        args.segment, args.history, args.lr, args.hidden) + os.linesep)

    use_segment = SEGMENTS[args.segment]
    numsegments = len(use_segment['locations'])
    dataset, metadata = create_dataset(use_segment)
	# segments, labels = get_batch(dataset, inds, histlen=3)

    # plt.figure(figsize=(14, 7))
    # plt.plot(dataset[0, :])
    # plt.savefig('dump.png')
    # plt.close()

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

        # TODO: strided indices for more batches
        seqlen = args.history + 1
        numseqs = dataset.shape[1] // seqlen
        seqinds = [int(ii * seqlen) for ii in range(numseqs)]
        shuffle(seqinds)

        losses = []
        for bii in range(0, len(seqinds)-args.batch, args.batch):
            batch_inds = seqinds[bii:bii+args.batch]
            batch_seq, batch_lbls = target_batch(
                dataset,
                args.target,
                batch_inds,
                history=args.history)

            # plt.figure(figsize=(14, 14))
            # for jj in range(4):
            #     plt.subplot(4, 1, jj+1)
            #     plt.gca().set_title(batch_lbls[jj]*100)
            #     for ii in range(len(use_segment['locations'])):
            #         plt.plot(batch_seq[jj, ii, :]*100)
            # plt.savefig('dump.png')
            # plt.close()
            # assert False

            # torch rnn format:
            #   numpy (batch x datapoints x sequence)
            #   torch (sequnce x batch x datapoints)
            batch_seq = np.transpose(batch_seq, [2, 0, 1])
            batch_seq = Variable(torch.from_numpy(batch_seq), requires_grad=True).to(device)
            batch_lbls = torch.from_numpy(batch_lbls).unsqueeze(1).to(device)

            lstm_state = seq.init_lstms(device=device)

            preds, _ = seq(batch_seq, lstm_state)

            loss = criterion(preds, batch_lbls)
            losses.append(loss.item())
            log.add_scalar('train/loss', loss, eii*numseqs+bii)
            # log.add_scalar('test/loss', loss, eii*numseqs+bii)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            sys.stdout.write('[E:%d/%d] B:%d/%d  loss: %.2f  \r' % (
                eii+1, args.epochs,
                (bii+1)//args.batch, len(seqinds)//args.batch,
                loss.item(),
            ))
            sys.stdout.flush()
    print()

    # writer.export_scalars_to_json("./all_scalars.json")
    log.close()


        # optimizer.step(closure)
        # train_loss = min(losslist)
        # loss_mat[i,0] = train_loss

        # # begin to predict, no need to track gradient here
        # with torch.no_grad():
        #     pred = seq(test_input, future=args.ahead)
        #     loss = criterion(pred, test_target)
        #     test_loss = loss.item()
        #     print('test loss:', test_loss)
        #     loss_mat[i,1] = test_loss
        #     y = pred.detach().numpy()
        #     ys_iter.append(y)

    # min_train_loss, min_test_loss = loss_mat.min(axis=0)
    # min_train_loss_iter, min_test_loss_iter = loss_mat.argmin(axis=0)
    # stat1 = 'min_train_loss = {}, min_train_loss_iter = {}'.format(min_train_loss, min_train_loss_iter + 1)
    # stat2 = 'min_test_loss = {}, min_test_loss_iter = {}'.format(min_test_loss, min_test_loss_iter + 1)
    # print(stat1)
    # print(stat2)

    # # save the predictions with min test loss
    # np.savetxt(os.path.join(saverootdir, '{}-predict-alldays-{}-{}hop-h{}-a{}-stopiter{:02d}.csv'.format(segname, nnname, args.hops, args.history, args.ahead, min_test_loss_iter)), ys_iter[min_test_loss_iter][:,1:], fmt='%f', delimiter=',')
    # np.savetxt(os.path.join(saverootdir, '{}-actual-alldays-h{}.csv'.format(segname, args.history)), test_target.numpy()[:,:-1], fmt='%f', delimiter=',')

    # with open(os.path.join(saverootdir, '{}-losses-{}-{}hop-h{}-a{}.txt'.format(segname, nnname, args.hops, args.history, args.ahead)), 'w') as fout:
    #     fout.write(stat1 + '\n')
    #     fout.write(stat2 + '\n')
    #     fout.write('Min train loss,test loss\n')
    #     np.savetxt(fout, loss_mat, delimiter=',', fmt='%f')
