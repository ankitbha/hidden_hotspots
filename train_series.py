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


def plot_preview(tag, epoch, seginfo, target, segments, predictions, loss, histlen, criterion, norm=100.0, interval=5, weather=None):
    (start, end, locations) = seginfo
    # imgname = 'preview/%s_pred_%d_epoch_%d.png' % (tag, target, epoch)
    currname = 'images/plots/%s_pred_%d_best.png' % (tag, target)

    tf = time.mktime(datetime.strptime(end, '%m/%d/%Y').timetuple())
    t0 = tf - interval * 60 * len(predictions)
    startstr = datetime.fromtimestamp(t0).strftime("%m/%d/%Y")
    dstr = datetime.fromtimestamp(tf).strftime("%m/%d/%Y")
    # print(startstr, dstr)
    # print()

    legend = []
    plt.figure(figsize=(14, 7))
    for sii, seg in enumerate(segments):
        if sii != target:
            pl = plt.plot(seg*norm, color='#CCCCCC')
            legend.append((pl, locations[sii]))
    pl = plt.plot(segments[target, :]*norm, color='C0')
    legend.append((pl, locations[target] + ' (target)'))
    if weather is not None:
        plt.plot(weather * 50, color='red')

    # mean of all other measurements at time tt
    avgseries = np.zeros(segments.shape[1])
    for sii, seg in enumerate(segments):
        if sii == target: continue
        avgseries += seg
    avgseries /= (segments.shape[0]-1)
    pl = plt.plot(avgseries*norm, color='C2')
    legend.append((pl, 'avg. of non-target'))
    avgloss = 0
    for tii, val in enumerate(avgseries):
        trueval = segments[target, tii]
        loss_tensor = criterion(torch.tensor(val), torch.tensor(trueval))
        avgloss += loss_tensor.cpu().numpy()
    avgloss /= len(avgseries)

    # continuous lstm predictions
    xs = list(range(len(predictions)))
    assert len(xs) == len(predictions)
    pl = plt.plot(np.array(predictions)*norm, color='C1')
    legend.append((pl, 'target preds.'))

    plt.gca().set_title('iter: %d  LSTM NN MSE: %.3f  Naive avg. MSE: %.3f' % (
        epoch,
        loss*norm**2,
        avgloss*norm**2))
    plts, lbls = zip(*legend)
    plts = [pl[0] for pl in plts] # ?? actual ref is first elem?
    plt.legend(plts, lbls, prop={'size': 6})
    plt.xticks([0, len(predictions)-1], [startstr, dstr])
    # plt.savefig(imgname, bbox_inches='tight')

    plt.savefig(currname, bbox_inches='tight')
    plt.close()

    for sii, seg in enumerate(segments):
        plt.figure(figsize=(14, 4))
        plt.title(locations[sii])
        plt.plot(seg*norm)
        plt.savefig('debug/%s.jpg' % locations[sii], bbox_inches='tight')
        plt.close()

    return avgloss

prevloss = 100000000
if __name__ == '__main__':
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser(description='LSTM for traffic congestion prediction')
    parser.add_argument('--history', type=int, required=True, help='Length of history')
    parser.add_argument('--hidden', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=4)
    megroup = parser.add_mutually_exclusive_group(required=True)
    megroup.add_argument('--segment', type=int, help='Segment to train on; (-1) for govdata; check configs...')
    megroup.add_argument('--num-nodes', type=int, help='Number of nodes/monitors to use')
    parser.add_argument('--target', type=int, required=True, help='Target to train for; check configs...')
    parser.add_argument('--stride', type=int, default=2, help='Stride factor')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--once', dest='once', action='store_true')
    parser.set_defaults(once=False)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    log = SummaryWriter()

    if args.segment == -1:
        (train_data, test_data), metadata = create_dataset_gov()
        numsegments = train_data.shape[0]
        start_date, end_date = '08/01/2018','10/01/2018'
        location_names = metadata[2]
        time_interval = 15        # 15 minutes
        tag = 'gov'
    elif args.segment == -2: # gov + our data
        USESEG = 0
        bad_govs = EXCLUDE[USESEG]
        seg = SEGMENTS[USESEG]
        assert args.target < len(seg['locations'])
        (train_data, test_data), (_, _, location_names) = create_dataset_joint(
            seg, exclude=bad_govs)
        numsegments = train_data.shape[0]
        start_date, end_date = seg['start'], seg['end']
        time_interval = 15        # gov data upsamped to 5 mins
        tag = 'joint'
    elif args.segment == -3: # gov + our data
        USESEG = 0
        seg = SEGMENTS[USESEG]
        assert args.target < len(seg['locations'])
        (train_data, test_data), location_names = create_dataset_weather(
            seg)
        numsegments = train_data.shape[0]
        start_date, end_date = seg['start'], seg['end']
        time_interval = 15        # gov data upsamped to 5 mins
        tag = 'wours'
    elif args.segment == -4: # gov + our data
        USESEG = 0
        bad_govs = EXCLUDE[USESEG]
        seg = SEGMENTS[USESEG]
        assert args.target < len(seg['locations'])
        (train_data, test_data), location_names = create_dataset_joint_weather(
            seg, exclude=bad_govs)
        numsegments = train_data.shape[0]
        start_date, end_date = seg['start'], seg['end']
        time_interval = 15        # gov data upsamped to 5 mins
        tag = 'weather'
    elif args.segment > 0:
        # FIXME: update set metadata variables
        use_segment = SEGMENTS[args.segment]
        numsegments = len(use_segment['locations'])
        (train_data, test_data), metadata = create_dataset(use_segment)
        tag = 'ours'
        start_date, end_date = use_segment['start'], use_segment['end']
        location_names = [name[0] for name in use_segment['locations']]
        time_interval = 15
    elif args.segment == None:
        # instead of a particular time segment, we will use 'k'
        # nearest neighbors and train a model
        (train_data, test_data) = create_dataset_knodes(args.num_nodes)
        tag = 'k-nearest'
        

    if args.load is not None:
        print(' Loading:', args.load)
        model = torch.load(args.load)
    else:
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
                loss.item() * 100.0**2,
            ))
            sys.stdout.flush()


        model.eval()
        series_loss = 0
        series_mape = 0
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
            continuous.append(predval)

            loss = criterion(preds, batch_lbls)
            series_loss += loss.detach().cpu().numpy()
            batch_lbls = batch_lbls.detach().squeeze().cpu().numpy()
            preds = preds.detach().squeeze().cpu().numpy()
            series_mape += np.abs((preds - batch_lbls) / batch_lbls)
        series_loss /= (test_data.shape[1] - seqlen)
        series_mape /= (test_data.shape[1] - seqlen)
        series_mape *= 100.0
        log.add_scalar('test/loss', series_loss, n_iter)
        print('\n   Testing Loss:  %.3f      Testing MAPE: %.1f%%' % (series_loss * 100.0**2, series_mape))

        if series_loss < prevloss:
            prevloss = series_loss
            mse_avg = plot_preview(
                tag,
                eii,
                (start_date, end_date, location_names),
                args.target,
                test_data[:-1] if args.segment == -3 else test_data,
                continuous,
                series_loss,
                args.history,
                criterion,
                interval=time_interval,
                weather=test_data[-1] if args.segment == -3 else None)

        logfile = 'outputs/%s_seg_%d_targ_%d.txt' % (tag, args.segment, args.target)
        if eii == 1:
            with open(logfile, 'w') as fl:
                fl.write('')
        # if eii % 5 == 0:
        with open(logfile, 'a') as fl:
            fl.write('EPOCH:%d\n' % eii)
            fl.write('MAPE:%.3f\n' % series_mape)
            fl.write('MSE_TEST:%.3f\n' % (series_loss * 100.0**2))
            fl.write('MSE_AVG:%.3f\n' % (mse_avg * 100.0**2))
            fl.write('MSE_TRAIN:%.3f\n' % (loss.item() * 100.0**2))

        if args.once:
            break

        if eii % 10 == 0:
            torch.save(model, 'checkpoints/%s_targ-%d.pth' % (model.name, args.target))


    print()

    log.close()
