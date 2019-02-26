
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# import threading

from datasets import create_dataset_knn, batch_knn
from nets import Series
from torch.autograd import Variable

def prettyprint_args(ns, outfile=sys.stdout):
    print('\nInput argument --', file=outfile)

    for k,v in ns.__dict__.items():
        print('{}: {}'.format(k,v), file=outfile)

    print(file=outfile)
    return

def frac(arg):
    try:
        val = float(arg)
        if val <= 0 or val >= 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError('train-test split should be in (0,1)')
    return


def train(K, args, logfile=None):
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

            batchcount = bii//args.batch
            if batchcount == args.max_batches:
                break
            
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

            disp_string = '[E:{}/{}] B:{}/{}  loss: {:.2f}\n'.format(
                eii+1, args.epochs,
                batchcount+1, nBatches,
                loss.item() * 100.0**2)
            sys.stdout.write(disp_string)
            
            if logfile != None:
                logfile.write(disp_string)
        
        nBatches = len(testrefs)//args.batch
        tloss = 0
        tmape = 0
        
        for bii in range(0, len(testrefs) - args.batch, args.batch):

            sys.stdout.write('\r')

            batchcount = bii//args.batch
            if batchcount == args.max_batches:
                break
            
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

            disp_string = 'Testing {}/{}'.format(batchcount+1, nBatches)
            sys.stdout.write(disp_string)
            sys.stdout.flush()
        
        logfile.write(disp_string)
        logfile.flush()
        
        tmape /= nBatches
        tmape *= 100.0
        tloss /= nBatches
        disp_string = '\n   Testing Loss:  {:.3f}      Testing MAPE: {:.1f}% \n'.format(tloss * 100.0**2, tmape)
        sys.stdout.write(disp_string)
        
        if logfile != None:
            logfile.write(disp_string)
    
    # save the trained model
    torch.save(model.state_dict(), 'models/model_K{:02d}_{}_{}_{}.pth'.format(K, args.knn_version, args.datesuffix, args.sensor))
    return




if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Spatio-temporal LSTM for air quality prediction')
    parser.add_argument('numneighbors', type=int, help='Number of nearest neighbors to use (K)')
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
    parser.add_argument('--max-batches', type=int, help='Max number of batches')
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
    fout = open('output/model_K{:02d}_{}_{}_{}.out'.format(args.numneighbors, args.knn_version, args.datesuffix, args.sensor), 'w')
    prettyprint_args(args, outfile=fout)
    
    # create threads for training
    # threads_list = []
    # for K in range(1, args.maxneighbors+1):
    #     thread = threading.Thread(target=train, args=(K, args), name='{}-NN trainer'.format(K))
    #     thread.start()
    #     threads_list.append(thread)
    
    # for thread in threads_list:
    #     thread.join()
    
    train(args.numneighbors, args, logfile=fout)
    fout.close()