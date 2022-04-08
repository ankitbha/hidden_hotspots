import argparse
import collections
import time
import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import get_adjacency_matrix, get_locations
from utils import frac_type, prettyprint_args

import dgl
from dgl.data.tree import SSTDataset

from tree_lstm import TreeLSTM


def generate_batch(data, histlen=8):
    """Data is an unstacked dataframe -- index is timestamp, columns are
    monitor locations.

    """
    nodes = data.columns
    u, v = th.tensor(nodes*len(nodes)), th.tensor(np.array([[i]*len(nodes) for i in nodes]).flatten())
    g = dgl.graph((u,v))
    n2t = lambda arr: torch.from_numpy(np.array(arr)).float()
    for ii in range(data.shape[0]-histlen):
        batch = data.iloc[ii:ii+histlen+1,:].values
        if np.isfinite(batch).all():
            X = batch[:histlen,:].T
            Xt = [[n2t(step).unsqueeze(0).unsqueeze(0) for step in node] for node in X]
            Y = batch[1:histlen+1,:]
            Yt = n2t(np.array([Y]))
            yield g, Xt, Yt

def main(args):
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)

    best_epoch = -1
    best_dev_rmse = 1.0

    cuda = args.gpu >= 0
    device = th.device('cuda:{}'.format(args.gpu)) if cuda else th.device('cpu')
    if cuda:
        th.cuda.set_device(args.gpu)


    data = pd.read_csv(args.fpath, index_col=[0,1], parse_dates=True)[args.sensor]
    data = data.unstack(level=0)
    data.sort_index(axis=1, inplace=True)
    data.drop('EastArjunNagar_CPCB', axis=1, inplace=True, errors='ignore')

    data = data / 100.0
    adj = get_adjacency_matrix('data', thres=args.adj_thres, n_max=args.adj_nmax)

    # training and validation sets
    val_start_ind = int(val_split * data.shape[0])
    data_train = data.iloc[:val_start_ind,:]
    data_val = data.iloc[val_start_ind:,:]

    nodes = data.columns
    res = data.index[1] - data.index[0]

    models = []
    for i in range(len(nodes)):
        model = TreeLSTM(len(nodes),
                         args.x_size,
                         args.h_size,
                         args.dropout).to(device)
        models.append(model)

    print(models)
    params_ex_emb =[x for x in list(model.parameters()) if x.requires_grad and x.size(0)!=trainset.vocab_size for model in models]

    for p in params_ex_emb:
        if p.dim() > 1:
            INIT.xavier_uniform_(p)

    optimizer = optim.Adagrad([
        {'params':params_ex_emb, 'lr':args.lr, 'weight_decay':args.weight_decay}])

    dur = []
    criterion = nn.MSELoss(reduction='sum').cuda()
    for epoch in range(args.epochs):
        t_epoch = time.time()
        for step, batch in enumerate(generate_batch(data_train)):
            g = batch[0].to(device)
            n = g.number_of_nodes()
            h = th.zeros((n, args.h_size)).to(device)
            c = th.zeros((n, args.h_size)).to(device)
            if step >= 3:
                t0 = time.time() # tik

            for q, model in enumerate(models):
                preds = model(batch[1], g, h, c)
                # each node has its own model, which is backpropagated through that node's predictions
                loss = criterion(preds, batch[2][:, q], reduction='sum')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step >= 3:
                    dur.append(time.time() - t0) # tok

                if step > 0 and step % args.log_every == 0:
                    print("Epoch {:05d} | Step {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
                        epoch, step, loss.item(), np.mean(dur)))
        print('Epoch {:05d} training time {:.4f}s'.format(epoch, time.time() - t_epoch))

        rmse = []
        for step, batch in enumerate(generate_batch(data_val)):
            g = batch[0].to(device)
            n = g.number_of_nodes()
            with th.no_grad():
                h = th.zeros((n, args.h_size)).to(device)
                c = th.zeros((n, args.h_size)).to(device)
                for q, model in enumerate(models):
                    preds = model(batch[1], g, h, c)
                    loss = criterion(preds, batch[2][:, q], reduction='sum')
                    rmse.append(loss.item())

        dev_rmse = 1.0*np.sum([x for x in rmse])/len(rmse)
        print("Epoch {:05d} | Loss {:.4f}".format(epoch, dev_rmse))

        if dev_rmse > best_dev_rmse:
            best_dev_rmse = dev_rmse
            best_epoch = epoch
            for q, model in enumerate(models):
                th.save(model.state_dict(), 'best_{}_{}.pkl'.format(args.seed, q))
        else:
            if best_epoch <= epoch - 10:
                break

        # lr decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(1e-5, param_group['lr']*0.99) #10
            print(param_group['lr'])

    # test
    rmse = []
    for q, model in enumerate(models):
        model.load_state_dict(th.load('best_{}_{}.pkl'.format(args.seed, q)))
        for step, batch in enumerate(generate_batch(data_val)):
            g = batch[0].to(device)
            n = g.number_of_nodes()
            with th.no_grad():
                h = th.zeros((n, args.h_size)).to(device)
                c = th.zeros((n, args.h_size)).to(device)
                for q, model in enumerate(models):
                    preds = model(batch[1], g, h, c)
                    loss = criterion(preds, batch[2][:, q], reduction='sum')
                    rmse.append(loss.item())

    dev_rmse = 1.0*np.sum([x for x in rmse])/len(rmse)
    print("Test Epoch {:05d} | Loss {:.4f}".format(epoch, dev_rmse))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--hidden', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--child-sum', action='store_true')
    parser.add_argument('--x-size', type=int, default=300)
    parser.add_argument('--h-size', type=int, default=150)
    parser.add_argument('--log-every', type=int, default=5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    args = parser.parse_args()
    print(args)
    main(args)
