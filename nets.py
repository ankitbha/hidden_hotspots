# ********************************************************************
#
# Author: Shiva R. Iyer, Ulzee An
#
# Date: Oct 10, 2018
#
# ********************************************************************

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class NextVal(nn.Module):
    name = 'nextval-v1'
    def __init__(self, batchsize, historylen, numsegments, hiddensize):
        super(NextVal, self).__init__()
        self.batchsize = batchsize
        self.historylen = historylen
        self.numsegments = numsegments
        self.hiddensize = hiddensize
        self.lstm = nn.LSTM(hiddensize, hiddensize, num_layers=2)
        self.prelstm = nn.Sequential(*[
            nn.Linear(numsegments, hiddensize),
            # nn.ReLU(),
        ])
        self.postlstm = nn.Sequential(*[
            nn.ReLU(),
            nn.Linear(hiddensize, 1),
        ])

    def init_lstms(self, device=None, grad=True, batch=None):
        # h_0 of shape (num_layers * num_directions, batch, hidden_size):
        #   tensor containing the initial hidden state for each element in the batch.
        # c_0 of shape (num_layers * num_directions, batch, hidden_size):
        #   tensor containing the initial cell state for each element in the batch.
        bsize = self.batchsize if batch is None else batch
        # FIXME: batchsize == numsegments for now...
        h_t = torch.zeros(2, bsize, self.hiddensize, requires_grad=grad, dtype=torch.double).to(device)
        c_t = torch.zeros(2, bsize, self.hiddensize, requires_grad=grad, dtype=torch.double).to(device)
        # h_t2 = torch.zeros(1, self.batchsize, self.hiddensize)
        # c_t2 = torch.zeros(1, self.batchsize, self.hiddensize)

        # h_t3 = torch.zeros(batchsize, self.hiddensize, dtype=torch.double)
        # c_t3 = torch.zeros(batchsize, self.hiddensize, dtype=torch.double)

        return (h_t, c_t)

    def forward(self, input, lstm_states, future=0):
        (h_t, c_t) = lstm_states

        # print(input.size())
        x = self.prelstm(input)

        # print(x.device, h_t.device, c_t.device)
        # Expected input: (histlen x batchsize x dense_t)
        x, (h_t, c_t) = self.lstm(x, (h_t, c_t))
        # x = F.relu(x)

        last_x = x[-1, :, :]
        x = self.postlstm(last_x)

        return x, (h_t, c_t)

    @staticmethod
    def demo():
        from torch import optim

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = NextVal(batchsize=2, historylen=5, numsegments=7, hiddensize=25) \
            .to(device)
        lstm_states = model.init_lstms(device=device)

        # rand_input = torch.randn(2, 7, 5)  # (batchsize x # segs x histlen)
        rand_input = torch.randn(5, 2, 7, requires_grad=True).to(device)  # (histlen x batchsize x  x #segs)
        print('Input:', rand_input.size())
        pred, lstm_states = model(rand_input, lstm_states)

        print('Output:', pred.size())

        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        rand_labels = torch.randn(2, 1).to(device)  # (histlen x batchsize x  x #segs)
        criterion = nn.MSELoss()(rand_labels, pred)
        criterion.backward()
        optimizer.step()

class Series(nn.Module):
    name = 'series-v1'
    def __init__(self, batchsize, historylen, numsegments, hiddensize):
        super(Series, self).__init__()

        self.batchsize = batchsize
        self.historylen = historylen
        self.numsegments = numsegments
        self.hiddensize = hiddensize

        self.lstm = nn.LSTM(hiddensize, hiddensize, num_layers=2)
        self.prelstm = nn.Sequential(*[
            nn.Linear(numsegments, hiddensize),
            # nn.ReLU(),
        ])
        self.postlstm = nn.Sequential(*[
            nn.ReLU(),
            nn.Linear(hiddensize, 1),
        ])

    def init_lstms(self, device=None, grad=True, batch=None):
        bsize = self.batchsize if batch is None else batch

        h_t = torch.zeros(2, bsize, self.hiddensize, requires_grad=grad, dtype=torch.double).to(device)
        c_t = torch.zeros(2, bsize, self.hiddensize, requires_grad=grad, dtype=torch.double).to(device)

        return (h_t, c_t)

    def forward(self, input, lstm_states, future=0):
        (h_t, c_t) = lstm_states

        # print(input.size())
        x = self.prelstm(input)

        # print(x.device, h_t.device, c_t.device)
        # Expected input: (histlen x batchsize x dense_t)
        x, (h_t, c_t) = self.lstm(x, (h_t, c_t))
        # x = F.relu(x)

        x = self.postlstm(last_x)

        return x, (h_t, c_t)

    @staticmethod
    def demo():
        from torch import optim

        model = Sequence(batchsize=2, historylen=5, numsegments=7, hiddensize=25)
        lstm_states = model.init_lstms()

        # rand_input = torch.randn(2, 7, 5)  # (batchsize x # segs x histlen)
        rand_input = torch.randn(5, 2, 7, requires_grad=True)  # (histlen x batchsize x  x #segs)
        print('Input:', rand_input.size())
        pred, lstm_states = model(rand_input, lstm_states)

        print('Output:', pred.size())

        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        rand_labels = torch.randn(2, 1)  # (histlen x batchsize x  x #segs)
        criterion = nn.MSELoss()(rand_labels, pred)
        criterion.backward()
        optimizer.step()

if __name__ == '__main__':

    NextVal.demo()

