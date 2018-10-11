
import torch
import torch.nn as nn
import numpy as np

class Sequence(nn.Module):
    def __init__(self, historylen, numsegments, hiddensize):
        super(Sequence, self).__init__()
        self.historylen = historylen
        self.numsegments = numsegments
        self.hiddensize = hiddensize
        self.lstm1 = nn.LSTMCell(historylen * numsegments, hiddensize)
        self.lstm2 = nn.LSTMCell(hiddensize, hiddensize)
        self.linear = nn.Linear(hiddensize, 1)

    def forward(self, input, future=0):
        outputs = []
        
        batchsize = input.size(0)
        
        h_t = torch.zeros(batchsize, self.hiddensize, dtype=torch.double)
        c_t = torch.zeros(batchsize, self.hiddensize, dtype=torch.double)
        h_t2 = torch.zeros(batchsize, self.hiddensize, dtype=torch.double)
        c_t2 = torch.zeros(batchsize, self.hiddensize, dtype=torch.double)
        # h_t3 = torch.zeros(batchsize, self.hiddensize, dtype=torch.double)
        # c_t3 = torch.zeros(batchsize, self.hiddensize, dtype=torch.double)

        # torch.chunk() splits the input into as many chunks as given
        # in the argument and returns a tuple of chunks. Each chunk is
        # a tensor of dim [batchsize x inputsize]
        numpoints = input.size(1)
        
        sindices = np.arange(numpoints - future - self.historylen + 1)
        pindices = np.arange(self.historylen, numpoints - future + 1)
        chunkendpoints = zip(sindices, pindices)
        
        for i, (starts, startp) in enumerate(chunkendpoints):
            chunkinput = input[:, starts:startp, :].reshape((batchsize, self.historylen*self.numsegments))
            h_t, c_t = self.lstm1(chunkinput, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            # h_t3, c_t3 = self.lstm2(h_t2, (h_t3, c_t3)) # adding another layer (let's see what this does...)
            output = self.linear(h_t2)
            outputs += [output]

        # input for future predictions are the predicted outputs thus
        # far for the segment in focus, but original inputs for all
        # the other segments in the neighborhood
        if future > 0:
            sindices_future = np.arange(numpoints - future - self.historylen + 1, numpoints - self.historylen + 1)
            pindices_future = np.arange(numpoints - future + 1, numpoints + 1)
            chunkendpoints_future = zip(sindices_future, pindices_future)

            input_future = input.clone()
            input_future[:, sindices_future[0]:pindices_future[0], 0] = torch.stack(outputs[-self.historylen:], 1).squeeze(2)

            # iteration for future prediction
            for i, (starts, startp) in enumerate(chunkendpoints_future):
                chunkinput_future = input_future[:, starts:startp, :].reshape((batchsize, self.historylen*self.numsegments))
                h_t, c_t = self.lstm1(chunkinput_future, (h_t, c_t))
                h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                # h_t3, c_t3 = self.lstm2(h_t2, (h_t3, c_t3))
                output = self.linear(h_t2)
                if startp < numpoints:
                    input_future[:, startp, 0] = output[:,0]
                outputs += [output]
        
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


class SequenceGC(nn.Module):
    def __init__(self, historylen, numsegments, hiddensize, convmat):
        super(SequenceGC, self).__init__()
        self.historylen = historylen
        self.numsegments = numsegments
        self.hiddensize = hiddensize
        self.lstm1 = nn.LSTMCell(historylen*numsegments, hiddensize)
        self.lstm2 = nn.LSTMCell(hiddensize, hiddensize)
        self.linear = nn.Linear(hiddensize, 1)
        self.convmat = convmat

    def forward(self, input, future = 0):
        outputs = []
        
        batchsize = input.size(0)
        
        h_t = torch.zeros(batchsize, self.hiddensize, dtype=torch.double)
        c_t = torch.zeros(batchsize, self.hiddensize, dtype=torch.double)
        h_t2 = torch.zeros(batchsize, self.hiddensize, dtype=torch.double)
        c_t2 = torch.zeros(batchsize, self.hiddensize, dtype=torch.double)
        # h_t3 = torch.zeros(batchsize, self.hiddensize, dtype=torch.double)
        # c_t3 = torch.zeros(batchsize, self.hiddensize, dtype=torch.double)

        # torch.chunk() splits the input into as many chunks as given
        # in the argument and returns a tuple of chunks. Each chunk is
        # a tensor of dim [batchsize x inputsize]

        startindices = np.arange(input.size(1))
        chunkendpoints = zip(startindices, startindices[self.historylen-1:])
        
        for i, (beg, end) in enumerate(chunkendpoints):
            # implement a convolution operation right here!
            chunkinput = input[:, beg:end+1, :].reshape((batchsize, self.historylen*self.numsegments))
            chunkchunks = chunkinput.chunk(self.historylen, dim=1)
            chunkchunks_gc = [torch.from_numpy(np.matmul(self.convmat, inp.numpy().T).T) for inp in chunkchunks]
            chunkinput_gc = torch.cat(chunkchunks_gc, dim=1)
            
            h_t, c_t = self.lstm1(chunkinput_gc, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            # h_t3, c_t3 = self.lstm2(h_t2, (h_t3, c_t3)) # adding another layer (let's see what this does...)
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            # h_t3, c_t3 = self.lstm2(h_t2, (h_t3, c_t3))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs
