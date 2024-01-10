
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.GRNN import *


class MP_THIN(nn.Module):
	def __init__(self, hsize):
		super(MP_THIN, self).__init__()

		self.lossy = False
		self.msg_op = nn.Linear(hsize*2, hsize)
		self.upd_op = nn.Linear(hsize*2, hsize)

	def msg(self, hself, hother):
		hcat = torch.cat([hself, hother], -1)
		return self.msg_op(hcat)

	def upd(self, hself, msg):
		hcat = torch.cat([hself, msg], -1)
		return self.upd_op(hcat)


class MP_DENSE(MP_THIN):
	def __init__(self, hsize):
		super().__init__(hsize)

		self.msg_op = nn.Sequential(
			nn.Linear(hsize*2, hsize),
			nn.ReLU(),
			nn.Linear(hsize, hsize),
		)
		self.upd_op = nn.Sequential(
			nn.Linear(hsize*2, hsize),
			nn.ReLU(),
			nn.Linear(hsize, hsize),
		)
		self.lossy = nn.Sequential(
			nn.ReLU(),
			nn.Dropout(0.5),
		)


class MPRNN(GRNN):
	'''Instantiates one RNN per location in input graph.
	Additionally, instantiates a message passing layer per node.

	MPNs are specified by the given adjacency matrix.

	nodes: list of nodes

	adj: adjacency matrix (as a pandas dataframe) denoting
	pairwise distances. If it's boolean, then it's taken to be
	unweighted graph.

	hidden_size: number of nodes in hidden layer (default: 256)

	rnnmdl: the RNN model (RNN_MIN, RNN, RNN_SNG, RNN_FCAST) (default: RNN_MIN)

	mpnmdl: the MP model (MP_THIN, MP_DENSE) (default: MP_THIN)

	single_mpn: whether to use a single MP model for all
	nodes. False means use a unique MP model for each
	node. (default: False)

	verbose: verbose output (default: False)

	'''

	name = 'mprnn'
	def __init__(self,
		nodes, adj,
		hidden_size=256,
		rnnmdl=RNN_MIN,
		mpnmdl=MP_THIN,
		single_mpn=False,
		verbose=False):
		super().__init__(len(nodes), hidden_size, rnnmdl)

		self.nodes = nodes
		self.adj = adj

		# whether to define a single global messaging rule for
		# all nodes, or one for each
		self.single_mpn = single_mpn

		# for every node that has at least neighbor, either a
		# define a new MPN model (if single_mpn is False) else
		# assign the MPN model (if single_mpn is True)
		self.mpn_dict = nn.ModuleDict()
		if single_mpn:
			mpn = mpnmdl(hsize=hidden_size)
			for nname in nodes:
				if (adj.loc[nname] > 0).any():
					self.mpn_dict.add_module(nname, mpn)
		else:
			for nname in nodes:
				if (adj.loc[nname] > 0).any():
					self.mpn_dict.add_module(nname, mpnmdl(hsize=hidden_size))

		if verbose:
			print('MPRNN')
			print(' [*] Defined over: {} nodes'.format(len(nodes)))
			print(' [*] Contains	: {} adjs'.format(len(adj)))

	def eval_input(self, ti, series):
		# at every node, evaluate the first RNN layer on the
		# input and produce the hidden layer output
		hevals = dict()
		for ni, (node_series, rnn, nname) in enumerate(zip(series, self.rnns, self.nodes)):
			value_t = node_series[ti]
			hout = rnn.inp(value_t)
			hevals[nname] = hout
		return hevals

	def eval_message(self, hevals):
		# for each node that has at least one neighbor,
		# compute a message for each of that node's neighbors
		# using the hidden layer outputs both the node and the
		# neighbor
		msgs = dict()
		for nname in self.nodes:
			# only defined over nodes w/ adj
			if not nname in self.mpn_dict:
				msgs[nname] = None
				continue

			# compute message by reading the hidden layer
			# outputs of neighbors. instead of just
			# summing up all the messages (which was done
			# for the traffic problem), do a weighted sum
			# based on the edge weights
			many = []
			denom = 0
			neighbors = self.adj.columns[self.adj.loc[nname] > 0]
			for neighbor in neighbors:
				dist = self.adj.loc[nname,neighbor]
				denom += (1/dist**2)
				many.append(self.mpn_dict[nname].msg(hevals[nname], hevals[neighbor]) / dist**2)
			many = torch.stack(many, -1)
			msg = torch.sum(many, -1) / denom
			msgs[nname] = msg

		return msgs

	def eval_update(self, hevals, msgs):
		for nname in self.nodes:
			# only defined over nodes w/ adj
			if msgs[nname] is None: continue

			# replaces hvalues before update
			hevals[nname] = self.mpn_dict[nname].upd(hevals[nname], msgs[nname])

	def eval_readout(self, hevals, hidden):
		values_t = []
		for ni, (nname, rnn, hdn) in enumerate(zip(self.nodes, self.rnns, hidden)):
			hin = hevals[nname].unsqueeze(0)
			hout, hdn = rnn.rnn(hin, hdn)
			hidden[ni] = hdn # replace previous lstm params
			hout = hout.squeeze(0)
			xout = rnn.out(hout)
			values_t.append(xout)

		return values_t

	def forward(self, series, hidden=None, dump=False):
		# print(len(series), len(series[0]), series[0][0].size())
		assert len(self.rnns) == len(series)

		# lstm params
		if hidden is None:
			bsize = series[0][0].size()[0]
			hshape = (1, bsize, self.hidden_size)
			hidden = [(torch.rand(*hshape).to(self.device),
				   torch.rand(*hshape).to(self.device)) for _ in range(len(series))]

		# defined over input timesteps
		tsteps = len(series[0])
		outs_bynode = [list() for _ in series]
		for ti in range(tsteps):

			# eval up to latent layer for each node
			hevals = self.eval_input(ti, series)

			# message passing
			msgs = self.eval_message(hevals)

			# updating hidden
			self.eval_update(hevals, msgs)

			# read out values from hidden
			values_t = self.eval_readout(hevals, hidden)

			for node_series, value in zip(outs_bynode, values_t):
				node_series.append(value)

		# print(len(outs_bynode), len(outs_bynode[0]), outs_bynode[0][0].size())
		out = list(map(lambda tens: torch.cat(tens, dim=1), outs_bynode))
		out = torch.stack(out, dim=-1)

		if dump:
			return out, hidden
		else:
			return out

	def params(self, lr=0.001):
		criterion = nn.MSELoss(reduction='sum').cuda()
		opt = optim.Adam(self.parameters(), lr=lr)
		sch = optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
		return criterion, opt, sch
