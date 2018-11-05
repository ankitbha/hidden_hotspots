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

from train_series import plot_preview

if __name__ == '__main__':
	np.random.seed(0)
	torch.manual_seed(0)

	parser = argparse.ArgumentParser(description='LSTM for traffic congestion prediction')
	parser.add_argument('--history', type=int, required=True, help='Length of history')
	parser.add_argument('--hidden', type=int, default=25)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--batch', type=int, default=32)
	parser.add_argument('--epochs', type=int, default=4)
	parser.add_argument('--eval_segment', type=int, required=True, help='Segment to evaluate on')
	parser.add_argument('--target', type=int, required=True, help='Target to train for; check configs...')
	parser.add_argument('--load', type=str, default=None)
	parser.add_argument('--once', dest='once', action='store_true')
	parser.set_defaults(once=False)
	args = parser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# device = torch.device("cpu")
	log = SummaryWriter()

	eval_segment = SEGMENTS[args.eval_segment]
	((trainmat, train_inds), evalmat), (locnames) = discrete_dataset_gov(
		eval_segment=eval_segment,
		history=args.history,
		split=0.8,
		minavail=0.8,
		exclude=EXCLUDE[args.eval_segment])
	numsegments = len(trainmat)
	state_date, end_date = eval_segment['start'], eval_segment['end']
	location_names = locnames
	time_interval = 15

	# if args.load is not None:
	#     print(' Loading:', args.load)
	#     model = torch.load(args.load)
	# else:
	model = Series(
		batchsize=args.batch,
		historylen=args.history,
		numsegments=numsegments,
		hiddensize=args.hidden).to(device).double()

	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	for eii in range(args.epochs):

		model.train()

		shuffle(train_inds)
		numbatches = len(train_inds) // args.batch

		losses = []
		for bii in range(numbatches):
			bii *= args.batch
			batch_inds = train_inds[bii:bii+args.batch]
			batch_seq, batch_lbls = series_batch(
				trainmat,
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
			n_iter = eii*len(train_inds)+bii
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

		# continue
		model.eval()
		series_loss = 0
		series_mape = 0
		continuous = []
		lstm_state = model.init_lstms(device=device, batch=1)
		for ind in range(evalmat.shape[1]):
			batch_seq, batch_lbls = series_batch(
				evalmat,
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
		series_loss /= (evalmat.shape[1] - args.history)
		series_mape /= (evalmat.shape[1] - args.history)
		series_mape *= 100.0
		log.add_scalar('test/loss', series_loss, n_iter)
		print('\n   Testing Loss:  %.3f      Testing MAPE: %.1f%%' % (series_loss * 100.0**2, series_mape))

		if eii % 10 == 0:
			mse_avg = plot_preview(
				eii,
				(state_date, end_date, location_names),
				args.target,
				evalmat,
				continuous,
				series_loss,
				args.history,
				criterion,
				interval=time_interval)

			with open('outputs/seg_%d_targ_%d.txt' % (args.eval_segment, args.target), 'w') as fl:
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
