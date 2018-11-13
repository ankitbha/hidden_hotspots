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

prevloss = 100000000
if __name__ == '__main__':
	# set random seed to 0
	np.random.seed(0)
	torch.manual_seed(0)

	parser = argparse.ArgumentParser(description='LSTM for traffic congestion prediction')
	parser.add_argument('--history', type=int, required=True, help='Length of history')
	parser.add_argument('--mode', type=int, required=True, help='Train data type')
	parser.add_argument('--target', type=int, required=True, help='Target to train for; check configs...')
	parser.add_argument('--fold', type=int)

	parser.add_argument('--hidden', type=int, default=25)
	parser.add_argument('--lr', type=float, default=1e-3)
	parser.add_argument('--batch', type=int, default=32)
	parser.add_argument('--epochs', type=int, default=4)
	parser.add_argument('--stride', type=int, default=1, help='Stride factor')
	parser.add_argument('--load', type=str, default=None)
	parser.add_argument('--once', dest='once', action='store_true')
	parser.set_defaults(once=False)
	args = parser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# device = torch.device("cpu")
	log = SummaryWriter()

	split = 0.8 if args.fold is None else 1.0

	DEFAULTSEG = 3
	if args.mode == 0:
		# gov data only
		seg = SEGMENTS[DEFAULTSEG]
		assert args.target < len(seg['locations'])
		(train_data, test_data), _ = create_dataset(seg, split=split)
		location_names = [nm[0] for nm in seg['locations']]
		numsegments = train_data.shape[0]
		start_date, end_date = seg['start'], seg['end']
		time_interval = 15
		tag = 'ours'
	elif args.mode == 1:
		# gov + our data
		bad_govs = EXCLUDE[DEFAULTSEG]
		seg = SEGMENTS[DEFAULTSEG]
		assert args.target < len(seg['locations'])
		(train_data, test_data), (_, _, location_names) = create_dataset_joint(
			seg, exclude=bad_govs, split=split)
		numsegments = train_data.shape[0]
		start_date, end_date = seg['start'], seg['end']
		time_interval = 15
		tag = 'joint'
	elif args.mode == 2:
		# our data + weather
		seg = SEGMENTS[DEFAULTSEG]
		assert args.target < len(seg['locations'])
		(train_data, test_data), location_names = create_dataset_weather(
			seg, split=split)
		numsegments = train_data.shape[0]
		start_date, end_date = seg['start'], seg['end']
		time_interval = 15
		tag = 'ours_w'
	elif args.mode == 3:
		# gov + our data + weather
		bad_govs = EXCLUDE[DEFAULTSEG]
		seg = SEGMENTS[DEFAULTSEG]
		assert args.target < len(seg['locations'])
		(train_data, test_data), location_names = create_dataset_joint_weather(
			seg, exclude=bad_govs, split=split)
		numsegments = train_data.shape[0]
		start_date, end_date = seg['start'], seg['end']
		time_interval = 15
		tag = 'joint_w'
	else:
		raise Exception('Unknown training mode!')

	nFolds = 9
	if args.fold is not None:
		unit = train_data.shape[1] / nFolds
		test_data = train_data[:, int(unit*args.fold):int(unit*(args.fold+1))]
		# print(test_data.shape)

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
		skipped = 0
		for ind in range(0, train_data.shape[1] - seqlen, args.stride):
			if args.fold is not None:
				if ind + seqlen > unit * args.fold and ind < unit * (args.fold+1):
					skipped += 1
					continue
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
			loss = loss.detach().cpu().numpy()
			assert loss >= 0
			series_loss += loss
			batch_lbls = batch_lbls.detach().squeeze().cpu().numpy()
			preds = preds.detach().squeeze().cpu().numpy()
			series_mape += np.abs((preds - batch_lbls) / batch_lbls)
		series_loss /= (test_data.shape[1])
		series_mape /= (test_data.shape[1])
		series_mape *= 100.0
		log.add_scalar('test/loss', series_loss, n_iter)
		print('\n   Testing Loss:  %.3f      Testing MAPE: %.1f%%' % (series_loss * 100.0**2, series_mape))

		if series_loss < prevloss:
			prevloss = series_loss
			imname = 'images/plots/%s-targ_%d-fold_%d_best.png' % (tag, args.target, args.fold)
			has_weather = args.mode == 2 or args.mode == 3
			mse_avg = plot_preview(
				imname,
				tag,
				eii,
				(start_date, end_date, location_names),
				args.target,
				test_data[:-1] if has_weather else test_data,
				continuous,
				series_loss,
				args.history,
				criterion,
				interval=time_interval,
				weather=test_data[-1] if has_weather else None)

		logfile = 'outputs/%s-targ_%d-fold_%d.txt' % (tag, args.target, args.fold)
		if eii == 1:
			with open(logfile, 'w') as fl:
				fl.write('')
		# if eii % 5 == 0:
		with open(logfile, 'a') as fl:
			fl.write(json.dumps({
				'EPOCH': eii,
				'MAPE': series_mape,
				'MSE_TEST': (series_loss * 100.0**2),
				'MSE_AVG': (mse_avg * 100.0**2),
				'MSE_TRAIN': (loss.item() * 100.0**2),
			}) + '\n')


		if args.once:
			break

		if eii % 10 == 0:
			torch.save(model, 'checkpoints/%s_targ-%d.pth' % (model.name, args.target))


	print()

	log.close()
