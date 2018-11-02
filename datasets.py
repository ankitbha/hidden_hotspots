# ********************************************************************
#
# Generate datasets for neural network training, validation and
# testing, using panel csv or dta data. Using time series data over a
# period of several days for multiple locations, we will generate a
# 2-D panel dataset, where first axis is spatial and second axis is
# temporal.
#
# Author: Shiva R. Iyer, Ulzee An
#
# Date: Aug 15, 2018
#
# ********************************************************************

import os, sys
import numpy as np
import pandas as pd
import torch
from configs import *
from glob import glob
from datetime import datetime
import time

def find_by_id(idname):
	locfiles = glob('./data/Kaiterra*.csv')
	for fpath in locfiles:
		if idname in fpath:
			return fpath
	raise Exception('No file found w/ name: %s' % idname)

def date_in_seconds(dobj):
	return time.mktime(dobj.timetuple())

def pad_valid(series):
	lastvalid = -1
	for ind, val in enumerate(series):
		if lastvalid < ind - 1:
			# if there is a gap
			#   fill with next valid value
			series[lastvalid+1:ind] = val
		if val != -1: lastvalid = ind

def create_dataset(segdef, split=0.8, fillmethod=pad_valid):
	'''
	TODO: Specify neighbors in neighborlist
		otherwise all are used

	histlen : length of history presented in batch
	sensor_id : name of specified target location
	'''

	raw_segments = [] # collection of segments
	t0 = datetime.strptime(segdef['start'], '%m/%d/%Y')
	tf = datetime.strptime(segdef['end'], '%m/%d/%Y')
	for sii, (locname, locid) in enumerate(segdef['locations']):
		segment = []
		fpath = find_by_id(locid)
		with open(fpath) as fl:
			fl.readline()
			line = fl.readline()
			while line:
				parts = line.split(',')
				timestamp_round = parts[0]
				tround = datetime.strptime(timestamp_round, '%Y-%m-%d %H:%M:%S')
				if tround < t0 or tround > tf:
					line = fl.readline()
					continue
				segment.append({
					'timestamp': tround,
					'location': locname,
					'pm25': float(parts[1])
				})
				line = fl.readline()
		raw_segments.append(segment)

		sys.stdout.write('[%d/%d] Loading segment...    \r' % (sii+1, len(segdef['locations'])))
		sys.stdout.flush()
	print()

	interval = 5 * 60
	tsteps = int((date_in_seconds(tf) - date_in_seconds(t0)) // (interval))
	print('Time Steps:', tsteps)
	for sii, seg in enumerate(raw_segments):
		print('Available: %.1f%%  Location: %s' % (len(seg)/ tsteps * 100, segdef['locations'][sii][0]))

	# empty values marked as negative
	datamat = -np.ones((len(raw_segments), tsteps + 1))
	for sii, segment in enumerate(raw_segments):
		for entry in segment:
			seconds = date_in_seconds(entry['timestamp']) - date_in_seconds(t0)
			# print(seconds)
			timeind = int(seconds // interval)
			assert timeind >= 0 and timeind < tsteps + 1
			datamat[sii, timeind] = entry['pm25']

	nmissing = -np.sum(datamat[datamat < 0])
	print('Total missing: %.1f%%' % (
		nmissing / ((tsteps + 1) * len(raw_segments)) * 100.0))


	for sii, segment in enumerate(datamat):
		fillmethod(segment)
	datamat /= 100.0 # normalize under 100

	splitind = int(datamat.shape[1] * split)
	train_data, test_data = datamat[:, :splitind], datamat[:, splitind:]
	print('Train test split:  %d / %d' % (splitind, datamat.shape[1] - splitind))

	train_meta = []
	test_meta = []
	for seg_metadata in raw_segments:
		train_meta.append(seg_metadata[:splitind])
		test_meta.append(seg_metadata[splitind:])

	return (train_data, test_data), (train_meta, test_meta)

def create_dataset_gov(
<<<<<<< HEAD
	start='08/01/2018', end='10/01/2018',
	split=0.8, fillmethod=pad_valid,
=======
	start='08/01/2018', end='10/01/2018',
	split=0.8, fillmethod=pad_valid,
>>>>>>> master
	exclude=['Sirifort, New Delhi - CPCB', 'Punjabi Bagh, Delhi - DPCC']):
	import json

	t0 = datetime.strptime(start, '%m/%d/%Y')
	tf = datetime.strptime(end, '%m/%d/%Y')
	print(' [*] Loading govdata from: %s to %s' % (start, end))
	with open('data/gov.json') as fl:
		govdata = json.load(fl)
<<<<<<< HEAD

=======

>>>>>>> master
	filtered = []
	for timeentry in govdata:
		te = datetime.strptime(timeentry['date'], '%d-%m-%Y %H:%M')
		if te < t0 or te > tf: continue
		filtered.append(timeentry)

	assert len(filtered) > 0
	__expected = (tf - t0).total_seconds() / 60 / 15
	print(' [*] Expecting %d entries at 15min intervals' % __expected)
	print(' [*] Found %d entries in this range' % len(filtered))
	assert abs(__expected - len(filtered)) <= 1
<<<<<<< HEAD

=======

>>>>>>> master
	__govnames = [ent['location'] for ent in filtered[0]['values']]
	print(' [*] Found %d gov locations' % len(__govnames))
	for gname in __govnames:
		print('    * %s %s' % ('' if gname not in exclude else '(excluded)', gname))
	selected = [ind for ind, name in enumerate(__govnames) if name not in exclude]
	__numgovs = len(selected)
	assert len(selected) == len(__govnames) - len(exclude)
	print(' [*] Tracking %d gov locations' % __numgovs)
	datamat = np.zeros((__numgovs, len(filtered)))

	__missingstat = [0] * __numgovs
	for tii, timeentry in enumerate(filtered):
		ind = 0
		for gii, govindex in enumerate(selected):
			byloc = timeentry['values'][govindex]

			if byloc['location'] in exclude: continue

			if byloc['pm25'] is None:
				datamat[gii, tii] = -1
				__missingstat[gii] += 1
			else:
				val = byloc['pm25']
				if val == 0: val = 1
				datamat[gii, tii] = val

	for gii, govindex in enumerate(selected):
		print('    * Available: %.1f%%  Location: %s' % (
			(len(filtered) - __missingstat[gii]) / len(filtered) * 100.0,
			__govnames[govindex]))

	for sii, segment in enumerate(datamat):
		fillmethod(segment)
	datamat /= 100.0 # normalize under 100

	splitind = int(datamat.shape[1] * split)
	train_data, test_data = datamat[:, :splitind], datamat[:, splitind:]
	print('Train test split:  %d / %d' % (splitind, datamat.shape[1] - splitind))

	# train_meta = []
	# test_meta = []
	# for seg_metadata in raw_segments:
	# 	train_meta.append(seg_metadata[:splitind])
	# 	test_meta.append(seg_metadata[splitind:])

	return (train_data, test_data), (None, None, [__govnames[gi] for gi in selected])

def create_dataset_joint(segdef, split=0.8, fillmethod=pad_valid, exclude=[]):
	t0 = datetime.strptime(segdef['start'], '%m/%d/%Y')
	tf = datetime.strptime(segdef['end'], '%m/%d/%Y')

	(ourdata, _), (_, _) = create_dataset(segdef, split=1.0, fillmethod=fillmethod)
	(govdata, _), (_, _, govnames) = create_dataset_gov(start=segdef['start'], end=segdef['end'], split=1.0, fillmethod=fillmethod, exclude=exclude)

	# govdata is in 15 min intervals; to preserve the resolution of our data, we upsamp the govdata to match
	upsamp = np.repeat(govdata, 3, axis=1)[:, :-2] # last two OOB
	assert ourdata.shape[1] == upsamp.shape[1]

	# print(ourdata.shape, upsamp.shape)
	datamat = np.concatenate([ourdata, upsamp], axis=0)

	splitind = int(datamat.shape[1] * split)
	train_data, test_data = datamat[:, :splitind], datamat[:, splitind:]
	print('Train test split:  %d / %d' % (splitind, datamat.shape[1] - splitind))

	ournames = [ent[0] for ent in segdef['locations']]
	return (train_data, test_data), (None, None, ournames + govnames)

def nextval_batch(datamat, target, inds, history=5):
	'''
	inds - array of starting indicies to get sequences from
	'''
	series = []
	labels = []
	for bii, ind in enumerate(inds):
		# all observable data up to time-ind
		data = datamat[:, ind:ind+history]
		# target is sinlge datapoint for segment target
		lbl = datamat[target, ind+history]

		series.append(data)
		labels.append(lbl)
	# labels =
	series = np.array(series)
	labels = np.array(labels)
	return series, labels

def series_batch(datamat, target, inds, history=5):
	'''
	Returns data in batches where time segment of `target` is missing.

	inds - array of starting indicies to get sequences from
	'''
	series = []
	labels = []
	for bii, ind in enumerate(inds):
		# all segments != target and their observations between ind:history
		data = []
		for segii, segment in enumerate(datamat):
			if segii == target: continue
			data.append(datamat[segii, ind:ind+history])
		data = np.array(data)
		# predictions made for entire missing target segment
		lbl = datamat[target, ind:ind+history]

		series.append(data)
		labels.append(lbl)
	# labels =
	series = np.array(series)
	labels = np.array(labels)
	return series, labels

if __name__ == '__main__':
	BATCHSIZE = 32
	# (train, test), metadata = create_dataset(SEGMENTS[0])

	# inds = [0] * BATCHSIZE
	# segments, labels = nextval_batch(train, 0, inds, history=3)
	# print('X - segments:', segments.shape)
	# print('Y - labels  :', labels.shape)

	# inds = [0] * BATCHSIZE
	# segments, labels = nextval_batch(train, 0, inds, history=3)
	# print('X - segments:', segments.shape)
	# print('Y - labels  :', labels.shape)

	# (train, test), metadata = create_dataset_gov()

	create_dataset_joint(SEGMENTS[0])
