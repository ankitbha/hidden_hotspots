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
from datetime import datetime, timedelta
import time
import json

def find_by_id(idname):
	locfiles = glob('./data/kaiterra_field*.csv')
	for fpath in locfiles:
		if idname in fpath:
			# print(fpath)
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
	if lastvalid != len(series) - 1:
		# did not end with valid vals
		series[lastvalid:] = series[lastvalid]

def load_ourdata(segdef, trange=None):
	raw_segments = []
	t0, tf = None, None
	if trange is not None: t0, tf = trange
	for sii, (locname, locid) in enumerate(segdef['locations']):
		segment = []
		fpath = find_by_id(locid)
		with open(fpath) as fl:
			fl.readline()
			line = fl.readline()
			lii = 0
			while line:
				parts = line.split(',')
				timestamp_round = parts[0]
				tround = datetime.strptime(timestamp_round, '%Y-%m-%d %H:%M:%S')
				if trange is not None and ( tround < t0 or tround > tf):
					line = fl.readline()
					continue
				# print(parts)
				if not parts[1]:
					line = fl.readline()
					continue # sometimes blank str??
				segment.append({
					'timestamp': tround,
					'location': locname,
					'pm25': float(parts[1])
				})
				line = fl.readline()
				# sys.stdout.write('%d: %s     \r' % (lii, fpath))
				# sys.stdout.flush()
				# lii+=1
		raw_segments.append(segment)

		sys.stdout.write('[%d/%d] Loading segment...    \r' % (sii+1, len(segdef['locations'])))
		sys.stdout.flush()

	return raw_segments

def create_dataset(segdef, split=0.8, fillmethod=pad_valid, interval = 15 * 60):
	'''
	TODO: Specify neighbors in neighborlist
		otherwise all are used

	histlen : length of history presented in batch
	sensor_id : name of specified target location
	'''
	raw_segments = [] # collection of segments
	t0 = datetime.strptime(segdef['start'], '%m/%d/%Y')
	tf = datetime.strptime(segdef['end'], '%m/%d/%Y')

	raw_segments = load_ourdata(segdef, (t0, tf))
	print()

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
			assert entry['pm25'] >= 0
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
	start='08/01/2018', end='10/01/2018',
	split=0.8, fillmethod=pad_valid,
	exclude=[]):
	import json

	t0 = datetime.strptime(start, '%m/%d/%Y')
	tf = datetime.strptime(end, '%m/%d/%Y')
	print(' [*] Loading govdata from: %s to %s' % (start, end))
	with open('data/gov.json') as fl:
		govdata = json.load(fl)
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

			if byloc['pm25'] is None or byloc['pm25'] < 0:
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
		# print(np.where(segment < 0))
		try:
			assert len(np.where(segment < 0)[0]) == 0
		except:
			print(np.where(segment < 0)[0])
			print(segment[1238:])
			assert False
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

def create_dataset_joint(segdef, split=0.8, fillmethod=pad_valid, exclude=[], upsamp=None):
	t0 = datetime.strptime(segdef['start'], '%m/%d/%Y')
	tf = datetime.strptime(segdef['end'], '%m/%d/%Y')

	(ourdata, _), (_, _) = create_dataset(segdef, split=1.0, fillmethod=fillmethod)
	(govdata, _), (_, _, govnames) = create_dataset_gov(start=segdef['start'], end=segdef['end'], split=1.0, fillmethod=fillmethod, exclude=exclude)

	if upsamp is not None:
		# govdata is in 15 min intervals; to preserve the resolution of our data, we upsamp the govdata to match
		uped = np.repeat(govdata, upsamp, axis=1)[:, :-2] # last two OOB
		assert ourdata.shape[1] == uped.shape[1]
		govdata = uped

	# print(ourdata.shape, uped.shape)
	datamat = np.concatenate([ourdata, govdata], axis=0)

	splitind = int(datamat.shape[1] * split)
	train_data, test_data = datamat[:, :splitind], datamat[:, splitind:]
	print('Train test split:  %d / %d' % (splitind, datamat.shape[1] - splitind))

	ournames = [ent[0] for ent in segdef['locations']]
	return (train_data, test_data), (None, None, ournames + govnames)


def create_dataset_joint_weather(segdef, split=0.8, fillmethod=pad_valid, exclude=[], upsamp=None):
	(datamat, _), (_, _, names) = create_dataset_joint(
		segdef, 1.0, fillmethod, exclude, upsamp)

	t0 = datetime.strptime(segdef['start'], '%m/%d/%Y')
	tf = datetime.strptime(segdef['end'], '%m/%d/%Y')

	with open('./data/open_weather_newdelhi.json') as fl:
		paid_data = json.load(fl)
	from datetime import date

	d0 = datetime(2018, 3, 1)
	w0 = int((t0 - d0).total_seconds() // (60 * 60)) + 5
	wf = int((tf - d0).total_seconds() // (60 * 60)) + 5
	# print(w0, wf, len(paid_data))
	wdata = [ent['main']['temp'] for ent in paid_data[w0:wf]]
	wdata = np.array(wdata)
	wdata -= 273.15  # kelvins to celsius
	wdata /= 50 # normalize celsius
	wdata = np.repeat(wdata, 4, axis=0)
	# print(datamat.shape, wdata.shape)
	datamat = np.concatenate([datamat[:, :-1], np.array([wdata])], axis=0)

	splitind = int(datamat.shape[1] * split)
	train_data, test_data = datamat[:, :splitind], datamat[:, splitind:]
	print('Train test split:  %d / %d' % (splitind, datamat.shape[1] - splitind))
	assert len(names) == len(train_data) - 1
	return (train_data, test_data), names

def create_dataset_weather(segdef, split=0.8, fillmethod=pad_valid, upsamp=None):
	(datamat, _), _ = create_dataset(
		segdef, split=1.0)

	t0 = datetime.strptime(segdef['start'], '%m/%d/%Y')
	tf = datetime.strptime(segdef['end'], '%m/%d/%Y')
	names = [nm[0] for nm in segdef['locations']]

	with open('./data/open_weather_newdelhi.json') as fl:
		paid_data = json.load(fl)
	from datetime import date

	d0 = datetime(2018, 3, 1)
	w0 = int((t0 - d0).total_seconds() // (60 * 60)) + 5
	wf = int((tf - d0).total_seconds() // (60 * 60)) + 5
	# print(w0, wf, len(paid_data))
	wdata = [ent['main']['temp'] for ent in paid_data[w0:wf]]
	wdata = np.array(wdata)
	wdata -= 273.15  # kelvins to celsius
	wdata /= 50 # normalize celsius
	wdata = np.repeat(wdata, 4, axis=0)
	# print(datamat.shape, wdata.shape)
	datamat = np.concatenate([datamat[:, :-1], np.array([wdata])], axis=0)

	splitind = int(datamat.shape[1] * split)
	train_data, test_data = datamat[:, :splitind], datamat[:, splitind:]
	print('Train test split:  %d / %d' % (splitind, datamat.shape[1] - splitind))
	assert len(names) == len(train_data) - 1
	return (train_data, test_data), names

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

def series_batch(datamat, target, inds, history=5, post=None):
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
			seg = datamat[segii, ind:ind+history]
			if post is not None: seg = post(seg)
			data.append(seg)
		data = np.array(data)
		# predictions made for entire missing target segment
		lbl = datamat[target, ind:ind+history]

		series.append(data)
		labels.append(lbl)
	# labels =
	series = np.array(series)
	labels = np.array(labels)
	return series, labels

def discrete_dataset_gov(
	eval_segment,         # takes in segdef format: segment will be reserved for testing
	split=0.8,
	fillmethod=pad_valid,
	history=16,           # will return segments of this length
	minavail=0.8,         # will ignore segments with too few valid
	minsensitive=0.8,     # TODO: will ignore segments w/ values stuck for too long
	exclude=[]):
	# scans the entire dataset (sans "exclude") for valid trainable segments

	import json
	print(' [*] Loading discrete govdata')
	with open('data/gov.json') as fl:
		govdata = json.load(fl)

	t0 = datetime.strptime(eval_segment['start'], '%m/%d/%Y')
	tf = datetime.strptime(eval_segment['end'], '%m/%d/%Y')

	__govnames = [ent['location'] for ent in govdata[0]['values']]
	print(' [*] Found %d gov locations' % len(__govnames))
	for gname in __govnames:
		print('    * %s %s' % ('' if gname not in exclude else '(excluded)', gname))
	# selected contains index (used to access govdata matrix) of selected gov sensors
	selected = [ind for ind, name in enumerate(__govnames) if name not in exclude]
	__numgovs = len(selected)
	assert len(selected) == len(__govnames) - len(exclude)
	print(' [*] Tracking %d gov locations' % __numgovs)

	reserved = []
	datamat = np.zeros((len(selected), len(govdata)))

	__missingstat = [0] * __numgovs
	for tii, timeentry in enumerate(govdata):
		for gii, govindex in enumerate(selected):
			locdata = timeentry['values'][govindex]
			if locdata['location'] in exclude: continue

			if locdata['pm25'] is None or locdata['pm25'] < 0:
				datamat[gii, tii] = -1
				__missingstat[gii] += 1
			else:
				val = locdata['pm25']
				if val == 0: val = 1
				datamat[gii, tii] = val

		te = datetime.strptime(timeentry['date'], '%d-%m-%Y %H:%M')
		if te >= t0 and te < tf:
			reserved.append(datamat[:, tii])

	reserved = np.transpose(np.array(reserved))
	for sii, segment in enumerate(reserved):
		fillmethod(segment)
		assert len(np.where(segment < 0)[0]) == 0

	train_inds = []
	first_after = None
	for tii, timeentry in enumerate(govdata):
		te = datetime.strptime(timeentry['date'], '%d-%m-%Y %H:%M')
		segment = None
		if te < t0:                 # before reserved
			if tii < history: continue  # seek forward until history available
			segment = datamat[:, tii-history:tii]
		elif te > tf:               # after reserved
			if first_after is None: first_after = tii
			if tii - first_after < history: continue  # seek forward until history available
			segment = datamat[:, tii-history:tii]
		if segment is None: continue


		n_missing = [-np.sum(row[row < 0]) for row in segment]
		perc_missing = [nm / segment.shape[1] for nm in n_missing]
		missing_many = [perc > (1-minavail) for perc in perc_missing]
		if any(missing_many):
			continue
		# TODO: detect "stuck" sensors

		train_inds.append(tii-history)


	print(' [*] Discovered %d/%d usable discrete segments' % (len(train_inds), datamat.shape[1]))

	for gii, govindex in enumerate(selected):
		print('    * Available: %.1f%%  Location: %s' % (
			(datamat.shape[1] - __missingstat[gii]) / datamat.shape[1] * 100.0,
			__govnames[govindex]))

	# pad fill datamat
	for sii, segment in enumerate(datamat):
		fillmethod(segment)
		assert len(np.where(segment < 0)[0]) == 0

	datamat /= 100.0 # normalize under 100
	reserved /= 100.0

	# TODO: option to limit to subsection of reserved for comparable testing w/ other

	# series_batch can be used to get batches from this info
	return ((datamat, train_inds), reserved), [__govnames[govindex] for govindex in selected]

def discrete_dataset_ours(
	eval_segment,         # takes in segdef format: segment will be reserved for testing
	split=0.8,
	fillmethod=pad_valid,
	history=16,           # will return segments of this length
	minavail=0.8,         # will ignore segments with too few valid
	minsensitive=0.8,     # TODO: will ignore segments w/ values stuck for too long
	exclude=[]):
	# scans the entire dataset (sans "exclude") for valid trainable segments

	raw_segments = load_ourdata(eval_segment, trange=None)
	print()

	reserved = []
	t0 = datetime.strptime('03-01-2018', '%m-%d-%Y')
	tf = datetime.strptime('11-01-2018', '%m-%d-%Y')
	tsteps = int((tf - t0).total_seconds() / (15 * 60))
	datamat = -np.ones((len(raw_segments), tsteps+1))
	__missingstat = [0] * len(eval_segment['locations'])
	interval = 15 * 60
	for sii, segment in enumerate(raw_segments):
		for entry in segment:
			seconds = date_in_seconds(entry['timestamp']) - date_in_seconds(t0)
			timeind = int(seconds // interval)
			assert timeind >= 0 and timeind < tsteps + 1
			assert entry['pm25'] >= 0
			datamat[sii, timeind] = entry['pm25']

	assert np.min(datamat) >= -1
	nmissing = -np.sum(datamat[datamat < 0])
	print(' [*] Total missing: %.1f%%' % (
		nmissing / ((tsteps + 1) * len(raw_segments)) * 100.0))
	for sii, series in enumerate(datamat):
		print('   * Available: %.1f%%   Location: %s' % (
			(len(series) + np.sum(series[series < 0])) / len(series) * 100,
			eval_segment['locations'][sii][0]))


	r0 = datetime.strptime(eval_segment['start'], '%m/%d/%Y')
	rf = datetime.strptime(eval_segment['end'], '%m/%d/%Y')
	start = (date_in_seconds(r0) - date_in_seconds(t0)) // (15 * 60)
	end = (date_in_seconds(rf) - date_in_seconds(t0)) // (15 * 60)
	reserved = datamat[:, int(start):int(end)].copy()

	print(' [*] Eval missing: %.1f%%' % (
		-np.sum(reserved[reserved < 0]) / (reserved.shape[1] * reserved.shape[0]) * 100))

	print(' [*] Eval segment availability:')
	for sii, series in enumerate(reserved):
		print('   * Available: %.1f%%   Location: %s' % (
			(len(series) + np.sum(series[series < 0])) / len(series) * 100,
			eval_segment['locations'][sii][0]))

	for sii, series in enumerate(reserved):
		fillmethod(series)
		assert len(np.where(series < 0)[0]) == 0

	train_inds = []
	first_after = None
	nbfr, naft = 0, 0
	for tii in range(datamat.shape[1]):
		segment = None
		if tii < start:                 # before reserved
			if tii < history: continue  # seek forward until history available
			segment = datamat[:, tii-history:tii]
			nbfr += 1
		elif tii >= end:               # after reserved
			if first_after is None: first_after = tii
			if tii - first_after < history: continue  # seek forward until history available
			segment = datamat[:, tii-history:tii]
			naft += 1
			# print(naft)
		if segment is None: continue


		n_missing = [-np.sum(row[row < 0]) for row in segment]
		perc_missing = [nm / segment.shape[1] for nm in n_missing]
		missing_many = [perc > (1-minavail) for perc in perc_missing]
		if any(missing_many):
			continue
		# TODO: detect "stuck" sensors

		train_inds.append(tii-history)

	print(' [*] Discovered %d/%d usable discrete segments' % (len(train_inds), datamat.shape[1]))

	# pad fill datamat
	for sii, segment in enumerate(reserved):
		fillmethod(segment)
		assert len(np.where(segment < 0)[0]) == 0
	reserved /= 100.0

	datamat /= 100.0 # normalize under 100

	# TODO: option to limit to subsection of reserved for comparable testing w/ other

	# series_batch can be used to get batches from this info
	return ((datamat, train_inds), reserved), [nm[0] for nm in eval_segment['locations']]


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

	# create_dataset(
	# 	SEGMENTS[0])
	create_dataset_weather(SEGMENTS[0], exclude=EXCLUDE[0])


