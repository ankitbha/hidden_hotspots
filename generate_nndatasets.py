# ********************************************************************
#
# Generate datasets for neural network training, validation and
# testing, using panel csv or dta data. Using time series data over a
# period of several days for multiple locations, we will generate a
# 2-D panel dataset, where first axis is spatial and second axis is
# temporal.
#
# Author: Shiva R. Iyer
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
	raise Exception('No file found w/ name: %s' % locname)

def date_in_seconds(dobj):
	return time.mktime(dobj.timetuple())

def create_dataset(segdef):
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
		lastvalid = -1
		for ind, val in enumerate(segment):
			if lastvalid < ind - 1:
				# if there is a gap
				#   fill with current value
				datamat[sii, lastvalid:ind] = val
			if val != -1: lastvalid = ind

	datamat /= 100.0 # normalize under 100

	return datamat, raw_segments

def target_batch(datamat, target, inds, history=5):
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


if __name__ == '__main__':
	BATCHSIZE = 32
	dataset, metadata = create_dataset(SEGMENTS[0])

	inds = [0] * BATCHSIZE
	segments, labels = target_batch(dataset, 0, inds, history=3)
	print('X - segments:', segments.shape)
	print('Y - labels  :', labels.shape)
