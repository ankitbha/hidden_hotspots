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
#from configs import *
from glob import glob
from datetime import datetime, timedelta
import time
import json

KAITERRA_CURRENT_SUFFIX = '20180501_20201101'
GOVDATA_CURRENT_SUFFIX = '20180501_20201101'


def get_data(datadir, source, sensor, res, start_date=None, end_date=None):
    
    if source == 'kaiterra':
        fpath = os.path.join(datadir, 'kaiterra', 'kaiterra_fieldeggid_{}_{}_panel.csv'.format(res, KAITERRA_CURRENT_SUFFIX))
        _, data_start_date, data_end_date, _ = os.path.basename(fpath).rsplit('_', 3)
    elif source == 'govdata':
        fpath = os.path.join(datadir, 'govdata', 'govdata_{}_{}.csv'.format(res, GOVDATA_CURRENT_SUFFIX))
        _, data_start_date, data_end_date = os.path.basename(fpath)[:-4].rsplit('_', 2)

    data_start_date, data_end_date = pd.Timestamp(data_start_date), pd.Timestamp(data_end_date)
    if start_date is not None:
        if start_date > data_end_date:
            return None
        elif start_date < data_start_date:
            start_date = None
    if end_date is not None:
        if end_date < data_start_date:
            return None
        elif end_date > data_end_date:
            end_date = None

    data = pd.read_csv(fpath, index_col=[0,1], parse_dates=True)
    data = data.loc[(slice(None), slice(start_date, end_date)), sensor]

    return data


def get_locations(datadir, source=None):

    fpath_kai = os.path.join(datadir, 'kaiterra', 'kaiterra_locations.csv')
    fpath_gov = os.path.join(datadir, 'govdata', 'govdata_locations.csv')
    
    if source == 'combined' or source is None:
        locs_df_kai = pd.read_csv(fpath_kai, usecols=[0,2,3,4], index_col=[0])
        locs_df_kai['Source'] = 'Kaiterra'
        locs_df_gov = pd.read_csv(fpath_gov, index_col=[0])
        locs_df_gov['Source'] = 'Govt'
        locs_df = pd.concat([locs_df_kai, locs_df_gov], axis=0, sort=False)

    locs_df.sort_index(inplace=True)

    return locs_df


def get_adjacency_matrix(datadir, source=None, thres=None, n_max=None):
    """'source': None/'combined', 'kaiterra' or 'govdata' (default: None)

    'thres': threshold distance in meters (default: None)

    'n_max': max number of neighbors (default: None)

    TODO: Should the graph be undirected or directed?  That is, is it
    ok for sensor A to be in the influence ilst of B, but not
    vice-versa? I think it should directed, esp if we take into
    account wind direction. If it directed, then the adj matrix is no
    longer symmetric.

    """

    if source is None:
        source = 'combined'
    savepath = os.path.join(datadir, '{}_distances.csv'.format(source))

    if os.path.exists(savepath):
        adj_matrix_df = pd.read_csv(savepath, index_col=[0])
    else:
        locs_df = get_locations(datadir)
        adj_matrix = np.empty((locs_df.index.size, locs_df.index.size), dtype=np.float64)

        for ii, mid in enumerate(locs_df.index):
            coords = (locs_df.loc[mid].Latitude, locs_df.loc[mid].Longitude)
            distances = [distance.distance(coords, (locs_df.loc[m].Latitude, locs_df.loc[m].Longitude)).meters for m in locs_df.index]
            adj_matrix[ii,:] = distances

        adj_matrix_df = pd.DataFrame(adj_matrix, index=locs_df.index, columns=locs_df.index)
        adj_matrix_df.to_csv(savepath, float_format='%.0f')

    # drop invalid sensor
    if 'D385' in adj_matrix_df.index:
        adj_matrix_df.drop('D385', axis=0, inplace=True)
    if 'D385' in adj_matrix_df.columns:
        adj_matrix_df.drop('D385', axis=1, inplace=True)

    # 'inf' or 'nan' should be used for 'no edges' rather than 0
    # (except for diagonal entries)
    adj_matrix_df[adj_matrix_df == 0] = np.inf
    for ii in range(adj_matrix_df.shape[0]):
        adj_matrix_df.iloc[ii,ii] = 0

    # apply distance threshold for edges
    if thres is not None:
        adj_matrix_df[adj_matrix_df >= thres] = np.inf

    # max number of neighbors we want to allow (we are keeping the
    # graph undirected for now, so we are cutting off edges both in
    # rows and columns i.e. if we remove e_ij, then we remove e_ji as
    # well).
    if n_max is not None:
        if 1 <= n_max <= adj_matrix_df.shape[0] - 1:
            idx = adj_matrix_df.values.argsort(axis=1)
            for ii in range(adj_matrix_df.shape[0]):
                adj_matrix_df.iloc[ii, idx[ii, n_max+1:]] = np.inf
                adj_matrix_df.iloc[idx[ii, n_max+1:], ii] = np.inf

    adj_matrix_df[np.isinf(adj_matrix_df.values)] = 0

    return adj_matrix_df


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

def create_dataset_knodes_sensorid(
	sensor_id, num_nodes, split=0.8):

	'''
	Returns train_data and test_data given the sensor_id, num_nodes (K)
	and the train-test split fraction.
	'''

	key = 'knn_pm25_{}_K{:02d}'.format(sensor_id, num_nodes)

	df = pd.read_csv(
		KPATH + '/' + key + '.csv',
		index_col=[0],
		parse_dates=True)
	with open('kNN_availability.csv') as fin:
			for line in fin:
					fields = line.split(',', 1)
					if fields[0] == key:
							start, end, _ = fields[1].split(',')
							break

	start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)
	datamat = df.loc[start_dt:end_dt].values

	datamat /= 100

	target = pd.read_csv(
		find_by_id(sensor_id),
		index_col=[0],
		parse_dates=True)
	targmat = target.loc[start_dt:end_dt].values[:, 0]
	# datamat = np.concatenate([
	# 	np.expand_dims(targmat, axis=1),
	# 	datamat], axis=1)

	targmat /= 100

	len_train = int(split * datamat.shape[0])
	train_data, test_data = datamat[:len_train,:].T, datamat[len_train:,:].T
	train_labels, test_labels = targmat[:len_train], targmat[len_train:]
	return (train_data, train_labels), (test_data,test_labels)

def create_dataset_knodes_sensorid_v2(
	sensor_id, num_nodes, split=0.8):

	'''
	Returns train_data and test_data given the sensor_id, num_nodes (K)
	and the train-test split fraction.
	'''

	original_key = 'knn_pm25_{}_K{:02d}'.format(sensor_id, num_nodes)
	filekey = 'knn_pm25disttheta_{}_K{:02d}'.format(sensor_id, num_nodes)

	df = pd.read_csv(
		KPATH2 + '/' + filekey + '.csv',
		index_col=[0],
		parse_dates=True)
	with open('kNN_availability.csv') as fin:
			for line in fin:
					fields = line.split(',', 1)
					if fields[0] == original_key:
							start, end, _ = fields[1].split(',')
							break

	start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)
	datamat = df.loc[start_dt:end_dt].values

	target = pd.read_csv(
		find_by_id(sensor_id),
		index_col=[0],
		parse_dates=True)
	targmat = target.loc[start_dt:end_dt].values[:, 0]

	# print(datamat.shape)
	# datamat /= 10.0
	datamat[:, ::3] /= 100.0
	datamat[:, 1::3] /= 1000.0
	datamat[:, 2::3] /= 100.0

	targmat /= 100.0

	len_train = int(split * datamat.shape[0])
	train_data, test_data = datamat[:len_train,:].T, datamat[len_train:,:].T
	train_labels, test_labels = targmat[:len_train], targmat[len_train:]

	return (train_data, train_labels), (test_data, test_labels)

def create_dataset_knodes(max_nodes=None, split=0.8, data_version=create_dataset_knodes_sensorid):
	'''
	Returns train_data and test_data given the num_nodes (K) and the
	train-test split fraction.

	Specify max_nodes to limit # nodes returned as examples

	'''

	with open('.k_train_indices.json') as fl:
		train_refs = json.load(fl)
	with open('.k_test_indices.json') as fl:
		test_refs = json.load(fl)

	train_refs = list(filter(lambda val: int(val.split('_')[1]) <= max_nodes, train_refs))
	test_refs = list(filter(lambda val: int(val.split('_')[1]) <= max_nodes, test_refs))

	dataset = {}
	for lid in LIDS:
		dataset[lid] = []
		for kval in range(10):
			dataset[lid].append(
				data_version(lid, kval+1))

	return dataset, train_refs, test_refs


def read_knn(fpath, version, normalize=True):
    
    df = pd.read_csv(fpath, index_col=[0])
    
    # some normalizations
    if normalize:
        df.iloc[:,0] /= 100.0
        
        if version == 'v1':
            df.iloc[:, 1:] /= 100.0
        else:
            df.iloc[:, 1::3] /= 100.0
            df.iloc[:, 2::3] /= 1000.0
            df.iloc[:, 3::3] /= 100.0
    
    return df


def read_refs(fpath):
    refs = []
    with open(fpath) as fin:
        for line in fin:
            fields = line.split()
            refs.append((fields[0], int(fields[1])))
    return refs


def read_refs_monitorid(fpath, monitorid):
    refs = []
    with open(fpath) as fin:
        for line in fin:
            fields = line.split()
            if fields[0] == monitorid:
                refs.append((fields[0], int(fields[1])))
    return refs


def save_refs(fpath, refs):
    refs.sort()
    with open(fpath, 'w') as fout:
        for ref in refs:
            fout.write('{} {}\n'.format(*ref))
    return


def create_dataset_knn_sensor(source, monitorid, sensor, numnodes, version, stride=1, histlen=32, split=0.8):

    '''Retrieve data for a value of K and a particular sensor, and split
    into train-test. Also return the indices for training in each file
    (read from cache, else cache it and return it)
    
    '''
    
    cache_path1 = 'output/train_test_refs/trainrefs_{}_{}_K{:02d}_h{}_s{}_f{}.txt'.format(source, sensor, numnodes, histlen, stride, split)
    cache_path2 = 'output/train_test_refs/testrefs_{}_{}_K{:02d}_h{}_s{}_f{}.txt'.format(source, sensor, numnodes, histlen, stride, split)
    
    if os.path.exists(cache_path1) and os.path.exists(cache_path2):
        fpath = None
        if version == 'v1':
            fpath = 'datasets/knn_{}_{}/knn_{}_{}_K{:02d}.csv'.format(version, source, sensor, monitorid, numnodes)
        else:
            fpath = 'datasets/knn_{}_{}/knn_{}disttheta_{}_K{:02d}.csv'.format(version, source, sensor, monitorid, numnodes)
        
        df = read_knn(fpath, version)
        df.rename({monitorid:'target'}, axis='columns', inplace=True)
    else:
        df_all, trainrefs, testrefs = create_dataset_knn(source, sensor, numnodes, version, stride, histlen, split)
        df = df_all.loc[monitorid,:]
    
    trainrefs = read_refs_monitorid(cache_path1, monitorid)
    testrefs = read_refs_monitorid(cache_path2, monitorid)

    return df, trainrefs, testrefs


def create_dataset_knn(source, sensor, numnodes, version, stride=1, histlen=32, split=0.8):
    
    '''Retrieve data for a value of K, and split into train-test. Also
    return the indices for training in each file (read from cache,
    else cache it and return it)
    
    '''
    from tqdm import tqdm
    
    pathgen = None
    if version == 'v1':
        pathgen = 'datasets/knn_{}_{}/knn_{}_*_K{:02d}.csv'.format(version, source, sensor, numnodes)
    else:
        pathgen = 'datasets/knn_{}_{}/knn_{}disttheta_*_K{:02d}.csv'.format(version, source, sensor, numnodes)
    
    flist = glob(pathgen)
    
    dfs_list = []
    trainrefs = []
    testrefs = []
    cache = False
    cache_path1 = 'output/train_test_refs/trainrefs_{}_{}_K{:02d}_h{}_s{}_f{}.txt'.format(source, sensor, numnodes, histlen, stride, split)
    cache_path2 = 'output/train_test_refs/testrefs_{}_{}_K{:02d}_h{}_s{}_f{}.txt'.format(source, sensor, numnodes, histlen, stride, split)
    if os.path.exists(cache_path1) and os.path.exists(cache_path2):
        cache = True
        trainrefs = read_refs(cache_path1)
        testrefs = read_refs(cache_path2)
    
    for fpath in tqdm(flist, desc='Reading for {}-NN'.format(numnodes)):
        
        df = read_knn(fpath, version)
        monitorid = df.columns[0]
        df.rename({monitorid:'target'}, axis='columns', inplace=True)
        
        # create new dataframe with MultiIndex (monitorid, index) for
        # quick indexing while creating batches during training
        df_reind = pd.DataFrame(data=df.values,
                                index=pd.MultiIndex.from_product([[monitorid], df.index], names=('monitorid', 'ref')),
                                columns=df.columns)
        dfs_list.append(df_reind)
        
        # get trainrefs and testrefs if not already cached -- valid
        # indices from which contiguous blocks of 'histlen' length of
        # data are available
        if not cache:
            # split into train and test
            test_begin = int(df_reind.shape[0] * split)
            
            traininds = []
            for tii in range(0, test_begin-histlen, stride):
                if not df_reind.iloc[tii:tii+histlen,:].isnull().any(axis=None):
                    traininds.append(tii)
            
            testinds = []
            for tii in range(test_begin, df_reind.shape[0]-histlen, stride):
                if not df_reind.iloc[tii:tii+histlen,:].isnull().any(axis=None):
                    testinds.append(tii)
            
            traininds = np.asarray(traininds)
            testinds = np.asarray(testinds)
            
            trainrefs.extend([(monitorid, ind) for ind in traininds])
            testrefs.extend([(monitorid, ind) for ind in testinds])
    
    if not cache and len(flist) > 0:
        save_refs(cache_path1, trainrefs)
        save_refs(cache_path2, testrefs)
    
    df_all = pd.concat(dfs_list, axis=0, sort=False, copy=False)
    return df_all, trainrefs, testrefs


def create_dataset_knn_testdays(source, sensor, numnodes, version, testdaysfilepath, stride=1, histlen=32):
    
    '''Retrieve data for a value of K, and split into train-test based on
    a fixed custom set of days set apart for testing. Also return the
    indices for training in each file (read from cache, else cache it
    and return it)
    
    '''
    from tqdm import tqdm
    
    pathgen = None
    if version == 'v1':
        pathgen = 'datasets/knn_{}_{}/knn_{}_*_K{:02d}.csv'.format(version, source, sensor, numnodes)
    else:
        pathgen = 'datasets/knn_{}_{}/knn_{}disttheta_*_K{:02d}.csv'.format(version, source, sensor, numnodes)
    
    flist = glob(pathgen)
    
    dfs_list = []
    trainrefs = []
    testrefs = []
    cache = False
    
    testsetname = os.path.basename(testdaysfilepath).split('_')[2]
    cache_path1 = 'output/train_test_refs/trainrefs_{}_{}_K{:02d}_h{}_s{}_testdays_{}.txt'.format(source, sensor, numnodes, histlen, stride, testsetname)
    cache_path2 = 'output/train_test_refs/testrefs_{}_{}_K{:02d}_h{}_s{}_testdays_{}.txt'.format(source, sensor, numnodes, histlen, stride, testsetname)
    
    if os.path.exists(cache_path1) and os.path.exists(cache_path2):
        cache = True
        trainrefs = read_refs(cache_path1)
        testrefs = read_refs(cache_path2)
    else:
        testdays = set(np.loadtxt(testdaysfilepath, delimiter=',', skiprows=1, usecols=[0], dtype=str))
    
    for fpath in tqdm(flist, desc='Reading for {}-NN'.format(numnodes)):
        
        df = read_knn(fpath, version)
        monitorid = df.columns[0]
        
        # get trainrefs and testrefs if not already cached -- valid
        # indices from which contiguous blocks of 'histlen' length of
        # data are available
        if not cache:
            traininds = []
            testinds = []
            for tii in range(0, df.shape[0]-histlen, stride):
                section = df.iloc[tii:tii+histlen,:]
                if not section.isnull().any(axis=None):
                    day_start = section.index[0][:10]
                    day_end = section.index[-1][:10]
                    if (day_start == day_end) and (day_start in testdays):
                        testinds.append(tii)
                    elif (day_start not in testdays) and (day_end not in testdays):
                        traininds.append(tii)
            
            traininds = np.asarray(traininds)
            testinds = np.asarray(testinds)
            
            trainrefs.extend([(monitorid, ind) for ind in traininds])
            testrefs.extend([(monitorid, ind) for ind in testinds])
        
        # create new dataframe with MultiIndex (monitorid, index) for
        # quick indexing while creating batches during training
        df.rename({monitorid:'target'}, axis='columns', inplace=True)
        df_reind = pd.DataFrame(data=df.values,
                                index=pd.MultiIndex.from_product([[monitorid], df.index], names=('monitorid', 'ref')),
                                columns=df.columns)
        dfs_list.append(df_reind)
    
    if not cache and len(flist) > 0:
        save_refs(cache_path1, trainrefs)
        save_refs(cache_path2, testrefs)
    
    df_all = pd.concat(dfs_list, axis=0, sort=False, copy=False)
    return df_all, trainrefs, testrefs


def knodes_batch(dataset, batch_refs, histlen=32, mode='train', pad=30):
	from time import time

	labels = np.zeros((len(batch_refs), histlen))
	batch = np.zeros((len(batch_refs), pad, histlen))
	for rii, ref in enumerate(batch_refs):
		lid, kval, tind = ref.split('_')
		kval, tind = int(kval), int(tind)

		train_set, test_set = dataset[lid][kval-1]

		inputs, targets = train_set if mode == 'train' else test_set
		# print(inputs.shape, targets.shape)
		seg = inputs[:, tind:tind+histlen]
		targ = targets[tind:tind+histlen]

		labels[rii, :] = targ
		batch[rii, :seg.shape[0], :] = seg

	return batch, labels


def batch_knn(dataset_df, batchrefs, histlen):
    
    labels = np.zeros((len(batchrefs), histlen))
    batch = np.zeros((len(batchrefs), dataset_df.shape[1]-1, histlen))
    
    for rii, ref in enumerate(batchrefs):
        monitorid, start = ref
        dataset_df_batch = dataset_df.loc[(monitorid, slice(None)), :].iloc[start:start+histlen,:]
        batch[rii, :, :] = dataset_df_batch.iloc[:,1:].values.T
        labels[rii, :] = dataset_df_batch.iloc[:,0].values
    
    return batch, labels


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


def cache_knn_availability(version, date):
    
    '''Cache availability of each kNN dataset (for a value of sensor and
    K) in a .csv file for easy future access. Availability basically
    means the largest contiguous segment for each kNN dataset.
    
    There is usually no need to call this function explicitly.
    
    '''
    from tqdm import tqdm
    
    # e.g. version 'v1', date '2018_Sep_28'
    inpath = os.path.join('datasets', 'knn_{}_{}'.format(version, date), '*.csv')
    fileslist = glob(inpath)
    fileslist.sort()
    
    print('Caching availability for kNN datasets, version {} and date {} ... '.format(version, date),
          end='', flush=True)

    outdir = 'output'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    fout = open(os.path.join(outdir, 'knn_{}_{}_availability.csv'.format(version, date)), 'w')
    fout.write('kNN dataset file,Start time,End time,Number of valid points\n')
    for fpath in tqdm(fileslist):
        df = pd.read_csv(fpath, index_col=[0], parse_dates=True)
        index_start_final = index_end_final = None
        count_final = 0
        index_start = index_end = None
        count = 0
        state = False
        for tup in df.itertuples():
            if ~np.isnan(tup[1:]).any():
                count += 1
                if state == False:
                    # entering a contiguous region
                    state = True
                    index_start = tup[0]
                    index_end = tup[0]
                else:
                    # already in a contiguous region
                    index_end = tup[0]
            else:
                if state == True:
                    # end of a contiguous region
                    state = False
                    if count > count_final:
                        count_final = count
                        index_start_final = index_start
                        index_end_final = index_end
                        count = 0
                        index_start = index_end = None
        fout.write('{},{},{},{}\n'.format(os.path.split(fpath)[1][:-4], index_start_final.isoformat(), index_end_final.isoformat(), count_final))
    fout.close()
    
    print('done.')

if __name__ == '__main__':
	BATCHSIZE = 32
	(train, test), metadata = create_dataset(SEGMENTS[0])

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
	# create_dataset_weather(SEGMENTS[0], exclude=EXCLUDE[0])


