# create a table for each sensor S and a value of K, where the columns
# in the table are the readings from the K nearest neighbors to
# S. Each such table adds to the training dataset for a value of K.

import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from geopy import distance
from operator import itemgetter

def get_bearing(ll1, ll2):
    lat1, lon1 = map(math.radians, ll1)
    lat2, lon2 = map(math.radians, ll2)
    dLon = lon2 - lon1;
    y = math.sin(dLon) * math.cos(lat2);
    x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dLon);
    brng = math.degrees(math.atan2(y, x));
    if brng < 0: 
        brng += 360
    return brng


def get_latlondict_kaiterra():
    # get lat and lon from field egg dta file
    fieldeggsdetails = pd.read_stata('/home/shivar/Dropbox/Delhi Pollution/06_Data_Collection/Pilot_2018_Kaiterra_FieldEggs/Dta files/field_egg_details.dta')
    fieldeggsdetails.udid = [s.replace('-','') for s in fieldeggsdetails.udid.values]
    fieldeggsdetails.set_index('udid', inplace=True)
    
    # map sensor IDs to lat,lon 
    latlondict = {}
    for row in fieldeggsdetails.itertuples():
        latlondict[row.Index.upper()[-4:]] = (row.latitude, row.longitude)
    
    return latlondict


def get_latlondict_gov():
    locs_df = pd.read_csv('data/govdata/govdata_locations.csv', index_col=[0])
    
    latlondict = {}
    for row in locs_df.itertuples():
        latlondict[row.Index] = (row.Latitude, row.Longitude)
    
    return latlondict


def make_knn_dataset_v1(NAME, df, monitorids_list, numnodes_min, numnodes_max, sensor, savedir):
    
    latlondict = get_latlondict_kaiterra() if NAME == 'kaiterra' else get_latlondict_gov()
    tsindex = df.index.levels[1]
    
    # The goal now is to run script for several combinations of
    # monitor ids and numnodes (K).
    # iterate through the monitor IDs
    for monitorid in monitorids_list:
        
        # compute and store the sorted distances of every monitor from
        # "monitorid"
        dists_list = [(idnum, distance.distance(latlondict[monitorid], latlondict[idnum]).meters) for idnum in monitorids_list if idnum != monitorid]
        dists_list.sort(key=itemgetter(1))
        
        # create new empty dataframe with 3 columns for every sensor
        # since we are also storing distance and heading (theta)
        colnames_list = [monitorid] + ['Monitor_{:02d}'.format(i) for i in range(1,numnodes_max+1)]
        data = pd.DataFrame(data=None, 
                            index=tsindex, 
                            columns=colnames_list,
                            dtype=np.double)
        
        # iterate through index and update data 
        for ts in tqdm(tsindex):
            
            # for each timestamp, choose "numnode" nearest neighbors
            # that have valid values (not NaNs)
            vals = df.loc[(slice(None),ts), sensor]
            
            count = 1
            vals_knn = np.ones(numnodes_max + 1) * np.nan
            vals_knn[0] = df.loc[(monitorid, ts)][sensor]
            for tup in dists_list:
                val = vals.loc[(tup[0],ts)]
                if not np.isnan(val):
                    vals_knn[count] = (val / tup[1]) * (1e7 / tup[1])
                    count += 1
                if count == numnodes_max + 1:
                    break
            data.loc[ts,:] = vals_knn
        
        for numnodes in range(numnodes_max, numnodes_min - 1, -1):
            
            print('Monitor ID: {}, K: {}'.format(monitorid, numnodes))
            
            # save the data
            data.to_csv(savedir + 'knn_{}_{}_K{:02d}.csv'.format(sensor, monitorid, numnodes))
            data.drop(['Monitor_{:02d}'.format(numnodes)], axis=1, inplace=True)


def make_knn_dataset_v2(NAME, df, monitorids_list, numnodes_min, numnodes_max, sensor, savedir):
    
    latlondict = get_latlondict_kaiterra() if NAME == 'kaiterra' else get_latlondict_gov()
    tsindex = df.index.levels[1]
    
    # The goal now is to run script for several combinations of
    # monitor id and numnodes (K).
    # iterate through the monitor IDs
    for monitorid in monitorids_list:
        
        # compute and store the sorted distances of every monitor from
        # "monitorid" as well as the compass bearings
        dists_bearings_list = []
        for idnum in monitorids_list:
            if idnum != monitorid:
                dist = distance.distance(latlondict[idnum], latlondict[monitorid]).meters
                
                # note: computing bearing of 'idnum' wrt 'monitorid'
                # (opposite of what was done previously for Nov 2018
                # Ubicomp submission)
                bearing = get_bearing(latlondict[monitorid], latlondict[idnum])
                
                dists_bearings_list.append((idnum, dist, bearing))
        
        # sort by distance
        dists_bearings_list.sort(key=itemgetter(1))
        
        # create new empty dataframe with 3 columns for every sensor
        # since we are also storing distance and heading (theta)
        colnames_list = [monitorid]
        for i in range(1, numnodes_max + 1):
            colkey = 'Monitor_{:02d}'.format(i)
            colnames_list.extend([colkey, colkey + '_dist', colkey + '_bearing'])
        data = pd.DataFrame(data=None, 
                            index=tsindex, 
                            columns=colnames_list,
                            dtype=np.double)
        
        # iterate through index and update data 
        for ts in tqdm(tsindex):
            # for each timestamp, choose "numnode" nearest neighbors
            # that have valid values (not NaNs)
            vals = df.loc[(slice(None),ts), sensor]
            
            count = 1
            
            # 3 columns for every sensor since we are also storing distance and heading (theta)
            vals_knn = np.ones(3*numnodes_max + 1) * np.nan
            vals_knn[0] = df.loc[(monitorid, ts)][sensor]
            for tup in dists_bearings_list:
                val = vals.loc[(tup[0],ts)]
                if not np.isnan(val):
                    vals_knn[count:count+3] = (val, tup[1], tup[2])
                    count += 3
                if count == 3*numnodes_max + 1:
                    break
            data.loc[ts,:] = vals_knn
        
        for numnodes in range(numnodes_max, numnodes_min - 1, -1):
            print('Monitor ID: {}, K: {}'.format(monitorid, numnodes))
            
            # save the data
            data.to_csv(savedir + 'knn_{}disttheta_{}_K{:02d}.csv'.format(sensor, monitorid, numnodes))
            colkey = 'Monitor_{:02d}'.format(numnodes)
            data.drop([colkey, colkey + '_dist', colkey + '_bearing'], axis=1, inplace=True)



if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('datafilepath', help='Input data dump (kaiterra_fieldeggid_*_panel.csv or govdata_*_panel.csv)')
    parser.add_argument('version', choices=('v1', 'v2'), help='Version v1 or v2')
    parser.add_argument('--min-K', type=int, default=1)
    parser.add_argument('--max-K', type=int, default=10)
    parser.add_argument('--start-ind', type=int, default=0)
    parser.add_argument('--end-ind', type=int)
    parser.add_argument('--sensor', choices=('pm25', 'pm10'), default='pm25')
    args = parser.parse_args()
    
    # suffix = '15min_2018_Sep_28'
    # suffix = '15min_2019_Feb_05'
    basename = os.path.basename(args.datafilepath)
    if basename.startswith('kaiterra'):
        # datesuffix = '_'.join(basename.split('_')[3:6])
        savedir = 'datasets/knn_{}_kaiterra/'.format(args.version)
        NAME = 'kaiterra'
    else:
        savedir = 'datasets/knn_{}_govdata/'.format(args.version)
        NAME = 'gov'
    
    if os.path.exists(savedir):
        print('Error! Dataset already exists for "{}" in {}. Please remove/move/rename it and run again.'.format(NAME, savedir), file=sys.stderr)
        raise SystemExit
    else:
        os.makedirs(savedir)
    
    #df = pd.read_csv(args.datafilepath, usecols=[0,1,4,5,6], index_col=[0,1], parse_dates=True)
    df = pd.read_csv(args.datafilepath, index_col=[0,1], parse_dates=True)
    # df.tz_localize('UTC', level=1, copy=False)
    # df.tz_convert('Asia/Kolkata', level=1, copy=False)
    
    monitorids_list = df.index.levels[0]
    
    if args.version == 'v1':
        make_knn_dataset_v1(NAME,
                            df,
                            monitorids_list[args.start_ind:args.end_ind],
                            args.min_K,
                            args.max_K,
                            args.sensor,
                            savedir)
    else:
        make_knn_dataset_v2(NAME,
                            df,
                            monitorids_list[args.start_ind:args.end_ind],
                            args.min_K,
                            args.max_K,
                            args.sensor,
                            savedir)
