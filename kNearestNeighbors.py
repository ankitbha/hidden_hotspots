# create a table for each sensor S and a value of K, where the columns
# in the table are the readings from the K nearest neighbors to
# S. Each such table adds to the training dataset for a value of K.

import os
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


def get_latlondict():
    # get lat and lon from field egg dta file
    fieldeggsdetails = pd.read_stata('/home/shivar/Dropbox/Delhi Pollution/06_Data_Collection/Pilot_2018_Kaiterra_FieldEggs/Dta files/field_egg_details.dta')
    fieldeggsdetails.udid = [s.replace('-','') for s in fieldeggsdetails.udid.values]
    fieldeggsdetails.set_index('udid', inplace=True)

    # map sensor IDs to lat,lon 
    latlondict = {}
    for row in fieldeggsdetails.itertuples():
        latlondict[row.Index.upper()[-4:]] = (row.latitude, row.longitude)

    return latlondict


def make_knn_dataset_v1(df, fieldeggids_list, numnodes_list, sensor, savedir):

    latlondict = get_latlondict()
    tsindex = df.index.levels[1]
    
    # The goal now is to run script for several combinations of
    # field egg id and numnodes (K).
    # iterate through the field egg IDs
    for fieldeggid in fieldeggids_list:

        # compute and store the sorted distances of every monitor from
        # "fieldeggid"
        dists_list = [(idnum, distance.distance(latlondict[fieldeggid], latlondict[idnum]).meters) for idnum in fieldeggids_list if idnum != fieldeggid]
        dists_list.sort(key=itemgetter(1))

        for numnodes in numnodes_list:

            print('Field egg ID: {}, K: {}'.format(fieldeggid, numnodes))

            # create new empty dataframe with 3 columns for every sensor
            # since we are also storing distance and heading (theta)
            colnames_list = [fieldeggid] + ['Monitor_{:02d}'.format(i) for i in range(1,numnodes+1)]
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
                vals_knn = np.ones(numnodes + 1) * np.nan
                vals_knn[0] = df.loc[(fieldeggid, ts)][sensor]
                for tup in dists_list:
                    val = vals.loc[(tup[0],ts)]
                    if not np.isnan(val):
                        vals_knn[count] = (val / tup[1]) * (1e7 / tup[1])
                        count += 1
                    if count == numnodes + 1:
                        break
                data.loc[ts,:] = vals_knn

            # save the data
            data.to_csv(savedir + 'knn_{}_{}_K{:02d}.csv'.format(sensor, fieldeggid, numnodes))


def make_knn_dataset_v2(df, fieldeggids_list, numnodes_list, sensor, savedir):

    latlondict = get_latlondict()
    tsindex = df.index.levels[1]
    
    # The goal now is to run script for several combinations of
    # field egg id and numnodes (K).
    # iterate through the field egg IDs
    for fieldeggid in fieldeggids_list:

        # compute and store the sorted distances of every monitor from
        # "fieldeggid" as well as the compass bearings
        dists_bearings_list = []
        for idnum in fieldeggids_list:
            if idnum != fieldeggid:
                dist = distance.distance(latlondict[idnum], latlondict[fieldeggid]).meters

                # note: computing bearing of 'idnum' wrt 'fieldeggid'
                # (opposite of what was done previously for Nov 2018
                # Ubicomp submission)
                bearing = get_bearing(latlondict[fieldeggid], latlondict[idnum])

                dists_bearings_list.append((idnum, dist, bearing))

        # sort by distance
        dists_bearings_list.sort(key=itemgetter(1))

        for numnodes in numnodes_list:

            print('Field egg ID: {}, K: {}'.format(fieldeggid, numnodes))

            # create new empty dataframe with 3 columns for every sensor
            # since we are also storing distance and heading (theta)
            colnames_list = [fieldeggid]
            for i in range(1,numnodes+1):
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
                vals_knn = np.ones(3*numnodes + 1) * np.nan
                vals_knn[0] = df.loc[(fieldeggid, ts)][sensor]
                for tup in dists_bearings_list:
                    val = vals.loc[(tup[0],ts)]
                    if not np.isnan(val):
                        vals_knn[count:count+3] = (val, tup[1], tup[2])
                        count += 3
                    if count == 3*numnodes + 1:
                        break
                data.loc[ts,:] = vals_knn

            # save the data
            data.to_csv(savedir + 'knn_{}disttheta_{}_K{:02d}.csv'.format(sensor, fieldeggid, numnodes))


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('datafilepath', help='Input data dump (kaiterra_fieldeggid_*_panel.csv)')
    parser.add_argument('version', choices=('v1', 'v2'), help='Version v1 or v2')
    parser.add_argument('--min-K', type=int, default=1)
    parser.add_argument('--max-K', type=int, default=10)
    parser.add_argument('--start-ind', type=int, default=0)
    parser.add_argument('--end-ind', type=int)
    parser.add_argument('--sensor', choices=('pm25', 'pm10'), default='pm25')
    args = parser.parse_args()
    
    # suffix = '15min_2018_Sep_28'
    # suffix = '15min_2019_Feb_05'
    datesuffix = '_'.join(os.path.basename(args.datafilepath).split('_')[3:6])
    df = pd.read_csv(args.datafilepath, index_col=[0,1], parse_dates=True)
    df.tz_localize('UTC', level=1, copy=False)
    df.tz_convert('Asia/Kolkata', level=1, copy=False)
    
    fieldeggids_list = df.index.levels[0]

    savedir = 'datasets/knn_{}_{}/'.format(args.version, datesuffix)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    if args.version == 'v1':
        make_knn_dataset_v1(df,
                            fieldeggids_list[args.start_ind:args.end_ind],
                            range(args.min_K, args.max_K+1),
                            args.sensor,
                            savedir)
    else:
        make_knn_dataset_v2(df,
                            fieldeggids_list[args.start_ind:args.end_ind],
                            range(args.min_K, args.max_K+1),
                            args.sensor,
                            savedir)
