#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


df = pd.read_csv('data/kaiterra_fieldeggid_15min_2018_Sep_28_panel.csv', index_col=[0,1], parse_dates=True)


grouped = df.groupby(level=0)
latlondict = {}

for name,group in grouped:
    latlondict[name] = (group.latitude[0], group.longitude[0])


tsindex = df.index.levels[1]

# The goal now is to run script for several combinations of
# field egg id and numnodes (K).
# iterate through the field egg IDs
fieldeggids_list = df.index.levels[0]

start_ind, end_ind = int(sys.argv[1]), int(sys.argv[2])

for fieldeggid in fieldeggids_list[start_ind:end_ind]:

    # compute and store the sorted distances of every monitor from
    # "fieldeggid" as well as the compass bearings
    dists_bearings_list = []
    for idnum in latlondict.keys():
        if idnum != fieldeggid:
            dist = distance.distance(latlondict[idnum], latlondict[fieldeggid]).meters
            bearing = get_bearing(latlondict[idnum], latlondict[fieldeggid])
            dists_bearings_list.append((idnum, dist, bearing))
    
    # sort by distance
    dists_bearings_list.sort(key=itemgetter(1))
    
    for numnodes in range(1,11):

        print('Field egg ID: {}, K: {}'.format(fieldeggid, numnodes))
        
        # create new empty dataframe with 3 columns for every sensor
        # since we are also storing distance and heading (theta)
        colnames_list = []
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
            vals = df.loc[(slice(None),ts),:].pm25
            
            count = 0
            
            # 3 columns for every sensor since we are also storing distance and heading (theta)
            vals_knn = np.ones(3 * numnodes) * np.nan
            for tup in dists_bearings_list:
                val = vals.loc[(tup[0],ts)]
                if not np.isnan(val):
                    vals_knn[count:count+3] = (val, tup[1], tup[2])
                    count += 3
                if count == 3*numnodes:
                    break
            data.loc[ts,:] = vals_knn
        
        # save the data
        data.to_csv('datasets/knn_pm25disttheta_{}_K{:02d}.csv'.format(fieldeggid, numnodes))

