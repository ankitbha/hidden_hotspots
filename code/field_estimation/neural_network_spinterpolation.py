import os
import sys
import pytz
import argparse
# import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from geopy import distance
import datetime
from copy import deepcopy
import pickle as pkl
from PIL import Image
import skimage.measure
import math
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
os.environ['PYTHONWARNINGS']='ignore'
import random
random.seed(42)
import scipy
import torch
from sklearn.model_selection import train_test_split
from scipy.interpolate import CubicSpline
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor

source = 'combined'
sensor = 'pm25'
res_time = '1H'
filepath_root = '/scratch/ab9738/pollution_with_sensors/'

filepath_data_kai = filepath_root+'data/kaiterra/kaiterra_fieldeggid_{}_current_panel.csv'.format(res_time)
filepath_data_gov = filepath_root+'data/govdata/govdata_{}_current.csv'.format(res_time)
filepath_locs_kai = filepath_root+'data/kaiterra/kaiterra_locations.csv'
filepath_locs_gov = filepath_root+'data/govdata/govdata_locations.csv'

locs_kai = pd.read_csv(filepath_locs_kai, index_col=[0])
locs_kai['Type'] = 'Kaiterra'
locs_gov = pd.read_csv(filepath_locs_gov, index_col=[0])
locs_gov['Type'] = 'Govt'
locs = pd.merge(locs_kai, locs_gov, how='outer',\
                on=['Monitor ID', 'Latitude', 'Longitude', 'Location', 'Type'], copy=False)
data_kai = pd.read_csv(filepath_data_kai, index_col=[0,1], parse_dates=True)[sensor]
data_gov = pd.read_csv(filepath_data_gov, index_col=[0,1], parse_dates=True)[sensor]
data = pd.concat([data_kai, data_gov], axis=0, copy=False)
data.replace(0,np.nan,inplace=True)

start_dt = data.index.levels[1][0]
end_dt = data.index.levels[1][-1]

if start_dt.tzname != 'IST':
        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize('UTC')
        start_dt = start_dt.tz_convert(pytz.FixedOffset(330))
    
if end_dt.tzname != 'IST':
    if end_dt.tzinfo is None: 
        end_dt = end_dt.tz_localize('UTC')
    end_dt = end_dt.tz_convert(pytz.FixedOffset(330))

# now, filter through the start and end dates
data.sort_index(inplace=True)
data = data.loc[(slice(None), slice(start_dt, end_dt))]

if(source=='govdata'):
    df = data_gov.unstack(level=0)
elif(source=='kaiterra'):
    df = data_kai.unstack(level=0)
else:
    df = data.unstack(level=0)
distances = pd.read_csv('/scratch/ab9738/pollution_with_sensors/data/combined_distances.csv', index_col=[0])
distances = distances.loc[df.columns, df.columns]
distances[distances == 0] = np.nan

df = np.log(df)

sens = np.log(data).to_frame().reset_index()

sens['hour_of_day'] = sens['timestamp_round'].apply(lambda x: x.hour)

spline = sens.groupby(['field_egg_id', 'hour_of_day']).mean()['pm25'].reset_index()
spline_avg = sens.groupby(['hour_of_day']).mean()['pm25'].reset_index()

fields = []
times = []
pm25 = []
for i in np.unique(spline['field_egg_id']):
    s_i = spline[spline['field_egg_id']==i]
    x = s_i['hour_of_day'].values
    y = [t for t in s_i['pm25'].values]
    c1 = CubicSpline(x[:8],y[:8])
    c2 = CubicSpline(x[8:16],y[8:16])
    c3 = CubicSpline(x[16:24],y[16:24])
    ix = [k/100.0 for k in range(2400)]
    iy = list(np.concatenate((c1(ix[:800]),c2(ix[800:1600]),c3(ix[1600:2400]))))
    fields += [i]*2400
    times += ix
    pm25 += iy

spline_df = pd.DataFrame((fields, times, pm25)).transpose()

spline_df.columns = ['field_egg_id', 'time', 'pm25']

hours_in_day = np.arange(24).astype(float)

spline_df = spline_df[spline_df['time'].isin(hours_in_day)]

spline_mat = np.transpose(spline_df['pm25'].to_numpy().reshape((60,24))).astype(float)

spline_df = pd.DataFrame(spline_mat,columns=df.columns)
df_full = deepcopy(df)
for idx,row in df.iterrows():
    df.loc[idx] = row-spline_df.loc[idx.hour]
df_spline = df_full-df

with open('./mlp_regressor_fulldata.pickle', 'rb') as pickle_file:
    saved_model = pkl.load(pickle_file)

x_tr, x_ts, y_tr, y_ts, spl_tr, spl_ts = [], [], [], [], [], []
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    lat = locs.loc[df.columns]['Longitude'].values
    long = locs.loc[df.columns]['Latitude'].values
    val = row.values
    
    cols = np.array(df.columns)[~np.isnan(val)]
    lat = lat[~np.isnan(val)]
    long = long[~np.isnan(val)]
    val = val[~np.isnan(val)]
    
    t_year = idx.year
    t_month = idx.month
    t_day = idx.day
    t_hour = idx.hour
    t_day_of_week = idx.day_name()

    if(len(val)<30):
        continue
        
    x = np.array([[la,lo,t_year, t_month, t_day, t_hour, t_day_of_week] for la, lo in zip(lat, long)])
    y = np.array(val)
    y = np.expand_dims(y, 1)
    
    x_train, x_test, y_train, y_test, cols_train, cols_test = train_test_split(
        x, y, cols, test_size=0.2, random_state=42
    )
    
    spl_train = spline_df.loc[idx.hour][cols_train].values
    spl_train = np.expand_dims(np.array(spl_train),1)
    spl_test = spline_df.loc[idx.hour][cols_test].values
    spl_test = np.expand_dims(np.array(spl_test),1)
    
    x_tr.append(x_train)
    x_ts.append(x_test)
    y_tr.append(y_train)
    y_ts.append(y_test)
    spl_tr.append(spl_train)
    spl_ts.append(spl_test)
    

ohe = OneHotEncoder()
ohe.fit(np.concatenate(x_tr)[:,[2,3,4,6]])
    
ape = []
ws = 10
for i in tqdm(range(ws,len(x_tr[ws:-ws]))):
    x_train, x_test, y_train, y_test, spl_train, spl_test = x_tr[i-ws:i+ws], x_ts[i], y_tr[i-ws:i+ws], y_ts[i],\
    spl_tr[i-ws:i+ws], spl_ts[i]
    x_train = x_train+x_ts[i-ws:i]+x_ts[i+1:i+ws]
    y_train = y_train+y_ts[i-ws:i]+y_ts[i+1:i+ws]
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    
    x_train = np.concatenate([x_train[:,[0,1,5]].astype(np.float64),ohe.transform(x_train[:,[2,3,4,6]]).toarray()], axis=1)
    x_test = np.concatenate([x_test[:,[0,1,5]].astype(np.float64),ohe.transform(x_test[:,[2,3,4,6]]).toarray()], axis=1)
    
    local_model = deepcopy(saved_model)
    
    for m in range(5):
        local_model.partial_fit(x_train, y_train)
    
    pred = local_model.predict(x_test)
    pred = np.expand_dims(pred, axis=1)
    
    ape.append(np.abs((np.exp(y_test+spl_test)-np.exp(pred+spl_test))/np.exp(y_test+spl_test)))
    
    ape_arr = np.concatenate(ape)
    mape = np.mean(ape_arr)
    print(i, mape)