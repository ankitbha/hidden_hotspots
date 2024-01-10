import os
import sys
seed=42
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from geopy import distance
import datetime
import tilemapbase
from copy import deepcopy
import pickle as pkl
from PIL import Image
import skimage.measure
import math
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
os.environ['PYTHONWARNINGS']='ignore'
import hyperopt
from joblib import Parallel, delayed
import random
random.seed(seed)
import scipy
import torch
from pykrige.ok import OrdinaryKriging
from pykrige.ok3d import OrdinaryKriging3D
from pykrige.uk import UniversalKriging
from sklearn.model_selection import train_test_split
from scipy.interpolate import CubicSpline


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
df = df.drop(['Pusa_IMD'], axis=1)


def process_row(idx, row):
    window_size = 3
    i = np.where(np.array(df.index) == idx)[0][0]
    df_slice = pd.concat([df[i-window_size:i],df[i+1:i+window_size+1]])
    x_win = locs.loc[df.columns]['Longitude'].values
    x_win = np.tile(x_win,df_slice.shape[0])
    y_win = locs.loc[df.columns]['Latitude'].values
    y_win = np.tile(y_win,df_slice.shape[0])    
    z_win = np.concatenate([np.arange(i-window_size,i),np.arange(i+1,i+window_size+1)])*0.01
    z_win = np.repeat(z_win,len(df.columns))
    vals_win = df_slice.values.flatten()
    x_win = x_win[~np.isnan(vals_win)]
    y_win = y_win[~np.isnan(vals_win)]
    z_win = z_win[~np.isnan(vals_win)]
    vals_win = vals_win[~np.isnan(vals_win)]
    
    x = locs.loc[df.columns]['Longitude'].values
    y = locs.loc[df.columns]['Latitude'].values
    z = np.ones_like(x)*i*0.01
    vals = row.values
    cols = np.array(df.columns)[~np.isnan(vals)]
    x = x[~np.isnan(vals)]
    y = y[~np.isnan(vals)]
    z = z[~np.isnan(vals)]
    vals = vals[~np.isnan(vals)]
    
    x_train = np.concatenate([x,x_win])
    y_train = np.concatenate([y,y_win])
    z_train = np.concatenate([z,z_win])
    vals_train = np.concatenate([vals,vals_win])


    OK3D = OrdinaryKriging3D(
        x_train,
        y_train,
        z_train,
        vals_train,
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
    )
    
    gridx = np.arange(77.01, 77.40, 0.01)
    gridy = np.arange(28.39, 28.78, 0.01)
    gridz = i*0.01
    vals_grid, ss_grid = OK3D.execute("grid", gridx, gridy, gridz)
    vals_grid = np.exp(vals_grid)
    return(vals_grid)

result_list = Parallel(n_jobs=20)(delayed(process_row)(idx,row) for idx,row in df[4:-4].iterrows())
grid_arr = np.concatenate(result_list)
df_heatmaps = pd.DataFrame(grid_arr.reshape(grid_arr.shape[0],1600), index=df.index[4:-4])
df_heatmaps.to_csv("./heatmaps_longitudinal.csv")