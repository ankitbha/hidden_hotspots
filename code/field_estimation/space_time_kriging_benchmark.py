import os
import sys
seed = int(sys.argv[1])
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
filepath_root = '/scratch/ab9738/hidden_hotspots/'

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
distances = pd.read_csv('/scratch/ab9738/hidden_hotspots/data/combined_distances.csv', index_col=[0])
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
spline_df = spline_df.drop(['Pusa_IMD'], axis=1)
df = df.drop(['Pusa_IMD'], axis=1)
spline_df = spline_df.mean(axis=1)
df_full = deepcopy(df)
for idx,row in df.iterrows():
    df.loc[idx] = row-spline_df.loc[idx.hour]
df_spline = df_full-df
    
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
    
    x_train, x_test, y_train, y_test, z_train, z_test, vals_train, vals_test, cols_train, cols_test = train_test_split(
        x, y, z, vals, cols, test_size=0.2, random_state=seed
    )
    
    x_train = np.concatenate([x_train,x_win])
    y_train = np.concatenate([y_train,y_win])
    z_train = np.concatenate([z_train,z_win])
    vals_train = np.concatenate([vals_train,vals_win])


    OK3D = OrdinaryKriging3D(
        x_train,
        y_train,
        z_train,
        vals_train,
        variogram_model="linear",
        verbose=False,
        enable_plotting=False,
    )

    vals_pred, ss_pred = OK3D.execute("points", x_test, y_test, z_test)
    spl_test = spline_df.loc[idx.hour]
    ape_for_row = np.abs((np.exp(vals_test+spl_test)-np.exp(vals_pred+spl_test))/np.exp(vals_test+spl_test))
    se_for_row = np.square(np.exp(vals_test+spl_test)-(np.exp(vals_pred+spl_test)))
    return(ape_for_row,se_for_row)

result_list = Parallel(n_jobs=12)(delayed(process_row)(idx,row) for idx,row in df[4:-4].iterrows())
ape = [x for (x,y) in result_list]
se = [y for (x,y) in result_list]
ape_arr = np.concatenate(ape)
se_arr = np.concatenate(se)
mape = np.mean(ape_arr)
rmse = np.sqrt(np.mean(se_arr))
print(mape, rmse)