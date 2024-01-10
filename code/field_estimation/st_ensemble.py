import os
import sys
import pytz
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff()
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
random.seed(42)
import scipy
import torch
from pykrige.ok import OrdinaryKriging
from pykrige.ok3d import OrdinaryKriging3D
from pykrige.uk import UniversalKriging
from sklearn.model_selection import train_test_split
from scipy.interpolate import CubicSpline
import scipy.ndimage
import scipy.stats
import scipy.optimize
from sklearn.linear_model import LinearRegression


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

spline_cols = df.columns # columns needed for creating spline_df
df['Pusa_IMD'] = (df['Pusa_IMD'] + df['Pusa_DPCC'])/2
df['Pusa_DPCC'] = (df['Pusa_IMD'] + df['Pusa_DPCC'])/2
df = np.log(df)

df_ws = pd.read_csv('/scratch/ab9738/pollution_with_sensors/hotspots/source_apportionment/wind_speeds.csv', parse_dates=True)
df_ws = df_ws.sort_values(['Timestamp']).reset_index(drop=True)
df_ws = df_ws.set_index(pd.DatetimeIndex(df_ws['Timestamp']))
df_ws = df_ws[['u-component', 'v-component']].groupby('Timestamp').mean()

brick_kilns = np.load('../source_apportionment/brick_kilns_intensity_80x80.npy')
industries = np.load('../source_apportionment/industries_intensity_80x80.npy')
power_plants = np.load('../source_apportionment/power_plants_intensity_80x80.npy')
population_density = np.load('../source_apportionment/population_density_intensity_80x80.npy')
traffic_06 = np.load('../source_apportionment/traffic_06_intensity_80x80.npy')
traffic_12 = np.load('../source_apportionment/traffic_12_intensity_80x80.npy')
traffic_18 = np.load('../source_apportionment/traffic_18_intensity_80x80.npy')
traffic_00 = np.load('../source_apportionment/traffic_00_intensity_80x80.npy')

def cell_to_cord(i,j,size):
    return(j, size-1-i)
def cord_to_cell(x,y,size):
    return(size-1-y, x)

def gpdm_filter(wind_vector, size):
    filt = np.zeros((size,size))
    dest_i, dest_j = int(size/2), int(size/2)
    dest_x, dest_y = cell_to_cord(dest_i, dest_j, size)
    for i in range(size):
        for j in range(size):
            src_i, src_j = i, j
            src_x, src_y = cell_to_cord(i,j,size)
            unit_wind_vector = wind_vector/np.linalg.norm(wind_vector)
            wind_magnitude = np.linalg.norm(wind_vector)
            if(dest_x!=src_x or dest_y!=src_y):
                distance_vector = np.array([dest_x-src_x, dest_y-src_y])*math.pow(10,3)
                dist_wind = np.dot(distance_vector, unit_wind_vector)
                if(dist_wind<=0):
                    filt[src_i,src_j] = 0
                else:
                    distance_magnitude = np.linalg.norm(distance_vector)
                    dist_per = math.pow(max(math.pow(distance_magnitude,2)-math.pow(dist_wind,2),0),0.5)
                    sigma_y = 213*math.pow(dist_wind*0.001,0.894)
                    baseline_dist = 707
                    dist_wind = max(dist_wind/baseline_dist,1)
                    if(dist_per<650):
                        filt[src_i,src_j] = 1/((dist_wind**3)*wind_magnitude)
                    else:
                        filt[src_i,src_j] = 0
            else:
                filt[src_i,src_j] = 1/(wind_magnitude)        
    return(torch.squeeze(torch.tensor(filt)))

def get_wind_vector(ts):
    ts = np.array([ts]).astype('datetime64[ns]')[0]
    cts = min(df_ws.index, key=lambda x:abs(x-ts))
    idx = df_ws.index.to_list().index(cts)
    v1 = df_ws.iloc[idx].values
    if((cts-ts).total_seconds()>0):
        v2 = df_ws.iloc[idx-1].values
    else:
        v2 = df_ws.iloc[idx+1].values
    ws = v1+((v2-v1)*(abs((cts-ts).total_seconds())/(3600*6)))
    ws = ws*(5.0/18)
    return(ws)

def compute_filter(ts):
    src_radius = 4
    wind_vector = get_wind_vector(ts)
    ts_filter = gpdm_filter(wind_vector, 2*src_radius+1)
    return(ts_filter)

def gpdm_at_ts(idx):
    ts_filter = compute_filter(idx)
    contrib_brick = scipy.signal.convolve2d(brick_kilns, ts_filter, mode='valid')
    contrib_industry = scipy.signal.convolve2d(industries, ts_filter, mode='valid')
    contrib_pop = scipy.signal.convolve2d(population_density, ts_filter, mode='valid')
    if(pd.Timestamp(idx).hour>3 and pd.Timestamp(idx).hour<9):
        traffic = traffic_06
    elif(pd.Timestamp(idx).hour>=9 and pd.Timestamp(idx).hour<15):
        traffic = traffic_12
    elif(pd.Timestamp(idx).hour>=15 and pd.Timestamp(idx).hour<21):
        traffic = traffic_18
    else:
        traffic = traffic_00
    contrib_traffic = scipy.signal.convolve2d(traffic, ts_filter, mode='valid')    
    contributions = np.array([contrib_brick[19:59,16:56], contrib_industry[19:59,16:56], contrib_pop[19:59,16:56], contrib_traffic[19:59,16:56]])
    return(contributions)

def krig_at_ts(idx):
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
    vals = df.loc[idx].values
    cols = np.array(df.columns)[~np.isnan(vals)]
    x = x[~np.isnan(vals)]
    y = y[~np.isnan(vals)]
    z = z[~np.isnan(vals)]
    vals = vals[~np.isnan(vals)]

    x_train, x_test, y_train, y_test, z_train, z_test, vals_train, vals_test, cols_train, cols_test = train_test_split(
        x, y, z, vals, cols, test_size=0.2, random_state=42
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
        exact_values=True,
    )
    
    return(OK3D)



def process_row(idx, row):
    gpd_concentrations = gpdm_at_ts(idx)
    krig_model = krig_at_ts(idx)
    # Set up the student teacher network
    i = np.where(np.array(df.index) == idx)[0][0]
    gridx = np.arange(77.01, 77.40, 0.01)
    gridy = np.arange(28.39, 28.78, 0.01)
    gridz = i*0.01
    krig_vals, krig_std = krig_model.execute("grid", gridx, gridy[::-1], gridz)
    y = krig_vals.flatten()
    y = np.exp(y)
    x = np.zeros((1600,4))
    for i in range(4):
        x[:,i] = gpd_concentrations[i].flatten()
    lr = LinearRegression()
    lr.fit(x,y)
    return(lr.score(x,y))

agreement_record = Parallel(n_jobs=12)(delayed(process_row)(idx,row) for idx,row in df[4:-4].iterrows())
agreement_record = np.array(agreement_record)
print("Mean Agreement",np.mean(agreement_record))
np.save("st_agreement.npy", agreement_record)