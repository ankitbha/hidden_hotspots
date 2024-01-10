import os
import sys
import pytz
import argparse
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
import hyperopt
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')


source = 'combined'
sensor = 'pm25'
res_time = '1H'
filepath_root = '/scratch/ab9738/pollution_with_sensors/'
spikes_file = filepath_root+'hotspots/spikes_combined_1H.csv'
time_high_file = filepath_root+'hotspots/hotspots_combined_temporalhigh_1H.pkl'
time_low_file = filepath_root+'hotspots/hotspots_combined_temporallow_1H.pkl'
space_high_file = filepath_root+'hotspots/hotspots_combined_spatialhigh_1H.pkl'
space_low_file = filepath_root+'hotspots/hotspots_combined_spatiallow_1H.pkl'

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


with open(space_high_file,'rb') as file:
    shsp_high = pkl.load(file)
    
df_ws = pd.read_csv('/scratch/ab9738/pollution_with_sensors/hotspots/source_apportionment/wind_speeds.csv', parse_dates=True)

df_ws = df_ws.sort_values(['Timestamp']).reset_index(drop=True)

df_ws = df_ws.set_index(pd.DatetimeIndex(df_ws['Timestamp']))

df_ws = df_ws[['u-component', 'v-component']].groupby('Timestamp').mean()

brick_kilns = np.load('brick_kilns_intensity_80x80.npy')
industries = np.load('industries_intensity_80x80.npy')
power_plants = np.load('power_plants_intensity_80x80.npy')
population_density = np.load('population_density_intensity_80x80.npy')
traffic_06 = np.load('traffic_06_intensity_80x80.npy')
traffic_12 = np.load('traffic_12_intensity_80x80.npy')
traffic_18 = np.load('traffic_18_intensity_80x80.npy')
traffic_00 = np.load('traffic_00_intensity_80x80.npy')

def gaussian_plume(src, dest, intensity, H, wind_speed, alpha, z, a, c, d, f, offset):
    if(intensity>0):
        distance_direction = np.array([dest[1]-src[1], dest[0]-src[0]])/math.sqrt((dest[1]-src[1])**2+(dest[0]-src[0])**2) 
        #reversing as lat=y-axis and long=x-axis
        distance_magnitude = distance.distance(src, dest).meters
        distance_vector = distance_magnitude * distance_direction
        unit_wind_vector = wind_speed/math.sqrt(wind_speed[0]**2 + wind_speed[1]**2)
        wind_magnitude = np.linalg.norm(wind_speed)
        distance_wind = np.dot(distance_vector, unit_wind_vector)
        if(distance_wind<=0):
            return(0.0)
        distance_wind = max(distance_wind,offset)
        distance_perpendicular = np.linalg.norm(np.subtract(distance_vector, distance_wind))
        sigma_z = c*math.pow(distance_wind,d)+f
        sigma_y = a*math.pow(distance_wind,0.894)
        concentration = ((alpha*intensity)/(2*math.pi*wind_magnitude*sigma_z*sigma_y))*math.exp(-distance_perpendicular**2/(2*sigma_y**2))*\
        (math.exp(-(z-H)**2/(2*sigma_z**2))+math.exp(-(z+H)**2/(2*sigma_z**2)))
        return(concentration*(10**9))
    else:
        return 0
    
    
def compute_concentration(dest, ts, alpha, wind_speed, stack_height=None, z=6.5, a=213, c=459.7, d=2.094, f=-9.6, offset=100):
    if(stack_height is None):
        stack_height = {'traffic':0, 'brick_kilns':25, 'population_density':10, 'industry':30}
    
    idx_x = int((dest[1]-76.85)/0.01)
    idx_y = 79-int((dest[0]-28.2)/0.01)
    src_radius = 7
    if(pd.Timestamp(ts).hour>3 and pd.Timestamp(ts).hour<9):
        traffic_srcs = traffic_06
    elif(pd.Timestamp(ts).hour>=9 and pd.Timestamp(ts).hour<15):
        traffic_srcs = traffic_12
    elif(pd.Timestamp(ts).hour>=15 and pd.Timestamp(ts).hour<21):
        traffic_srcs = traffic_18
    else:
        traffic_srcs = traffic_00
        
    contrib_brick, contrib_industry, contrib_population, contrib_traffic = 0.0,0.0,0.0,0.0
        
    for i in range(idx_y-src_radius, idx_y+src_radius+1):
        for j in range(idx_x-src_radius, idx_x+src_radius+1):
            src = (28.2+((79-j)*0.01)+0.005, 76.85+(i*0.01)+0.005)
            contrib_brick += gaussian_plume(src, dest, brick_kilns[i,j], stack_height['brick_kilns'], wind_speed, alpha['brick_kilns'],z,a,c,d,f,offset)
            contrib_industry += gaussian_plume(src, dest, industries[i,j], stack_height['industry'], wind_speed, alpha['industry'], z,a,c,d,f,offset)
            # contrib_power += gaussian_plume(src, dest, power_plants[i,j], stack_height['power_plant'], wind_speed, alpha, z,a,c,d,f)
            contrib_population += gaussian_plume(src, dest, population_density[i,j], stack_height['population_density'], wind_speed,\
                                                 alpha['population_density'], z,a,c,d,f,offset)
            contrib_traffic += gaussian_plume(src, dest, traffic_srcs[i,j], stack_height['traffic'], wind_speed, alpha['traffic'], z,a,c,d,f,offset)
            
    total_concentration = contrib_brick + contrib_industry + contrib_population + contrib_traffic
    contrib = [contrib_brick, contrib_industry, contrib_population, contrib_traffic]
    return total_concentration, contrib


def find_sensors_in_region(sensor):
    coord = (locs.loc[sensor]['Latitude'], locs.loc[sensor]['Longitude'])
    region_lat_b, region_lat_t, region_long_l, region_long_r = coord[0]-0.025, coord[0]+0.025, coord[1]-0.025, coord[1]+0.025
    subset_locs = locs[(locs['Latitude']<region_lat_t) & (locs['Latitude']>region_lat_b) &\
                       (locs['Longitude']<region_long_r) & (locs['Longitude']>region_long_l)]
    sensors = np.intersect1d(subset_locs.index.to_numpy(),df.columns)
    return(sensors)

def get_wind_speed_estimate(hsp):
    ts = np.array([hsp[0]]).astype('datetime64[ns]')[0]

    if(pd.Timestamp(ts).hour in [5,11,17,23]):
        ts = ts+np.timedelta64(30,'m')
    else:
        ts = ts-np.timedelta64(30,'m')

    ws = df_ws.loc[ts].values
    ws = ws*(5.0/18)
    return(ws)


ws_timestamps = df_ws.index.to_numpy()
hsp_timestamps = np.array(shsp_high)[:,0].astype('datetime64[ns]')

df_hsp_timestamps = pd.DataFrame(hsp_timestamps)

shsp_toexplain = np.array(shsp_high)[df_hsp_timestamps[0].apply(lambda x: True if x.hour in [5,6,11,12,17,18,23,0] else False).to_numpy()]

np.random.seed(42)
np.random.shuffle(shsp_toexplain)
train, test = shsp_toexplain[:int(0.8*len(shsp_toexplain))], shsp_toexplain[int(0.8*len(shsp_toexplain)):]

def obj_hsp(hsp,alpha,H,z,a,c,d,f,offset):
    wind_speed = get_wind_speed_estimate(hsp)

    sensors = find_sensors_in_region(hsp[1])

    sensors = df.loc[hsp[0]][sensors].dropna().index.to_numpy() 
    obj = 0
    for sensor in sensors:
        dest = (locs.loc[sensor]['Latitude'], locs.loc[sensor]['Longitude'])
        ts = hsp[0]
        computed_val, _ = compute_concentration(dest=dest, ts=ts, alpha=alpha, wind_speed=wind_speed, stack_height=H, z=z, \
                                            a=a, c=c, d=d, f=f, offset=offset)
        measured_val = df.loc[hsp[0]][sensor]
        obj = obj + (measured_val-computed_val)**2
        
    return(obj,len(sensors))



def objective(params):
    
    [alpha_traffic, alpha_brick, alpha_population, alpha_industry, H_traffic, H_brick, H_population, H_industry, offset, stability] = params
    alpha = {'traffic':alpha_traffic, 'brick_kilns':alpha_brick, 'population_density':alpha_population, 'industry':alpha_industry}
    H = {'traffic':H_traffic, 'brick_kilns':H_brick, 'population_density':H_population, 'industry':H_industry}
    # stability = 'A'
    stability_dict = {'A': [213,459.7,2.084,-9.6], 'B':[156,108.2,1.098,2.0], 'C':[104,61.0,0.911,0.0], 'D':[68,44.5,0.516,-13.0],\
                     'E':[50.5,55.4,0.305,-34.0], 'F':[34,62.6,0.180,-48.6]}
    [a,c,d,f] = stability_dict[stability]
    z = 5
    obj_sens = np.array(Parallel(n_jobs=15)(delayed(obj_hsp)(train[i],alpha,H,z,a,c,d,f,offset) for i in range(len(train))))
    obj = np.sum(obj_sens[:,0])
    N = np.sum(obj_sens[:,1])  
    rmse = math.sqrt(obj/N)
    return(rmse)



# def objective(params):
    
#     [alpha_traffic, alpha_brick, alpha_population, alpha_industry, H_traffic, H_brick, H_population, H_industry, offset] = params
#     alpha = {'traffic':alpha_traffic, 'brick_kilns':alpha_brick, 'population_density':alpha_population, 'industry':alpha_industry}
#     H = {'traffic':H_traffic, 'brick_kilns':H_brick, 'population_density':H_population, 'industry':H_industry}
#     stability = 'A'
#     stability_dict = {'A': [213,459.7,2.084,-9.6], 'B':[156,108.2,1.098,2.0], 'C':[104,61.0,0.911,0.0], 'D':[68,44.5,0.516,-13.0],\
#                      'E':[50.5,55.4,0.305,-34.0], 'F':[34,62.6,0.180,-48.6]}
#     [a,c,d,f] = stability_dict[stability]
#     z = 5
#     obj, N = 0, 0
#     for i in tqdm(range(len(train))):

#         hsp = train[i]

#         wind_speed = get_wind_speed_estimate(hsp)

#         sensors = find_sensors_in_region(hsp[1])

#         sensors = df.loc[hsp[0]][sensors].dropna().index.to_numpy() 

#         for sensor in sensors:
#             dest = (locs.loc[sensor]['Latitude'], locs.loc[sensor]['Longitude'])
#             ts = hsp[0]
#             computed_val, _ = compute_concentration(dest=dest, ts=ts, alpha=alpha, wind_speed=wind_speed, stack_height=H, z=z, \
#                                                 a=a, c=c, d=d, f=f, offset=offset)
#             measured_val = df.loc[hsp[0]][sensor]
#             obj = obj + (measured_val-computed_val)**2
#             N = N+1
            
#     rmse = math.sqrt(obj/N)
#     return(rmse)


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

space = [hp.uniform('alpha_traffic', 1, 100),hp.uniform('alpha_brick', 1, 100),hp.uniform('alpha_population', 1, 100),\
         hp.uniform('alpha_industry', 1, 100),hp.uniform('H_traffic',0,3),hp.uniform('H_brick',20,60), hp.uniform('H_population', 0, 20),\
        hp.uniform('H_industry', 30, 60), hp.uniform('offset',100,500), hp.choice('stability',['A','B','C','D','E','F'])]

trials = Trials()

best = fmin(objective, space, algo=tpe.suggest, max_evals=200, trials=trials, trials_save_file='trials_200.hyperopt')

import json
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
with open('best.json', 'w') as fp:
    json.dump(best, fp, cls=NpEncoder)