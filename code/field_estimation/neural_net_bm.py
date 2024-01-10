import os
import sys
seed = int(sys.argv[1])
import pytz
import numpy as np
import pandas as pd
from geopy import distance
import datetime
from copy import deepcopy
import pickle as pkl
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
os.environ['PYTHONWARNINGS']='ignore'
import random
random.seed(seed)
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
spline_df = spline_df.drop(['Pusa_IMD'], axis=1)
df = df.drop(['Pusa_IMD'], axis=1)
spline_df = spline_df.mean(axis=1)
df_full = deepcopy(df)
for idx,row in df.iterrows():
    df.loc[idx] = row-spline_df.loc[idx.hour]
df_spline = df_full-df


x_tr, x_ts, y_tr, y_ts, spl_tr, spl_ts = [], [], [], [], [], []
for idx, row in df.iterrows():
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

        
    x = np.array([[la,lo,t_year, t_month, t_day, t_hour, t_day_of_week] for la, lo in zip(lat, long)])
    y = np.array(val)
    y = np.expand_dims(y, 1)
    
    x_train, x_test, y_train, y_test, cols_train, cols_test = train_test_split(
        x, y, cols, test_size=0.2, random_state=seed
    )
    
    spl_train = np.ones_like(y_train)*spline_df.loc[idx.hour]
    spl_test = np.ones_like(y_test)*spline_df.loc[idx.hour]
    
    x_tr.append(x_train)
    x_ts.append(x_test)
    y_tr.append(y_train)
    y_ts.append(y_test)
    spl_tr.append(spl_train)
    spl_ts.append(spl_test)
    

x_tr = np.concatenate(x_tr)
x_ts = np.concatenate(x_ts)
y_tr = np.concatenate(y_tr)
y_ts = np.concatenate(y_ts)
spl_tr = np.concatenate(spl_tr)
spl_ts = np.concatenate(spl_ts)

ohe = OneHotEncoder()

ohe.fit(x_tr[:,[2,3,4,6]])

x_tr = np.concatenate([x_tr[:,[0,1,5]].astype(np.float64),ohe.transform(x_tr[:,[2,3,4,6]]).toarray()], axis=1)
x_ts = np.concatenate([x_ts[:,[0,1,5]].astype(np.float64),ohe.transform(x_ts[:,[2,3,4,6]]).toarray()], axis=1)


# reg_model = MLPRegressor(hidden_layer_sizes=(1024,256,64,16,), random_state=seed, max_iter=50, verbose=False)
# reg_model.fit(x_tr, y_tr)

# with open('./mlp_regressor_seed_{}.pickle'.format(seed), 'wb') as pickle_file:
#     pkl.dump(reg_model, pickle_file)
    
with open('./mlp_regressor_seed_{}.pickle'.format(seed), 'rb') as pickle_file:
    saved_model = pkl.load(pickle_file)
    

pred = saved_model.predict(x_ts)
pred = np.expand_dims(pred, axis=1)
ape_arr = np.abs(np.exp(y_ts+spl_ts)-np.exp(pred+spl_ts))/np.exp(y_ts+spl_ts)
mape = np.mean(ape_arr)
se_arr = np.square(np.exp(y_ts+spl_ts)-(np.exp(pred+spl_ts)))
rmse = np.sqrt(np.mean(se_arr))
print(mape,rmse)