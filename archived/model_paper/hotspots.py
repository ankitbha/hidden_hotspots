# ********************************************************************
# Compute different types of hotspots.
#
# Date: May 2020
#
# Author: Shiva R. Iyer
#
# ********************************************************************

# show spatial and temporal hotspots
import os
import sys
import pytz
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tilemapbase
tilemapbase.init(create='True')

from tqdm import tqdm
from geopy import distance

# (1) spatial hotspots are regions in space where there is "unusual"
# activity, which can be defined as one of a) a rapid rise in pollution, b) 
# a prolonged high level when other sensors around it are reporting low
# values, c) a rapid fall in pollution, d) a prolonged low level when other sensors
# are reporting high values

# All these four types can be subdivided further into two more categories: micro 
# and macro hotspots. The distinction lies in the radius of the spatial hotspot. 

# Spatial hotspots can be defined at the granularity of a single day, a typical/average day, 
# a typical/average week, a typical/average month or a typical/average year

# (2) temporal hotspots are regions in time when there is unusual activity, 
# of the same types as a)-d) above.

# Temporal hotspots can be computed from data in various granularities. 
# We choose the following: 1 hour, 6 hour, 24 hour, 1 week, 1 month ...

# Temporal hotspots will have to be defined at a few spatial resolutions:
# each sensor location, and a few chosen radii (1 km, 5 km)


# To compute and show a spatial hotspot:

# Type 2/4: Define a radius and a threshold percentage. For each
# location, if the readings from that location is greater/lesser than
# those from all the sensors in the radius by the percentage, then mark
# that location as a hotspot.

# Type 1/3: Similar to above, but repeat with the diff time series
# instead of raw time series.

# Above two types can be computed for each hour, a typical hour, each
# day, a typical day, each week, a typical week.


# To compute and show a temporal hotspot:

# Type 2/4: Have a sliding window moving over the entire time series,
# and define a threshold percentage. When the reading on a particular
# day is greater/lesser than the values in the sliding window by the
# threshold percentage, then mark that day as a temporal hotspot.

# Type 1/3: Similar to above, but repeat with the diff time series
# instead of raw time series.


# Temporal hotspot
def is_win_thigh(win, ff):
    # "win" is of type numpy.ndarray
    c_ii = (len(win)-1)//2
    c = win[c_ii]
    m = np.maximum(win[:c_ii].max(), win[c_ii+1:].max())
    r = c >= (1 + ff)*m
    return r


def is_win_tlow(win, ff):
    # "win" is of type numpy.ndarray
    c_ii = (len(win)-1)//2
    c = win[c_ii]
    m = np.minimum(win[:c_ii].min(), win[c_ii+1:].min())
    r = c <= ff*m
    return r


def get_hotspots(data, sensor, params, locs, savedir):

    # get data and distances; data is expected to be a pandas.Series,
    # not pandas.DataFrame, containing only one column (either pm25 or
    # pm10)
    df = data.unstack(level=0)
    distances = pd.read_csv('data/combined_distances.csv', index_col=[0])

    # select only the locations that are in the data
    distances = distances.loc[df.columns, df.columns]

    # invalidate diagonal entries so that sensor M does not get
    # counted in the M's radius
    distances[distances == 0] = np.nan

    # res: three digit entries 'abc' or NaN, where a/b/c = 1 or 9
    #
    # a == 9 => thigh, a == 1 => tlow
    # b == 9 => shigh, b == 1 => slow
    # c == 9 => jhigh, c == 1 => jlow
    res = pd.DataFrame(index=df.index, columns=df.columns)

    # (1) WINDOW HOTSPOTS

    # **Temporal Window Hotspot**: a timestamp is marked as a temporal
    # window hotspot if the value at that time is greater/lesser than
    # a window (radius "wtr") around it by a threshold fraction "wttf"
    wts = 2*params['wtr'] + 1
    rolling_wt = df.rolling(wts, min_periods=wts, center=True)
    res_win_thigh = rolling_wt.apply(is_win_thigh, raw=True, args=(params['wttf'],))
    res_win_tlow = rolling_wt.apply(is_win_tlow, raw=True, args=(params['wttf'],))
    res[res_win_thigh == 1] = 900
    res[res_win_tlow == 1] = 100

    # **Spatial Window Hotspot**: A location is marked as a
    # spatial window hotspot if, at a given time, the value at
    # that location is greater/lesser than the max of values in a
    # radius ("wsr") around it by a threshold frac "wstf"
    res_win_shigh = pd.DataFrame(index=df.index, columns=df.columns)
    res_win_slow = pd.DataFrame(index=df.index, columns=df.columns)
    for mid in df.columns:
        neighborhood = (distances.loc[mid] <= params['wsr'] * 1000)
        neighborhood_max = df.loc[:,neighborhood].max(axis=1)
        neighborhood_min = df.loc[:,neighborhood].min(axis=1)
        res_win_shigh.loc[:, mid] = (df[mid] > ((1 + params['wstf']) * neighborhood_max))
        res_win_shigh.loc[neighborhood_max.isna() | df[mid].isna(), mid] = np.nan
        res_win_slow.loc[:, mid]  = (df[mid] < (params['wstf'] * neighborhood_min))
        res_win_slow.loc[neighborhood_min.isna() | df[mid].isna(), mid] = np.nan
    res[(res_win_shigh == 1) & res.notna()] += 90
    res[(res_win_shigh == 1) & res.isna()] = 90
    res[(res_win_slow == 1) & res.notna()] += 10
    res[(res_win_slow == 1) & res.isna()] = 10

    # (2) JUMP HOTSPOTS

    # for jumps: first the data is smoothened using a rolling window
    # of radius "jtr", then every timestamp where the change from the
    # previous timestamp is greater/lesser than the threshold "jtv" is
    # marked, and finally a timestamp+location is marked as a hotspot
    # if the change in the values at that location is the
    # highest/lowest in a radius "jsr"
    rolling_j = df.rolling(2*params['jtr'] + 1, min_periods=1, center=True).mean().diff()
    res_jump_high = pd.DataFrame(index=df.index, columns=df.columns)
    res_jump_low = pd.DataFrame(index=df.index, columns=df.columns)
    for mid in df.columns:
        neighborhood = (distances.loc[mid] <= params['jsr'] * 1000)
        neighborhood_max = rolling_j.loc[:,neighborhood].max(axis=1)
        neighborhood_min = rolling_j.loc[:,neighborhood].min(axis=1)
        res_jump_high.loc[:, mid] = (rolling_j[mid] > params['jtv']).to_numpy() & (rolling_j[mid] > neighborhood_max).to_numpy()
        res_jump_high.loc[rolling_j[mid].isna() | neighborhood_max.isna(), mid] = np.nan
        res_jump_low.loc[:, mid] = (rolling_j[mid] < -params['jtv']).to_numpy() & (rolling_j[mid] < neighborhood_min).to_numpy()
        res_jump_low.loc[rolling_j[mid].isna() | neighborhood_min.isna(), mid] = np.nan
    res[(res_jump_high == 1) & res.notna()] += 9
    res[(res_jump_high == 1) & res.isna()] = 9
    res[(res_jump_low == 1) & res.notna()] += 1
    res[(res_jump_low == 1) & res.isna()] = 1

    suffix = 'wtr{}_wttf{:02.0f}_wsr{}_wstf{:02.0f}_jtr{}_jtv{:03.0f}_jsr{}'.format(params['wtr'], params['wttf']*100,
                                                                                    params['wsr'], params['wstf']*100,
                                                                                    params['jtr'], params['jtv'], params['jsr'])
    finaldir = os.path.join(savedir, sensor, suffix)
    if not os.path.exists(finaldir):
        os.makedirs(finaldir)

    res.to_csv(os.path.join(finaldir, 'table.csv'.format(suffix)), float_format='%.0f')

    # illustrate each hotspot occurrence on a combined time
    # series-cum-map plot. show all the type of hotspots in the
    # figures -- high-time/high-space, high-time/low-space,
    # low-time/high-space, low-time/low-space, high-time, low-time,
    # high-space, low-space, high-jump, low-jump
    serial_index = np.arange(res.index.size)

    lon_max, lat_max = locs.Longitude.max(), locs.Latitude.max()
    lon_min, lat_min = locs.Longitude.min(), locs.Latitude.min()
    #lon_center, lat_center = locs.Longitude.mean(), locs.Latitude.mean()
    #lat_pad = 1.1 * max(lat_center - lat_min, lat_max - lat_center)
    #lon_pad = 1.1 * max(lon_center - lon_min, lon_max - lon_center)
    #extent = tilemapbase.Extent.from_lonlat(lon_center - lon_pad,
    #                                        lon_center + lon_pad,
    #                                        lat_center - lat_pad,
    #                                        lat_center + lat_pad)
    D_true = distance.distance((lat_max, lon_max), (lat_min, lon_min)).km
    x_max, y_max = tilemapbase.project(lon_max, lat_max)
    x_min, y_min = tilemapbase.project(lon_min, lat_min)
    D_proj = np.sqrt((x_min - x_max)**2 + (y_min - y_max)**2)
    wsr_proj = params['wsr'] * D_proj / D_true
    jsr_proj = params['jsr'] * D_proj / D_true
    ang_wsr_pts = np.linspace(0, 2*np.pi, 41)
    ang_jsr_pts = np.linspace(0, 2*np.pi, 41) + ang_wsr_pts[1]/2

    #extent_proj = extent.to_project_3857
    #color_dict = {'Kaiterra' : 'r', 'Govt' : 'b'}
    
    # formula for computing marker size proportional to the pm value
    pm_min, pm_max = 1, data.max()
    ms_min, ms_max = 1, 300
    size_ratio = (ms_max - ms_min) / (pm_max - pm_min)

    plt.rc('font', size=12)
    #tile = tilemapbase.tiles.Stamen_Toner_Background
    #tile = tilemapbase.tiles.Carto_Light
    tile = tilemapbase.tiles.Stamen_Terrain

    #mid_list = res.columns.drop(['113E', '1FD7', '20CA', '2E9C', '3ACF', '498F', '4BE7', '56C3', '5D7A'])
    mid_list = res.columns
    for jj, mid in enumerate(mid_list, 1):
        
        print('{}/{} {}'.format(jj, len(mid_list), mid))

        # create directory for saving
        subdir = os.path.join(finaldir, mid)
        if not os.path.exists(subdir):
            os.makedirs(subdir)

        series = data.loc[mid]

        # compute projections for plotting
        lon_mid, lat_mid = locs.loc[mid].Longitude, locs.loc[mid].Latitude
        x_mid, y_mid = tilemapbase.project(lon_mid, lat_mid)

        wsr_neighborhood = distances.columns[distances.loc[mid] < params['wsr'] * 1000]
        jsr_neighborhood = distances.columns[distances.loc[mid] < params['jsr'] * 1000]
        neighborhood = wsr_neighborhood if params['wsr'] >= params['jsr'] else jsr_neighborhood
        if len(neighborhood) > 0:
            x_pts, y_pts = zip(*[tilemapbase.project(locs.loc[l].Longitude, locs.loc[l].Latitude) for l in neighborhood])
        else:
            x_pts, y_pts = [], []

        # plot the neighborhood radius
        x_wsr_pts, y_wsr_pts = x_mid - wsr_proj*np.cos(ang_wsr_pts), y_mid + wsr_proj*np.sin(ang_wsr_pts)
        x_jsr_pts, y_jsr_pts = x_mid - jsr_proj*np.cos(ang_jsr_pts), y_mid + jsr_proj*np.sin(ang_jsr_pts)

        suptitle_str = 'Location: {}, Sensor: {}'.format(mid, sensor)

        length = 2.2 * max(params['wsr'], params['jsr']) * D_proj/D_true
        extent = tilemapbase.Extent.from_centre_lonlat(lon_mid, lat_mid, aspect=1.3, ysize=length)
        plotter = tilemapbase.Plotter(extent, tile, width=300)

        indices = serial_index[res[mid].notna()]

        count_dict = dict()
        for ind in tqdm(indices):
            code = res[mid].iloc[ind]
            if not code in count_dict:
                count_dict[code] = 0
            count_dict[code] += 1

            win_rad = params['wtr'] if divmod(code, 100)[0] != 0 else params['jtr']
            
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            s_ii = 0 if ind-win_rad < 1 else ind-win_rad-1
            series.iloc[s_ii:ind+win_rad+2].plot(ax=ax1, marker='o', ms=6, fontsize='small')
            series.iloc[ind:ind+1].plot(ax=ax1, marker='o', c='r', ms=10, fontsize='small')
            fig1.suptitle(suptitle_str)
            ax1.set_title('Hotspot {:03d}, {}'.format(code, series.index[ind]))
            ax1.set_xlabel('Time')
            ax1.set_ylabel(sensor)
            
            # size of marker should be proportional to "pm" value; we
            # make a linear relationship between pm value and marker
            # size
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            plotter.plot(ax2, tile)
            #print('Location:', mid)
            #print('Value:', df[mid].iloc[ind])
            #print('WSR Radius:', wsr_radius)
            #print('WSR Radius values:', df.iloc[ind].loc[wsr_radius].values)
            #print('WSR Radius max:', df.iloc[ind].loc[wsr_radius].max())
            #print('JSR Radius:', jsr_radius)
            #print('JSR Radius values:', df.iloc[ind].loc[jsr_radius].values)
            #print('JSR Radius max:', df.iloc[ind].loc[jsr_radius].max())

            ms = (df.iloc[ind].loc[mid] - pm_min) * size_ratio + ms_min
            ax2.scatter(x_mid, y_mid, marker='.', alpha=0.5, color='r', s=ms**2, edgecolors='none')

            ms_pts = [((pm - pm_min) * size_ratio + ms_min)**2 for pm in df.iloc[ind].loc[neighborhood]]
            ax2.scatter(x_pts, y_pts, marker='.', alpha=0.5, color='b', s=ms_pts, edgecolors='none')
            # for (l, pm) in df.iloc[ind].iteritems():
            #     x, y = tilemapbase.project(locs.loc[l].Longitude, locs.loc[l].Latitude)
            #     ms = (pm - pm_min) * (ms_max - ms_min) / (pm_max - pm_min) + ms_min
            #     if l == mid:
            #         ax2.scatter(x, y, marker='.', alpha=0.4, color='r', s=ms**2, edgecolors='none')
            #         ax2.text(x, y, l, fontsize='xx-small')
            #     else:
            #         ax2.scatter(x, y, marker='.', alpha=0.8, color='y', s=ms**2, edgecolors='none')
            #     if l in wsr_radius and not np.isnan(pm):
            #        ax2.text(x, y, l, fontsize='xx-small')
            
            # draw a dotted circle showing the radius
            ax2.plot(x_wsr_pts, y_wsr_pts, c='#003300', ls='--', lw=1)
            ax2.plot(x_jsr_pts, y_jsr_pts, c='#003300', ls=':', lw=3)

            fig2.suptitle(suptitle_str)
            ax2.set_title('Hotspot {:03d}, {}'.format(code, series.index[ind]))

            fig1.savefig(os.path.join(subdir, 'h_{:03d}_{:02d}_ts.png'.format(code, count_dict[code])))
            fig2.savefig(os.path.join(subdir, 'h_{:03d}_{:02d}_map.png'.format(code, count_dict[code])))
            plt.close(fig1)
            plt.close(fig2)
        #     break
        # break
    
    return res



if __name__ == '__main__':

    ts_type = lambda arg: pd.Timestamp(arg, tz=pytz.FixedOffset(330))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('source', choices=('kaiterra', 'govdata', 'combined'), help='Source (kaiterra or govdata or combined)')
    parser.add_argument('sensor', choices=('pm25', 'pm10'), help='Type of sensor data')
    parser.add_argument('res_time', choices=('1H', '3H', '6H', '12H', '1D', '1W', '1M'))
    parser.add_argument('res_space', choices=('0', '1km', '5km'))
    parser.add_argument('--start-dt', type=ts_type, help='Start timestamp in IST (=UTC+5:30)')
    parser.add_argument('--end-dt', type=ts_type, help='End timestamp in IST (=UTC+5:30)')
    parser.add_argument('--win-time-radius', '-WTr', dest='wtr', type=int, default=1, 
                        help='Temporal window length for window hotspots (integral multiple of chosen res_time)')
    parser.add_argument('--win-time-threshold-frac', '-WTf', dest='wttf', type=float, default=0.5, 
                        help='Temporal threshold for window hotspots')
    parser.add_argument('--win-spatial-radius', '-WSr', dest='wsr', type=float, default=5, 
                        help='Spatial radius for window hotspots (km)')
    parser.add_argument('--win-spatial-threshold-frac', '-WSf', dest='wstf', type=float, default=0.5, 
                        help='Spatial threshold for window hotspots')
    parser.add_argument('--jump-smooth-radius', '-Jtr', dest='jtr', type=int, default=1, 
                        help='Window length for smoothing data for jump hotspots')
    parser.add_argument('--jump-threshold-val', '-Jv', dest='jtv', type=float, 
                        help='Threshold for jump hotspots')
    parser.add_argument('--jump-spatial-radius', '-Jsr', dest='jsr', type=float, default=5,
                        help='Spatial radius for jump hotspots (km)')

    args = parser.parse_args()

    # read in the necessary data files (location and sensor data)
#     filepath_data_kai = 'data/kaiterra/kaiterra_fieldeggid_{}_current_panel.csv'.format(args.res_time)
#     filepath_data_gov = 'data/govdata/govdata_{}_current.csv'.format(args.res_time)
    filepath_data_kai = 'data/kaiterra/kaiterra_fieldeggid_all_current_panel.csv'
    filepath_data_gov = 'data/govdata/govdata_current.csv'
    filepath_locs_kai = 'data/kaiterra/kaiterra_locations.csv'
    filepath_locs_gov = 'data/govdata/govdata_locations.csv'

    if args.source == 'kaiterra':
        locs = pd.read_csv(filepath_locs_kai, index_col=0, usecols=[0,2,3,4])
        data = pd.read_csv(filepath_data_kai, index_col=[0,1], parse_dates=True)[args.sensor]
    elif args.source == 'govdata':
        locs = pd.read_csv(filepath_locs_gov, index_col=0)
        data = pd.read_csv(filepath_data_gov, index_col=[0,1], parse_dates=True)[args.sensor]
    else:
        locs_kai = pd.read_csv(filepath_locs_kai, index_col=[0])
        locs_kai['Type'] = 'Kaiterra'
        locs_gov = pd.read_csv(filepath_locs_gov, index_col=[0])
        locs_gov['Type'] = 'Govt'
        locs = pd.merge(locs_kai, locs_gov, how='outer', on=['Monitor ID', 'Latitude', 'Longitude', 'Location', 'Type'], copy=False)
        data_kai = pd.read_csv(filepath_data_kai, index_col=[0,1], parse_dates=True)[args.sensor]
        data_gov = pd.read_csv(filepath_data_gov, index_col=[0,1], parse_dates=True)[args.sensor]
        data = pd.concat([data_kai, data_gov], axis=0, copy=False)        

    # filter start and end dates
    start_dt = args.start_dt
    if args.start_dt is None:
        start_dt = data.index.levels[1][0]

    end_dt = args.end_dt
    if args.end_dt is None:
        end_dt = data.index.levels[1][-1]

    # if start & end dates are not in IST, then convert them to IST,
    # assuming they are in UTC if they are not tz-aware
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

    # default values for threshold
    tv = 100 if args.sensor == 'pm25' else 200
    if args.jtv is None:
        args.jtv = tv

    # compute hotspots for each location and save them
    print('Hotspot parameters:')
    print('wtr={} (x {}), wttf={:02.0f}%'.format(args.wtr, args.res_time, args.wttf*100))
    print('wsr={} km, wstf={:02.0f}%'.format(args.wsr, args.wstf*100))
    print('jtr={} (x {}), jtv={:03.0f} ug/m3, jsr={} km'.format(args.jtr, args.res_time, args.jtv, args.jsr))

    hotspots_params = {'wtr' : args.wtr, 'wttf' : args.wttf,
                       'wsr' : args.wsr, 'wstf' : args.wstf,
                       'jtr' : args.jtr, 'jtv' : args.jtv, 'jsr' : args.jsr}

    # suffixes = {'wint' : 'wtr{}_wttf{:02.0f}'.format(args.wtr, args.wttf*100),
    #             'wins' : 'wsr{}_wstf{:02.0f}'.format(args.wsr, args.wstf*100),
    #             'jump' : 'jtr{}_jtv{:03.0f}_jsr{}'.format(args.jtr, args.jtv, args.jsr)}

    savedir = os.path.join('output', 'hotspots_revised', args.source, 
                           '{}_{}'.format(start_dt.strftime('%Y%m%d'), end_dt.strftime('%Y%m%d')), 
                           'tres{}_sres{}'.format(args.res_time, args.res_space))
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # compute and save the hotspots
    hotspots = get_hotspots(data, args.sensor, hotspots_params, locs, savedir)
