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
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

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
def is_hotspot_high(win, ff):
    # "win" is of type numpy.ndarray
    c_ii = (len(win)-1)//2
    c = win[c_ii]
    m = np.maximum(win[:c_ii].max(), win[c_ii+1:].max())
    r = c >= (1 + ff)*m
    return r


def is_hotspot_low(win, ff):
    # "win" is of type numpy.ndarray
    c_ii = (len(win)-1)//2
    c = win[c_ii]
    m = np.minimum(win[:c_ii].min(), win[c_ii+1:].min())
    r = c <= ff*m
    return r


def get_temporal_hotspots(data, mid, sensor, wr, tf, jr, tv):

    df = data.loc[mid,sensor]

    # for window hotspots (types 1 & 3) - a timestamp is marked as a
    # hotspot if the value at that time is greater/lesser than the
    # window around it by threshold fraction "tf"
    ws = 2*wr + 1
    obj = df.rolling(ws, min_periods=ws, center=True)
    res_whigh = obj.apply(is_hotspot_high, raw=True, args=(tf,))
    res_wlow = obj.apply(is_hotspot_low, raw=True, args=(tf,))
    
    # for jump hotspots (types 2 & 4) - a timestamp is marked as a
    # hotspot if the change in the values at that time and the
    # previous timestamp is greater/lesser than the threshold "tv"
    obj = df.rolling(2*jr + 1, min_periods=1, center=True).mean().diff()
    res_jhigh = (obj > tv)
    res_jhigh.loc[obj.isna()] = np.nan
    res_jlow = (obj < -tv)
    res_jlow.loc[obj.isna()] = np.nan

    res_whigh.name, res_wlow.name, res_jhigh.name, res_jlow.name = 'is_whigh', 'is_wlow', 'is_jhigh', 'is_jlow'
    
    res = pd.concat([res_whigh, res_wlow, res_jhigh, res_jlow], axis=1, sort=False)

    return res


def get_spatial_hotspots(data, mid, sensor, locs):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(title='Data analysis',
                                       help='List of implemented analyses on the data',
                                       dest='operation')
    parser_a = subparsers.add_parser('hotspot_temporal')
    parser_a.add_argument('source', choices=('kaiterra', 'govdata'), help='Source (kaiterra or govdata)')
    parser_a.add_argument('sensor', choices=('pm25', 'pm10'), help='Type of sensor data')
    parser_a.add_argument('res_time', choices=('1H', '6H', '12H', '1D', '1W', '1M'))
    parser_a.add_argument('res_space', choices=('0', '1km', '5km'))
    parser_a.add_argument('--h-type', choices=('whigh', 'wlow', 'jhigh', 'jlow'))
    parser_a.add_argument('--start-date', default='20180501')
    parser_a.add_argument('--end-date', default='20200201')
    parser_a.add_argument('--window_radius', '-wr', type=int, default=1, help='Window radius for hotspots of type 1 & 3')
    parser_a.add_argument('--threshold_frac', '-tf', type=float, default=0.5, help='Threshold fraction for hotspots of type 1 & 3')
    parser_a.add_argument('--jump_radius', '-jr', type=int, default=1, help='Radius for smoothing data for hotspots of type 2 & 4')
    parser_a.add_argument('--threshold_val', '-tv', type=float, help='Threshold value for hotspots of type 2 & 4')

    parser_b = subparsers.add_parser('hotspot_spatial')
    parser_b.add_argument('source', choices=('kaiterra', 'govdata'), help='Source (kaiterra or govdata)')
    parser_b.add_argument('sensor', choices=('pm25', 'pm10'), help='Type of sensor data')
    parser_b.add_argument('res_time', choices=('1H', '6H', '12H', '1D', '1W', '1M'))
    parser_b.add_argument('res_space', choices=('0', '1km', '5km'))
    parser_b.add_argument('--window_radius', '-R', type=int, default=0, help='In kilometres')
    parser_b.add_argument('--threshold_frac', '-T', type=float)

    args = parser.parse_args()

    if args.operation is None:
        print('At least one operation operation should be specified.')
        parser.print_usage()
        sys.exit(-1)

    # read in the necessary data files (location and sensor data)
    if args.source == 'kaiterra':
        locs = pd.read_csv('data/kaiterra/kaiterra_locations.csv', index_col=0, usecols=[0,2,3,4])
        filepath = 'data/kaiterra/kaiterra_fieldeggid_{}_{}_{}_panel.csv'.format(args.res_time, args.start_date, args.end_date)
    else:
        locs = pd.read_csv('data/govdata/govdata_locations.csv', index_col=0)
        filepath = 'data/govdata/govdata_{}_{}_{}.csv'.format(args.res_time, args.start_date, args.end_date)

    data = pd.read_csv(filepath, index_col=[0,1], parse_dates=True)

    if args.operation == 'hotspot_temporal':

        savedir = os.path.join('output', 'hotspots', args.source, args.sensor, 'temporal')
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        res_subdir = 'tres{}_sres{}'.format(args.res_time, args.res_space)

        # hotspot_radius = [1,2,3] if args.hotspot_radius is None else [args.hotspot_radius]
        # threshold_frac = [0.5] if args.threshold_frac is None else [args.threshold_frac]
        # smooth_radius = [1,2,3] if args.smooth_radius is None else [args.smooth_radius]
        wr, tf, jr = args.window_radius, args.threshold_frac, args.jump_radius
        tv = args.threshold_val
        if tv is None:
            if args.sensor == 'pm25':
                tv = 100
            elif args.sensor == 'pm10':
                tv = 200
            pass

        name_keys = dict()
        name_keys['whigh'] = name_keys['wlow'] = 'wr{}_tf{:02.0f}'.format(wr, tf*100)
        name_keys['jhigh'] = name_keys['jlow'] = 'jr{}_tv{:03.0f}'.format(jr, tv)

        # compute hotspots for each location and save them
        h_type_list = [args.h_type] if args.h_type is not None else ['whigh', 'wlow', 'jhigh', 'jlow']

        print('Computing following temporal hotspots:', ', '.join(h_type_list))
        print('Hotspot parameters: wr={}, tf={:02.0f}, jr={}, tv={:03.0f}'.format(wr, tf*100, jr, tv))

        hotspots_tables = dict()

        for h_type in h_type_list:
            hotspots_tables[h_type] = pd.DataFrame(index=data.index.levels[1], columns=data.index.levels[0])

        for count1, mid in enumerate(tqdm(data.index.levels[0]), 1):

            #print('Location {}/{}:'.format(count1, len(data.index.levels[0])), mid)

            # compute the hotspots
            hotspots = get_temporal_hotspots(data, mid, args.sensor, wr, tf, jr, tv)

            hotspots_times_all = dict()
            serial_index = np.arange(hotspots.index.size)
            series = data.loc[mid,args.sensor]

            # save the hotspots
            for h_type in h_type_list:
                hotspots_tables[h_type][mid] = hotspots['is_' + h_type].values
                hotspots_times_all[h_type] = serial_index[hotspots['is_' + h_type].values == 1]

                finaldir = os.path.join(savedir, h_type, res_subdir, mid)
                if not os.path.exists(finaldir):
                    os.makedirs(finaldir)

                win_rad = wr if h_type in ('whigh', 'wlow') else jr

                # display hotspots in a figure (one figure for each hotspot instance)
                hotspots_times = hotspots_times_all[h_type]
                for count2, ind in enumerate(hotspots_times, 1):
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    s_ii = 0 if ind-win_rad < 1 else ind-win_rad-1
                    series.iloc[s_ii:ind+win_rad+2].plot(ax=ax, marker='o', ms=6)
                    series.iloc[ind:ind+1].plot(ax=ax, marker='o', c='r', ms=10)
                    fig.suptitle('Location: {}, Sensor: {}'.format(mid, args.sensor))
                    ax.set_title('Hotspot type - ' + h_type)
                    ax.set_xlabel('Time')
                    ax.set_ylabel(args.sensor)
                    fig.savefig(os.path.join(finaldir, 'h_{}_{}_{}.png'.format(h_type, name_keys[h_type], count2)))
                    plt.close(fig)

        for h_type in h_type_list:
            savepath = os.path.join(savedir, h_type, res_subdir, 'table_{}_{}.csv'.format(h_type, name_keys[h_type]))
            print('Saving "{}" hotspot results in'.format(h_type), savepath)
            hotspots_tables[h_type].to_csv(savepath, float_format='%.0f')

    elif args.operation == 'hotspot_spatial':
        pass
