# ********************************************************************
# 
# Collection of functions to preprocessing tasks on the data.
#
# Date: Jan 28, 2020
#
# Author: Shiva R. Iyer
#
# ********************************************************************

import os
import pandas as pd


def average_kaiterra_data(ddir, interval, end_dt, start_dt=None, fname='kaiterra_fieldeggid_all_current.csv'):

    # start and end dates for data
    if start_dt is None:
        start_dt = pd.Timestamp('2018-03-01', tz='Asia/Kolkata')
    
    # if start & end dates are not in IST, then convert them to IST
    if not start_dt.tzname != 'IST':
        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize('UTC')
        start_dt = start_dt.tz_convert('Asia/Kolkata')

    if not end_dt.tzname != 'IST':
        if end_dt.tzinfo is None: 
            end_dt = end_dt.tz_localize('UTC')
        end_dt = end_dt.tz_convert('Asia/Kolkata')
    
    fpath = os.path.join(ddir, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError('Input file \"{}\" not found!'.format(fpath))
    
    data = pd.read_csv(fpath, index_col=[2,0], parse_dates=True)
    
    # rename the short ids to make them uppercase
    data.rename(str.upper, axis=0, level=0, inplace=True)
    
    # sort the df by ids first, time next
    data.sort_index(level=0, inplace=True)
    
    # set the timezone in data to be UTC so that the df becomes tz-aware
    data.tz_localize('UTC', level=1, copy=False)
    
    # now, the shift the timestamps in the data to the new timezone
    # (can be done only in tz-aware df)
    data.tz_convert('Asia/Kolkata', level=1, copy=False)
    
    # now, filter through the start and end dates
    data = data.loc[(slice(None), slice(start_dt, end_dt)),:]
    
    # for the averaging operation, first append a column with rounded timestamps
    data.reset_index(inplace=True)
    data['timestamp_round'] = data.time.apply(lambda ts: ts.floor(interval))

    # reindexing so that all timestamps are in same range for every sensor
    newindex = pd.date_range(start_dt, end_dt, freq=interval, closed='left', tz='Asia/Kolkata', name='timestamp_round')

    # do the averaging for each monitor id separately
    grouped = data.groupby('short_id')
    groups = []
    for name, group in grouped:
        df = group.groupby('timestamp_round')[['pm_25', 'pm_10']].mean()
        df = df.reindex(newindex, axis=0)
        df['field_egg_id'] = name
        df['device_udid'] = group.device_udid.iloc[0]
        df.reset_index(inplace=True)
        df.set_index(['field_egg_id', 'timestamp_round'], inplace=True)
        groups.append(df)

    data_avged = pd.concat(groups)
    data_avged.rename({'pm_25':'pm25', 'pm_10':'pm10'}, axis=1, inplace=True)

    # save the final dataframe
    savename = 'kaiterra_fieldeggid_{}_{}_{}_panel.csv'.format(interval, start_dt.strftime('%Y%m%d'), end_dt.strftime('%Y%m%d'))
    data_avged[['pm25', 'pm10']].to_csv(os.path.join(ddir, savename))

    return data_avged
