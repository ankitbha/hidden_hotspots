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
import argparse
import pandas as pd


def average_kaiterra_data(fpath, interval, start_dt=None, end_dt=None):

    # read in the data with index as the tuple (id, timestamp)
    data = pd.read_csv(fpath, index_col=[2,0], parse_dates=True)
    
    # sort the df by ids first, time next
    data.sort_index(level=0, inplace=True)
    
    # start and end dates for data
    if start_dt is None:
        start_dt = pd.Timestamp('2018-03-01', tz='Asia/Kolkata')
    
    # if start & end dates are not in IST, then convert them to IST,
    # assuming they are in UTC
    if start_dt.tzname != 'IST':
        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize('UTC')
        start_dt = start_dt.tz_convert('Asia/Kolkata')
    
    if end_dt.tzname != 'IST':
        if end_dt.tzinfo is None: 
            end_dt = end_dt.tz_localize('UTC')
        end_dt = end_dt.tz_convert('Asia/Kolkata')
    
    if not os.path.exists(fpath):
        raise FileNotFoundError('Input file \"{}\" not found!'.format(fpath))
    
    # rename the short ids to make them uppercase
    data.rename(str.upper, axis=0, level=0, inplace=True)
    
    # set the timezone in data to be UTC so that the df becomes tz-aware
    if data.index.levels[1].tz is None:
        data = data.tz_localize('UTC', level=1, copy=False)
    
    # now, the shift the timestamps in the data to the new timezone
    # (can be done only in tz-aware df)
    data = data.tz_convert('Asia/Kolkata', level=1, copy=False)
    
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

    data_final = data_avged[['pm25', 'pm10']]

    # save the final dataframe
    saveprefix = 'kaiterra_fieldeggid_{}_{}_{}'.format(interval, start_dt.strftime('%Y%m%d'), end_dt.strftime('%Y%m%d'))
    savedir = os.path.dirname(fpath)
    data_final.to_csv(os.path.join(savedir, saveprefix + '_panel.csv'))

    # also save the data for each sensor separately
    subdir = os.path.join(savedir, saveprefix)
    if not os.path.exists(subdir):
        os.mkdir(subdir)

    grouped = data_final.groupby(level=0)
    for name, group in grouped:
        group.reset_index(level=0, drop=True, inplace=True)
        group.to_csv(os.path.join(subdir, name + '.csv'))

    return data_avged


if __name__ == '__main__':

    ts_type = lambda arg: pd.Timestamp(arg, tz='Asia/Kolkata')

    fpath_default = os.path.join('data', 'kaiterra', 'kaiterra_fieldeggid_all_current.csv')

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(title='Preprocessing operations on raw data',
                                       help='List of valid preprocessing operations',
                                       dest='operation')
    parser_a = subparsers.add_parser('average', aliases=['avg'])
    parser_a.add_argument('interval', help='Averaging interval (e.g. 15T, 1H, 1D)')
    parser_a.add_argument('--file', '-f', help='Full path to data file')
    parser_a.add_argument('--start', type=ts_type, help='Start timestamp in IST (=UTC+5:30)')
    parser_a.add_argument('--end', type=ts_type, help='End timestamp in IST (=UTC+5:30)')

    args = parser.parse_args()

    if args.operation is None:
        print('At least one preprocessing operation should be specified.')
        parser.print_usage()

    elif args.operation in ('average', 'avg'):
        if args.file is None:
            print('Input file not provided. Using the default: "{}"'.format(fpath_default))
            args.file = fpath_default
        data_avged = average_kaiterra_data(args.file, args.interval, args.start, args.end)
    
