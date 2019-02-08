# This script determines the best time period with minimum percentage
# of gaps, and successively removes sensors one or two or multiple at
# a time in order to obtain the highest percentage of valid values.
# 
# WARNING: This script potentially runs for a long time.
#
# Author: Shiva R. Iyer

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from itertools import combinations
from tqdm import tqdm

if __name__ == '__main__':
    
    # the input file should be output of 5 min/15 min averaging
    filepath = 'data/kaiterra_fieldeggid_15min_2019_Feb_05_panel.csv'
    df = pd.read_csv(filepath, index_col=[0,1], parse_dates=True)

    # very important! since pandas stores it as tz-naive dataframe
    # with timestamps converted to UTC by default
    df.tz_localize('UTC', level=1, copy=False)
    df.tz_convert('Asia/Kolkata', level=1, copy=False)

    START_DT = pd.Timestamp(datetime(2018, 4, 1), tz='Asia/Kolkata')
    END_DT = df.index.levels[1].max()
    PERIOD_LENGTH = pd.DateOffset(months=6)

    # select sensors with valid count frac <= 0.3 or 0.4
    grouped = df.groupby(level=0)
    validfracs = grouped.pm25.count() / grouped.pm25.size()
    sensorids_select = validfracs.index[validfracs <= 0.4]
    # sensorids_select_set = set(sensorids_select.values)
    num_sensorids_select = sensorids_select.size

    # now begin
    
    # try to get percentage of valid elements higher than 50, by
    # removing one or more sensors progressively, but only from the
    # selected list above
    num_remove_lim = int(sensorids_select.size / 2)
    
    fout = open('./valid_counts_percents_cutoff40.txt', 'w')

    for num_remove in range(0,num_remove_lim+1):
        
        print('Removing {} sensors'.format(num_remove))

        # try removing every combination of "num_remove" elements from
        # the selected list

        # compute total number of combinations beforehand so that
        # progressbar can be displayed accurately. This is the fastest
        # way to get the no of combinations. scipy.special.comb uses
        # recursion and is very slow!
        num_combs = math.factorial(num_sensorids_select) / (math.factorial(num_remove) * math.factorial(num_sensorids_select - num_remove))
        for tup in tqdm(combinations(sensorids_select.values, num_remove), total=num_combs):
            
            fout.write('Removing {}\n'.format(tup))

            validcount_max = 0
            validpercent_max = 0
            start_dt_max = None
            
            # this is the one of the fastest ways to do this
            df_select = df.loc[(np.setdiff1d(sensorids_select.values, tup, assume_unique=True), slice(None))]
            
            end_dt = END_DT
            start_dt = END_DT - PERIOD_LENGTH
            
            while start_dt >= START_DT:
                df_select_time = df_select.loc[(slice(None), slice(start_dt, end_dt)),:]
                validcount = df_select_time.pm25.count()
                validpercent = validcount * 100 / df_select_time.pm25.size
                if validpercent > validpercent_max:
                    validcount_max = validcount
                    validpercent_max = validpercent
                    start_dt_max = start_dt
                # if validpercent >= 65:
                #     fout.write('\t{}, 6M, {}/{}, {:.3f}%\n'.format(start_dt.date(), validcount, df_select_time.pm25.size, validpercent))
                start_dt = start_dt - pd.DateOffset(days=1)
                end_dt = end_dt - pd.DateOffset(days=1)

            fout.write('\tMax: {}, 6M, {}/{}, {:.3f}%\n'.format(start_dt.date(), validcount_max, df_select_time.pm25.size, validpercent_max))
            fout.flush()

    fout.close()
