# ********************************************************************
#
# Functions for frequency domain analysis on the pollution data.
#
# Author: Shiva R. Iyer
#
# ********************************************************************

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import fftpack
#from itertools import chain

import utils


def compute_spectra(df, ntop=100):
    '''
    Compute spectrum for each contiguous region for each sensor.
    '''

    fout = open('freqs_new.txt', 'w')
    fout.write('MID,Sensor,Start,End,Length,Top Periods\n')

    #N_TOP_FRAC = 0.8
    N_TOP = ntop

    #for mid in chain(df1.index.levels[0].values, df2.index.levels[0].values):
    for mid in df.index.levels[0].values:
        
        # get data for this sensor
        #series = df.loc['CBC7'].pm25
        #series = df1.loc[mid,sensor] if mid in df1.index else df2.loc[mid,sensor]
        series = df.loc[mid,sensor]
        
        # extract largest contiguous region
        #_, inds = utils.extract_contiguous([series.values])

        # extract every single contiguous region and show top N_TOP freq components in the series
        count = 0
        for _, inds in utils.contiguous(series.values):
            count += 1

            block = series.iloc[inds]
            N = block.size

            if N < N_TOP:
                continue

            start_time, end_time = block.index[0], block.index[-1]

            y = block.values
            freqs = fftpack.rfftfreq(N) * f_s
            Y = fftpack.rfft(y) / N

            # sort the freq components by magnitude and apply filter
            Y_df = pd.DataFrame(np.hstack((freqs[:,np.newaxis], Y[:,np.newaxis], np.abs(Y[:,np.newaxis]))), columns=['freqs', 'dft', 'dft_abs'])
            Y_filter_df = Y_df.sort_values('dft_abs', ascending=False)
            topfreqs = Y_filter_df.freqs.values
            with np.errstate(divide='ignore'):
                topperiods = 1/topfreqs

            #dft_cumsum_frac = Y_filter_df.dft_abs.cumsum() / Y_filter_df.dft_abs.sum()
            #N_TOP = sum(dft_cumsum_frac <= N_TOP_FRAC)

            Y_filter_df.loc[Y_filter_df.index[N_TOP:], ['dft', 'dft_abs']] = 0
            Y_filter_df.sort_index(inplace=True)
            y_filter_recon = fftpack.irfft(Y_filter_df.dft.values * N)
            block_filter_recon = pd.Series(y_filter_recon, index=block.index, name=sensor + ' recon (N_TOPFREQS = {}/{})'.format(N_TOP,N))

            print(mid, 'Start:', start_time, 'End:', end_time, 'Length:', N)

            fig, axs = plt.subplots(2, 1, figsize=(20,20))
            fig.suptitle('{}, {}, {} to {}'.format(mid, sensor, start_time, end_time))
            block.plot(c='k', ls='-', ax=axs[0])
            block_filter_recon.plot(c='r', ls='--', ax=axs[0])
            # axs[0].plot(freqs, y, 'k-', freqs, y_filter_recon, 'r--')
            axs[0].legend(loc=0, ncol=2, fontsize='small')
            axs[1].stem(freqs[1:], np.abs(Y[1:]))
            axs[1].set_xlabel('Freq (cycles per day)')
            axs[1].set_ylabel('Magnitude of freq component')
            axs[1].set_title('Spectrum')
            axs[1].set_xlim(0, f_s/2)

            fig.subplots_adjust(bottom=0.2)
            fig.savefig('figures/{}_{}_{:03d}_DFT.png'.format(mid, sensor, count))

            plt.close(fig)

            block.to_csv('figures/{}_{}_{:03d}_T.csv'.format(mid, sensor, count), header=True)
            Y_df.to_csv('figures/{}_{}_{:03d}_DFT.csv'.format(mid, sensor, count), columns=['freqs', 'dft'], index=False)
            
            fout.write('{},{},{},{},{},[{}]\n'.format(mid, sensor, start_time, end_time, N, ' '.join(['{:.5f}'.format(per) for per in topperiods[1:N_TOP+1]])))

    fout.close()
    
    return


def filter_data(filt_type):
    '''
    Low-pass or high-pass filter.
    '''
    return
    

if __name__ == '__main__':
    sensor = 'pm25'
    f_s = 96
    
    #plt.style.use('seaborn')
    plt.rc('font', size=28)
    
    #suffix = '2019_Feb_28'
    #df1 = pd.read_csv('data/kaiterra_fieldeggid_15min_{}_panel.csv'.format(suffix), index_col=[0,1], parse_dates=True)
    #df2 = pd.read_csv('data/govdata/govdata_15min_panel.csv', index_col=[0,1], parse_dates=True)
    #df = pd.concat([df1, df2], axis=0, sort=False, copy=False)
    #compute_spectra(df)

    
