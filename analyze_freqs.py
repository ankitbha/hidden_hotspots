# ********************************************************************
#
# Functions for frequency domain analysis on the pollution data.
#
# Author: Shiva R. Iyer
#
# ********************************************************************

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from scipy import fftpack
#from itertools import chain
from tqdm import tqdm

import utils


def compute_spectra(df, sensor, ntop=100):
    '''
    Compute spectrum for each contiguous region for each sensor.
    '''

    fout = open('topperiods_new.txt', 'w')
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

            # np.fft.rfft returns a complex array, which is the
            # correct representation. The output of fftpack.rfft is a
            # bit more confusing and harder to deal with.
            freqs = np.fft.rfftfreq(N) * f_s
            Y = np.fft.rfft(y)

            # compute magnitude and argument for each freq
            Ymag = np.absolute(Y)
            Yarg = np.angle(Y, deg=True)

            # sort the freq components by magnitude and apply filter
            #Y_df = pd.DataFrame([freqs[:,np.newaxis], Y[:,np.newaxis], Ymag[:,np.newaxis], Yarg[:,np.newaxis]], columns=['freqs', 'dft', 'dft_mag', 'dft_arg_deg'])
            Y_df = pd.DataFrame(list(zip(freqs, Y, Ymag, Yarg)), columns=['freqs', 'dft', 'dft_mag', 'dft_arg_deg'])
            # Y_df = pd.DataFrame(columns=['freqs', 'dft', 'dft_mag', 'dft_arg_deg'])
            # Y_df.freqs = freqs
            # Y_df.dft = Y
            # Y_df.dft_mag = Ymag
            # Y_df.dft_arg_deg = Yarg
            Y_filter_df = Y_df.sort_values('dft_mag', ascending=False)
            topfreqs = Y_filter_df.freqs.values
            with np.errstate(divide='ignore'):
                topperiods = 1/topfreqs

            #dft_cumsum_frac = Y_filter_df.dft_abs.cumsum() / Y_filter_df.dft_abs.sum()
            #N_TOP = sum(dft_cumsum_frac <= N_TOP_FRAC)

            Y_filter_df.loc[Y_filter_df.index[N_TOP:], ['dft', 'dft_mag', 'dft_arg_deg']] = 0
            Y_filter_df.sort_index(inplace=True)
            Y_filter = Y_filter_df.dft.values
            y_filter_recon = np.fft.irfft(Y_filter, N)
            block_filter_recon = pd.Series(y_filter_recon, index=block.index, name=sensor + ' recon (N_TOPFREQS = {}/{})'.format(N_TOP,N))

            print(mid, 'Start:', start_time, 'End:', end_time, 'Length:', N)
            
            #plt.style.use('seaborn')
            plt.rc('font', size=28)
            
            fig, axs = plt.subplots(2, 1, figsize=(20,20))
            fig.suptitle('{}, {}, {} to {}'.format(mid, sensor, start_time, end_time))
            block.plot(c='k', ls='-', ax=axs[0])
            block_filter_recon.plot(c='r', ls='--', ax=axs[0])
            # axs[0].plot(freqs, y, 'k-', freqs, y_filter_recon, 'r--')
            axs[0].legend(loc=0, ncol=2, fontsize='small')
            axs[1].stem(freqs[1:], Ymag[1:])
            axs[1].set_xlabel('Freq (cycles per day)')
            axs[1].set_ylabel('Magnitude of freq component')
            axs[1].set_title('Spectrum')
            axs[1].set_xlim(0, f_s/2)
            
            fig.subplots_adjust(bottom=0.2)
            fig.savefig('figures/{}_{}_{:03d}_DFT.png'.format(mid, sensor, count))
            
            plt.close(fig)

            plt.rcdefaults()
            
            block.to_csv('figures/{}_{}_{:03d}_T.csv'.format(mid, sensor, count), header=True)
            Y_df.sort_values('dft_mag', ascending=False, inplace=True)
            Y_df.to_csv('figures/{}_{}_{:03d}_DFT.csv'.format(mid, sensor, count), columns=['freqs', 'dft_mag', 'dft_arg_deg'], index=False, float_format='%.4f')
            
            fout.write('{},{},{},{},{},[{}]\n'.format(mid, sensor, start_time, end_time, N, ' '.join(['{:.5f}'.format(per) for per in topperiods[1:N_TOP+1]])))
    

    fout.close()
    
    return


def bin_topperiods(thres, filter_type='low', nbins=10):
    '''Bin all the top frequencies so we can use the buckets now instead
    of actual freq values, because the freq values are approximate.'''
    
    assert filter_type in ('low', 'high')
    
    dft_df = pd.read_csv('topperiods_new.txt')
    allperiods_list = []
    
    for ind in range(dft_df.shape[0]):
        arr = dft_df.loc[ind, 'Top Periods']
        allperiods_list.extend(eval(arr.replace(' ', ',')))
    
    print('Total number of freq components:', len(allperiods_list))
    
    plt.style.use('seaborn')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.suptitle('All top 100 freq components')
    
    allperiods_unique_list = sorted(set(allperiods_list))
    
    print('Total number of unique freq components:', len(allperiods_unique_list))
    
    # apply threshold
    print('Threshold:', thres)

    if filter_type == 'low':
        allperiods_unique_thres_list = list(filter(lambda ele: thres * ele > 1, allperiods_unique_list))
    else:
        allperiods_unique_thres_list = list(filter(lambda ele: thres * ele < 1, allperiods_unique_list))

    print('Total number of unique freq components after filtering:', len(allperiods_unique_thres_list))
    
    #ax1.stem(allperiods_unique_list, use_line_collection=True)
    ax1.plot(allperiods_unique_thres_list, np.ones(len(allperiods_unique_thres_list)), 'r.')
    ax1.set_yticks([])
    ax1.tick_params(top=0, bottom=0, labeltop=0, labelbottom=0)
    
    hist, bins, _ = ax2.hist(allperiods_unique_thres_list, bins=nbins)
    binpairs = np.vstack((bins[:-1], bins[1:])).T
    print(os.linesep + 'Non zero bin pairs and bin counts:', sum(hist>0))
    binpairs_pos = binpairs[hist>0,:]
    hist_pos = hist[hist>0]
    hist_pos_cumsum = hist_pos.cumsum()
    print('{:>10}{:>10}{:>10}{:>10}{:>10}'.format('Bin beg', 'Bin end', 'Bincount', 'Cumcount', 'Cumfrac'))
    for (bin_beg, bin_end), bincount, cumcount, cumfrac in zip(binpairs_pos, hist_pos, hist_pos_cumsum, hist_pos_cumsum/hist_pos.sum()):
        print('{:10.4f}{:10.4f}{:10.0f}{:10.0f}{:10.6f}'.format(bin_beg, bin_end, bincount, cumcount, cumfrac))
    
    print(os.linesep + 'Zero bin pairs:', sum(hist==0))
    #print(binpairs[hist>0,:].round(4))
    #print(hist[hist>0])
    print('{:>10}{:>10}'.format('Bin beg', 'Bin end'))
    for bin_beg, bin_end in binpairs[hist==0,:]:
        print('{:10.4f}{:10.4f}'.format(bin_beg, bin_end))
    ax2.set_xlabel('Freq component from DFT (time period in days)')
    ax2.set_ylabel('Bin count')
    
    plt.show()
    
    plt.close(fig)
    
    plt.rcdefaults()
    
    return
    

def filter_data(filter_type):

    '''
    Low-pass or high-pass filter.
    '''
    
    freqs_df = pd.read_csv('freqs_new.txt')
    l = []

    for ind in range(freqs_df.shape[0]):
        arr = freqs_df.loc[ind, 'Top Periods']
        l.append(np.array(eval(arr.replace(' ', ','))).round(1))
    
    freqs_df['top_periods'] = l
    freqs_df.drop('Top Periods', axis=0, inplace=True)
    
    return


if __name__ == '__main__':
    
    f_s = 96
    
    #suffix = '2019_Feb_28'
    #df1 = pd.read_csv('data/kaiterra_fieldeggid_15min_{}_panel.csv'.format(suffix), index_col=[0,1], parse_dates=True)
    #df2 = pd.read_csv('data/govdata/govdata_15min_panel.csv', index_col=[0,1], parse_dates=True)
    #df = pd.concat([df1, df2], axis=0, sort=False, copy=False)
    #compute_spectra(df, 'pm25')

    bin_freqs(100)
