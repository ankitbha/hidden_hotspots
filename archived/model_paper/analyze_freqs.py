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


def compute_spectra(df, sensor, ntop=100, savedir=None):
    '''
    Compute spectrum for each contiguous region for each sensor.
    '''

    #fout = open('topperiods_new.txt', 'w')
    #fout.write('MID,Sensor,Start,End,Length,Top Periods\n')

    #N_TOP_FRAC = 0.8
    #N_TOP = ntop

    #for mid in chain(df1.index.levels[0].values, df2.index.levels[0].values):
    for mid in df.index.levels[0].values:

        if not savedir is None:
            savesubdir = os.path.join(savedir, mid)
            if not os.path.exists(savesubdir):
                os.makedirs(savesubdir)
        
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

            if N <= 1:
                continue
            #if N < N_TOP:
            #    continue

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
            Y_df = pd.DataFrame(list(zip(freqs, Y, Ymag, Yarg)), columns=['freqs', 'dft', 'dft_mag', 'dft_arg_deg'])
            #Y_filter_df = Y_df.sort_values('dft_mag', ascending=False)
            #topfreqs = Y_filter_df.freqs.values
            #with np.errstate(divide='ignore'):
            #    topperiods = 1/topfreqs

            #dft_cumsum_frac = Y_filter_df.dft_abs.cumsum() / Y_filter_df.dft_abs.sum()
            #N_TOP = sum(dft_cumsum_frac <= N_TOP_FRAC)

            #Y_filter_df.loc[Y_filter_df.index[N_TOP:], ['dft', 'dft_mag', 'dft_arg_deg']] = 0
            #Y_filter_df.sort_index(inplace=True)
            #Y_filter = Y_filter_df.dft.values
            #y_filter_recon = np.fft.irfft(Y_filter, N)
            #block_filter_recon = pd.Series(y_filter_recon, index=block.index, name=sensor + ' recon (N_TOPFREQS = {}/{})'.format(N_TOP,N))

            print(mid, 'Start:', start_time, 'End:', end_time, 'Length:', N)
            
            #plt.style.use('seaborn')
            plt.rc('font', size=28)
            
            fig, axs = plt.subplots(2, 1, figsize=(20,20))
            fig.suptitle('{}, {}, {} to {}'.format(mid, sensor, start_time, end_time))
            block.plot(c='k', ls='-', ax=axs[0])
            #block_filter_recon.plot(c='r', ls='--', ax=axs[0])
            #axs[0].plot(freqs, y, 'k-', freqs, y_filter_recon, 'r--')
            axs[0].legend(loc=0, ncol=2, fontsize='small')
            axs[1].stem(freqs[1:], Ymag[1:])
            axs[1].set_xlabel('Freq (cycles per day)')
            axs[1].set_ylabel('Magnitude of freq component')
            axs[1].set_title('Spectrum')
            axs[1].set_xlim(0, f_s/2)
            
            fig.subplots_adjust(bottom=0.2)
            if savedir is None:
                plt.show()

            else:
                fig.savefig(os.path.join(savesubdir, '{}_{}_{:03d}_DFT.png'.format(mid, sensor, count)))
                block.to_csv(os.path.join(savesubdir, '{}_{}_{:03d}_T.csv'.format(mid, sensor, count)), header=True)
                Y_df.sort_values('dft_mag', ascending=False, inplace=True)
                Y_df.to_csv(os.path.join(savesubdir, '{}_{}_{:03d}_DFT.csv'.format(mid, sensor, count)),
                            columns=['freqs', 'dft_mag', 'dft_arg_deg'], index=False, float_format='%.4f')
            
            plt.close(fig)

            plt.rcdefaults()
            
            
            #fout.write('{},{},{},{},{},[{}]\n'.format(mid, sensor, start_time, end_time, N, ' '.join(['{:.5f}'.format(per) for per in topperiods[1:N_TOP+1]])))
    
    #fout.close()
    
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


def bin_freqs(inpdir, sensor, n_bins=10, savedir=None):
    '''Bin all the frequencies from all the DFTs in "inpdir", computed
    using the "compute_spectra" function (a better version than
    bin_topperiods, since it takes the entire DFT into account for
    each contiguous region of each sensor, instead of only the top
    N_TOP freq components).
    
    '''
    
    if savedir is None:
        outpath = os.path.join(inpdir, 'allfreqs_{}_{}_hist.csv'.format(sensor, n_bins))
    else:
        outpath = os.path.join(savedir, 'allfreqs_{}_{}_hist.csv'.format(sensor, n_bins))
    
    if os.path.exists(outpath):
        table = np.genfromtxt(outpath, delimiter=',', skip_header=1)
        bins = table[:,0]
        hist = table[1:,1]
        return bins, hist
    
    dftpathlist = glob.glob(os.path.join(inpdir, '*_{}_*_DFT.csv'.format(sensor)))
    dftpathlist.sort()
    
    allfreqs = set()
    
    for dftpath in tqdm(dftpathlist):
        freqs = np.loadtxt(dftpath, delimiter=',', skiprows=1, usecols=[0])
        allfreqs.update(freqs)

    allfreqs = np.array(sorted(allfreqs))

    plt.figure()
    hist, bins, _ = plt.hist(allfreqs, bins=n_bins)
    binpairs = np.vstack((bins[:-1], bins[1:])).T
    
    print(os.linesep + 'Non zero bin pairs and bin counts:', sum(hist>0))
    binpairs_pos = binpairs[hist>0,:]
    hist_pos = hist[hist>0]
    hist_pos_cumsum = hist_pos.cumsum()
    print('{:>10}{:>10}{:>10}{:>10}{:>10}'.format('Bin beg', 'Bin end', 'Bincount', 'Cumcount', 'Cumfrac'))
    for (bin_beg, bin_end), bincount, cumcount, cumfrac in zip(binpairs_pos, hist_pos, hist_pos_cumsum, hist_pos_cumsum/hist_pos.sum()):
        print('{:10.4f}{:10.4f}{:10.0f}{:10.0f}{:10.6f}'.format(bin_beg, bin_end, bincount, cumcount, cumfrac))
    
    print(os.linesep + 'Zero bin pairs:', sum(hist==0))
    print('{:>10}{:>10}'.format('Bin beg', 'Bin end'))
    for bin_beg, bin_end in binpairs[hist==0,:]:
        print('{:10.4f}{:10.4f}'.format(bin_beg, bin_end))
    
    plt.xlabel('Freq component from DFT (cycles per day)')
    plt.ylabel('Bin count')

    plt.savefig(outpath.replace('csv', 'png'))
    
    plt.show()
    
    plt.close()

    with open(outpath, 'w') as fout:
        fout.write('bins,hist' + os.linesep)
        np.savetxt(fout, np.array([[bins[0],np.nan]]), delimiter=',', fmt='%f')
        np.savetxt(fout, np.vstack((bins[1:],hist)).T, delimiter=',', fmt='%f')
    
    return bins, hist


def filter_plot_single(ypath, dftpath, sensor, thres, bins, disp=True, save=False):

    y_df = pd.read_csv(ypath)
    y = y_df[sensor].values
    
    dft_df = pd.read_csv(dftpath, index_col=0)
    dft_df.sort_index(inplace=True)

    dft_df_lp = dft_df.copy(deep=True)
    dft_df_hp = dft_df.copy(deep=True)
    
    dft_df_lp.loc[dft_df.index > thres,:] = 0
    dft_df_hp.loc[((dft_df.index < thres) & (dft_df.index != 0)),:] = 0

    freqs_lp = dft_df.index.values[((dft_df.index.values <= thres) & (dft_df.index.values != 0))]
    freqs_hp = dft_df.index.values[dft_df.index.values >= thres]
    n_lp = len(freqs_lp)
    n_hp = len(freqs_hp)

    if disp:
        print('No of low freq components:', n_lp)
        print('No of high freq components:', n_hp)

    dft_lp = dft_df_lp.dft_mag * np.exp(1j * (dft_df_lp.dft_arg_deg * np.pi / 180))
    dft_hp = dft_df_hp.dft_mag * np.exp(1j * (dft_df_hp.dft_arg_deg * np.pi / 180))
    
    y_lp = np.fft.irfft(dft_lp.values, y.size)
    y_hp = np.fft.irfft(dft_hp.values, y.size)
    
    rmse_lp = np.sqrt(np.mean((y_lp - y)**2))
    mape_lp = np.mean(np.abs((y_lp - y) / np.ma.masked_invalid(y))) * 100
    
    rmse_hp = np.sqrt(np.mean((y_hp - y)**2))
    mape_hp = np.mean(np.abs((y_hp - y) / np.ma.masked_invalid(y))) * 100
    
    if disp:
        print('Low pass approximation -- RMSE: {:.2f} ug/m^3, MAPE: {:.2f} %'.format(rmse_lp, mape_lp))
        print('High pass approximation -- RMSE: {:.2f} ug/m^3, MAPE: {:.2f} %'.format(rmse_hp, mape_hp))

    # plot filtered signals
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8,8))
    ax1.plot(y, label='Raw signal')
    ax1.plot(y_lp, 'r--', label='Low pass')
    
    ax2.plot(y, label='Raw signal')
    ax2.plot(y_hp, 'y--', label='High pass')
    
    fig.suptitle('Threshold = {} cycles per day'.format(thres), fontsize='medium')
    
    ax1.set_title(r'Low pass filter, # components {}, RMSE {:.2f} $\mu g/m^3$, MAPE {:.2f} %'.format(n_lp, rmse_lp, mape_lp), fontsize='small')
    ax1.set_ylabel(sensor + r' conc ($\mu g/m^3$)')
    ax1.legend(ncol=2, fontsize='small')
    
    ax2.set_title(r'High pass filter, # components {}, RMSE {:.2f} $\mu g/m^3$, MAPE {:.2f} %'.format(n_hp, rmse_hp, mape_hp), fontsize='small')
    ax2.set_ylabel(sensor + r' conc ($\mu g/m^3$)')
    ax2.legend(ncol=2, fontsize='small')
    
    ax2.set_xlabel('Time (15 min intervals)')
    
    if save:
        y_filter_df = y_df.copy(deep=False)
        y_filter_df[sensor + '_lowpass'] = y_lp
        y_filter_df[sensor + '_highpass'] = y_hp
        saveprefix = ypath[:-4] + '_filter_thres{:02d}CPD'.format(thres)
        y_filter_df.to_csv(saveprefix + '.csv', index=False)
        fig.savefig(saveprefix + '.png')
    
    # bin the frequencies
    ind_hp = (bins > thres)
    bins_hp = bins[ind_hp]
    
    ind_lp = ~ind_hp
    ind_lp[bins == bins_hp[0]] = True
    bins_lp = bins[ind_lp]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, figsize=(8,8))
    hist_lp, _, _ = ax1.hist(freqs_lp, bins=bins_lp, density=True)
    ax1.set_xticks(bins_lp)
    ax1.set_ylim(0, 1)
    hist_hp, _, _ = ax2.hist(freqs_hp, bins=bins_hp, density=True)
    #ax2.set_xticks(bins_hp)
    ax2.set_ylim(0, 1)
    
    fig.suptitle('Threshold = {} cycles per day'.format(thres), fontsize='medium')
    ax1.set_title('Low pass filter, # components {}'.format(n_lp), fontsize='small')
    ax2.set_title('High pass filter, # components {}'.format(n_hp), fontsize='small')
    ax1.set_ylabel('Bin prob')
    ax2.set_ylabel('Bin prob')
    ax2.set_xlabel('Freq component from DFT (cycles per day)')
    
    if save:
        saveprefix = dftpath[:-4].replace('DFT', 'freqbuckets_filter_thres{:02d}CPD'.format(thres))
        
        with open(saveprefix + '_lowpass.csv', 'w') as fout:
            fout.write('bins,hist' + os.linesep)
            np.savetxt(fout, np.array([[bins_lp[0],np.nan]]), delimiter=',', fmt='%f')
            np.savetxt(fout, np.vstack((bins_lp[1:],hist_lp)).T, delimiter=',', fmt='%f')
        
        with open(saveprefix + '_highpass.csv', 'w') as fout:
            fout.write('bins,hist' + os.linesep)
            np.savetxt(fout, np.array([[bins_hp[0],np.nan]]), delimiter=',', fmt='%f')
            np.savetxt(fout, np.vstack((bins_hp[1:],hist_hp)).T, delimiter=',', fmt='%f')
        
        fig.savefig(saveprefix + '.png')
    
    if disp:
        plt.show()
    
    plt.close('all')
    
    return


def filter_plot(inpdir, sensor, thres, bins, disp=False, save=True):

    '''Low-pass or high-pass filter.
    
    inpdir: folder containing csv files
    
    thres: threshold for cut-off (for either low-pass or high-pass)

    Low-pass filters out the high freq components, which is usually
    noise. High-pass filters does the opposite.
    
    '''

    dftpathlist = glob.glob(os.path.join(inpdir, '*_{}_*_DFT.csv'.format(sensor)))
    dftpathlist.sort()
    ypathlist = [dftpath.replace('DFT', 'T') for dftpath in dftpathlist]

    n_iters = len(ypathlist)
    
    for ypath, dftpath in tqdm(zip(ypathlist, dftpathlist), total=n_iters):
        filter_plot_single(ypath, dftpath, sensor, thres, bins, disp, save)
    
    return


if __name__ == '__main__':
    
    sampling_interval = '3H'
    
    sensor = 'pm25'
    
    inpdir = 'figures/freq_components/raw/{}/'.format(sampling_interval)
    
    # number of samples per day (freq of sampling)
    f_s = None
    if sampling_interval[-1] == 'T':
        f_s = (24 * 60) / int(sampling_interval[:-1])
    elif sampling_interval[-1] == 'H':
        f_s = 24 / int(sampling_interval[:-1])
    elif sampling_interval[-1] == 'D':
        f_s = 1 / int(sampling_interval[:-1])
    
    suffix = '20180501_20191101'
    
    #df1 = pd.read_csv('data/kaiterra_fieldeggid_15min_{}_panel.csv'.format(suffix), index_col=[0,1], parse_dates=True)
    #df2 = pd.read_csv('data/govdata/govdata_15min_panel.csv', index_col=[0,1], parse_dates=True)
    #df = pd.concat([df1, df2], axis=0, sort=False, copy=False)
    df = pd.read_csv('data/kaiterra/kaiterra_fieldeggid_{}_{}_panel.csv'.format(sampling_interval, suffix),
                     index_col=[0,1], parse_dates=True)
    compute_spectra(df, 'pm25', savedir=inpdir)
    
    # threshold for filtering in cycles per day (frequency unit)
    #thres = 1
    
    # binning interval for binning the frequencies
    #n_bins = 100
    
    # bin ALL the frequencies so that those bins can be used for
    # filter_plot and other things
    #bins, hist = bin_freqs(inpdir, sensor, n_bins, 'figures/freq_components/')

    #print('Frequency bins:', bins)

    #dftpathlist = glob.glob(os.path.join(inpdir, '*_{}_*_DFT.csv'.format(sensor)))
    #dftpath = dftpathlist[np.random.randint(0, len(dftpathlist))]
    #dftpath = os.path.join(inpdir, '113E_pm25_001_DFT.csv')
    #dftpath = os.path.join(inpdir, 'EAC8_pm25_008_DFT.csv')
    #dftpath = os.path.join(inpdir, 'AshokVihar_DPCC_pm25_051_DFT.csv')
    #dftpath = os.path.join(inpdir, '2E9C_pm25_002_DFT.csv') # this is an interesting one
    #dftpath = os.path.join(inpdir, 'AyaNagar_IMD_pm25_232_DFT.csv') # this should not be taken seriously since it is of one day
    #dftpath = os.path.join(inpdir, 'RKPuram_DPCC_pm25_212_DFT.csv') # this is good
    #dftpath = os.path.join(inpdir, 'AnandVihar_DPCC_pm25_189_DFT.csv') # this is bad when thres=3, indicates temporal hotspot
    #dftpath = os.path.join(inpdir, 'NSIT_CPCB_pm25_009_DFT.csv') # has 0 values for y
    #dftpath = os.path.join(inpdir, 'VivekVihar_DPCC_pm25_060_DFT.csv') # another interesting one
    #dftpath = os.path.join(inpdir, 'Pusa_IMD_pm25_079_DFT.csv')
    #print(dftpath)
    
    #ypath = dftpath.replace('DFT', 'T')

    #filter_plot_single(ypath, dftpath, sensor, thres, bins, disp=False, save=True)
    #filter_plot(inpdir, sensor, thres, bins, disp=False, save=True)
