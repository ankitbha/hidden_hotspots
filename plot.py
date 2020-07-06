
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import glob
import sys
import os

from tqdm import tqdm

def plot_hotspots_map(fpath, save=False, disp=False):
    """ Show total count of hotspots at each location on a map. """

    # hotspot parameters
    _, h_type, a, b = os.path.splitext(os.path.basename(fpath))[0].split('_')

    a_str, a_val = a[:2], int(a[2:])
    b_str, b_val = b[:2], int(b[2:])

    tres, sres = os.path.basename(os.path.dirname(fpath)).split('_')

    table = pd.read_csv(fpath, index_col=[0], parse_dates=True, dtype=np.float16)

    pass


def plot_hotspots_temporal_binary(fpath, source, sensor, save=False, disp=False):
    """Show hotspots in a table format based on the binary hotspots table. """

    # hotspot parameters
    _, h_type, a, b = os.path.splitext(os.path.basename(fpath))[0].split('_')

    a_str, a_val = a[:2], int(a[2:])
    b_str, b_val = b[:2], int(b[2:])

    tres, sres = os.path.basename(os.path.dirname(fpath)).split('_')

    table = pd.read_csv(fpath, index_col=[0], parse_dates=True, dtype=np.float16)

    title_str = 'Hotspot temporal {}, {} {}, {} {}, {} {}'.format(h_type, tres, sres, a_str, a_val, b_str, b_val)
    savename = 'hotspots_temporal_{}_{}_{}_{}_{}_{}{}_{}{}'.format(source, sensor, h_type, tres, sres, a_str, a_val, b_str, b_val)

    plt.rc('font', size=16)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    fig.suptitle(title_str)

    for count, mid in enumerate(table.columns, 1):

        tab = table[mid]

        valid = tab.where(tab.isna(), other=count+1)
        valid.plot(ax=ax, c='#00FF00', lw=3, alpha=0.5)
        #gaps = tab.mask(tab.isna(), count+1).mask((tab == 0) | (tab == 1), np.nan)
        #gaps.plot(ax=ax, lw=3, c='r')

        #table[mid].plot(kind='bar', bottom=count-0.45, ax=ax)
        values = tab.where(tab==1, other=np.nan) + count
        values.plot(ax=ax, ls='none', marker='o', c='k')

    ax.set_yticks(np.arange(2, count+3))
    ax.set_yticklabels(table.columns.values, fontsize='x-small')
    ax.tick_params(axis='y', right=0, left=1, length=10)
    ax.grid(axis='y')
    # fig.tight_layout()

    if disp:
        plt.show()

    if save:
        fig.savefig('figures/' + savename + '.png')

    plt.close(fig)

    totalcount = table.sum(axis=1).fillna(0).astype(int)
    totalcount.name = 'hotspot_count'
    fig = plt.figure(figsize=(8,2))
    ax = fig.add_subplot(111)
    fig.suptitle(title_str)
    totalcount.plot(ax=ax)
    ax.yaxis.set_major_locator(plt.MultipleLocator())
    ax.tick_params(axis='y', right=0, left=1, length=10)
    ax.grid(axis='y')

    if disp:
        plt.show()

    if save:
        fig.savefig('figures/' + savename + '_totalcount.png')
        totalcount.sort_values(ascending=False).to_csv('figures/' + savename + '_totalcount.csv', header=True)

    plt.close(fig)

    plt.rcdefaults()

    return


def plot_cluster_freqbuckets(inpdir, n_clusters, thres_cpd=3, sensor='pm25', min_points=1000, monitor_id=None, save=False):
    import glob
    from sklearn import cluster

    assert 2 <= n_clusters <= 10

    if monitor_id is None:
        flist = glob.iglob(os.path.join(inpdir, '*_{}_*_freqbuckets_filter_thres{:02d}CPD_lowpass.csv'.format(sensor, thres_cpd)))
    else:
        flist = glob.iglob(os.path.join(inpdir, '{}_{}_*_freqbuckets_filter_thres{:02d}CPD_lowpass.csv'.format(monitor_id, sensor, thres_cpd)))
    X = []
    n_lines = []

    for ind, fname in enumerate(flist):
        nameparts = os.path.basename(fname).rsplit('_', 6)
        ts = np.loadtxt(os.path.join(inpdir, '{}_{}_{}_T.csv'.format(nameparts[0], sensor, nameparts[2])), delimiter=',', usecols=[1], skiprows=1)
        if ts.shape[0] > min_points:
            # only consider those portions that have more than a
            # threshold no of points
            X.append(np.loadtxt(fname, delimiter=',', skiprows=2, usecols=[1]))
            n_lines.append(ts.shape[0])

    if len(X) == 0:
        return

    freqbuckets = np.loadtxt(fname, delimiter=',', skiprows=2, usecols=[0])

    X = np.asarray(X)

    if X.shape[0] > n_clusters:
        centroids, labels, _ = cluster.k_means(X, n_clusters)
    else:
        n_clusters = X.shape[0]
        centroids = X
        labels = np.arange(X.shape[0])

    print('Count of data points in each cluster:', np.bincount(labels))

    plotdims_dict = {1: (1,1),
                     2: (1,2),
                     3: (2,2),
                     4: (2,2),
                     5: (2,3),
                     6: (2,3),
                     7: (3,3),
                     8: (3,3),
                     9: (3,3),
                     10: (4,3)}

    rowdim, coldim = plotdims_dict[n_clusters][0], plotdims_dict[n_clusters][1]

    # (1) plot only the centroids
    fig1, axs1 = plt.subplots(rowdim, coldim, squeeze=False,
                            sharex=True, sharey=True,
                            subplot_kw=dict(xlabel='CPD', ylabel='Bin prob'),
                            figsize=(rowdim*3, coldim*4))
    for centroid, ax in zip(centroids, axs1.flat):
        ax.plot(freqbuckets, centroid, 'k-', alpha=0.5)

    fig1.suptitle('All cluster centroids')    

    fig2, axs2 = plt.subplots(rowdim, coldim, squeeze=False,
                            sharex=True, sharey=True,
                            subplot_kw=dict(xlabel='CPD', ylabel='Bin prob'),
                            figsize=(rowdim*3, coldim*4))
    n_points_min = [np.inf] * n_clusters
    n_points_max = [-np.inf] * n_clusters
    for ind, label in enumerate(labels):
        axs2.flat[label].plot(freqbuckets, X[ind,:], 'k-', alpha=0.5)
        if n_lines[ind] < n_points_min[label]:
            n_points_min[label] = n_lines[ind]
        if n_lines[ind] > n_points_max[label]:
            n_points_max[label] = n_lines[ind]
    
    for ind in range(n_clusters):
        axs1.flat[ind].set_title('n_points {} to {}'.format(n_points_min[ind], n_points_max[ind]))
        axs2.flat[ind].set_title('n_points {} to {}'.format(n_points_min[ind], n_points_max[ind]))
    
    fig2.suptitle('All data points')
    
    if save:
        if monitor_id is None:
            save_dir = 'figures/cluster_freqbuckets'
            save_suffix = 'K{:02d}_{}_filter_thres{:02d}_all'.format(n_clusters, sensor, thres_cpd)
        else:
            save_dir = 'figures/cluster_freqbuckets/{}'.format(monitor_id)
            save_suffix = 'K{:02d}_{}_filter_thres{:02d}_{}'.format(n_clusters, sensor, thres_cpd, monitor_id)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig1.savefig(os.path.join(save_dir, 'centroids_' + save_suffix + '.png'))
        fig2.savefig(os.path.join(save_dir, 'clusters_' + save_suffix + '.png'))
    else:
        plt.show()

    plt.close('all')
    
    return
    

def plot_timeseries_weekwise(monitorid):
    
    pass


def plot_bars_K_histlen_perf(modelname, quant, source, sensor, knn_version, mid, K=None, histlen=None):
    
    # plot bars for performance
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if quant == 'reg':
        fig.suptitle('Estimation of {} conc'.format(sensor))
    else:
        fig.suptitle('Estimation of excess over LF baseline of {} conc'.format(sensor))

    mat = np.loadtxt('output/output_models/model_{}_{}_{}_{}_knn_{}_{}.csv'.format(modelname, quant, source, sensor, knn_version, mid), delimiter=',', skiprows=1, usecols=[0,1,2])

    n_hist = 4
    n_K = 10
    
    barw = 0.6/n_hist
    
    for h in range(n_hist):
        x_h = np.arange(1,n_K+1) - (1 + n_hist/2)*barw/2 + h*barw
        rmse_h = mat[mat[:,1] == h+1,2]
        # print(x_h)
        # print(rmse_h)
        ax.bar(x_h, rmse_h, barw, label='hist {}'.format(h))

    ax.legend(fontsize='small', loc=0, ncol=n_hist)
    ax.set_title('RMSE at location {}'.format(mid))
    ax.set_xlabel('No of nearest neighbors in input')
    ax.set_ylabel('RMSE of {} conc (ug/u^3)'.format(sensor))

    # fig.tight_layout()
    
    fig.savefig('figures/model_{}_{}_{}_{}_knn_{}_{}.png'.format(modelname, quant, source, sensor, knn_version, mid))

    plt.close(fig)

    return


def plot_bars_mid_quant_bestperf(mids_list, source, sensor, knn_version):

    rmse_list = [[], []]
    
    for mid in mids_list:

        for count, quant in enumerate(['reg', 'excLF']):
            
            flist = glob.glob('output/output_models/model_*_{}_{}_{}_knn_{}_{}.csv'.format(quant, source, sensor, knn_version, mid))
            flist.sort()
            
            rmse_best = np.inf
            for fpath in flist:
                mat = np.loadtxt(fpath, delimiter=',', skiprows=1, usecols=[0,1,2])
                mat = mat[mat[:,0] == 10,:]
                if mat[:,2].min() < rmse_best:
                    rmse_best = mat[:,2].min()
            
            rmse_list[count].append(rmse_best)

    barw = 0.3
    x_reg = np.arange(1, len(mids_list)+1) - barw/2
    x_excLF = np.arange(1, len(mids_list)+1) + barw/2

    plt.figure()
    plt.suptitle('Comparison of perf between {} conc and excess over LF baseline'.format(sensor))
    plt.title('RMSE of estimation')
    plt.bar(x_reg, rmse_list[0], barw, label='reg')
    plt.bar(x_excLF, rmse_list[1], barw, label='exc LF')
    plt.xticks(np.arange(1, len(mids_list)+1), list(map(str, mids_list)))
    plt.xlabel('Sensor location')
    plt.ylabel('RMSE of {} conc (ug/u^3)'.format(sensor))
    plt.legend(fontsize='medium', loc=0, ncol=2)
    plt.savefig('figures/compare_bestperf_quant_{}_{}_knn_{}.png'.format(source, sensor, knn_version))
    plt.close()
        
    return


def plot_scatter_1nn(mid, source, sensor):
    '''Scatter plot of values from a sensor and its nearest neighbor.'''

    from datasets import read_knn

    fpath = os.path.join('datasets', 'knn_v2_{}'.format(source), 'knn_{}disttheta_{}_K01.csv'.format(sensor, mid))
    if not os.path.exists(fpath):
        print(fpath + ' does not exist!')
        return
    
    df = read_knn(fpath, 'v2', False)

    valid_pos = np.isfinite(df[[mid, 'Monitor_01']].values).all(axis=1)
    n_valid = sum(valid_pos)
    if n_valid == 0:
        print('Dataset for {} is empty!'.format(mid))
        return
    
    valid_times = df.index.values[valid_pos]
    dists = df.Monitor_01_dist.values[valid_pos]
    dists = (dists - dists.min()) / (dists.max() - dists.min())
    c_dists = list(zip(dists, dists, np.ones(n_valid) * 0.1, np.ones(n_valid) * 0.6))

    x = df.loc[valid_pos, mid].values
    y = df.Monitor_01.values[valid_pos]

    from scipy import stats
    pearsonr, pvalue = stats.pearsonr(x,y)
    spearmanr = stats.spearmanr(x,y)
    kendalltau = stats.kendalltau(x,y)
    
    # 'Null hypothesis H: The two quantities are NOT correlated at all.'
    # 
    # p-value then indicates the probability, when H is true, that the
    # correlation coefficients of the two quantities is at least as
    # high as the sample correlation coefficients computed below.
    print('# points:', len(x))
    print('Pearson R: {:.2f}, p-value: {:.2f}'.format(pearsonr, pvalue))
    print('Spearman R: {:.2f}, p-value: {:.2f}'.format(spearmanr.correlation, spearmanr.pvalue))
    print('Kendall tau: {:.2f}, p-value: {:.2f}'.format(kendalltau.correlation, kendalltau.pvalue))
    
    plt.figure()
    plt.scatter(x, y, c=c_dists, label='n = {}'.format(len(x)))
    plt.suptitle('{} scatter plot: {} and 1-NN'.format(sensor, mid))
    plt.title('Pearson {0:.2f} ({1:.2f}), Spearman {2.correlation:.2f} ({2.pvalue:.2f}), Kendall {3.correlation:.2f} ({3.pvalue:.2f})'.format(pearsonr, pvalue, spearmanr, kendalltau))
    plt.xlabel(r'{} conc at {} ($\mu g/m^3$)'.format(sensor, mid))
    plt.ylabel(r'{} conc at nearest neighbor ($\mu g/m^3$)'.format(sensor))
    plt.legend(loc=0)
    #plt.show()
    plt.savefig(os.path.join('figures', 'scatter_1nn', source, mid + '.png'))
    plt.close()
    
    return x, y


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('plot', help='Which plot to make')
    parser.add_argument('--yes', '-y', action='store_true', help='Save all figures')
    parser.add_argument('--disp', '-d', action='store_true', help='Display all figures')
    parser.add_argument('--source', choices=('kaiterra', 'govdata'), default='kaiterra')
    parser.add_argument('--sensor', choices=('pm25', 'pm10'), default='pm25')
    args = parser.parse_args()

    # type of plot
    plot_types = ['linear_model', 'scatter_1nn', 'cluster_freqbuckets', 'hotspots_temporal']
    if not args.plot in plot_types:
        print('Specify a plot type from the following.', file=sys.stderr)
        print(plot_types)
        sys.exit(1)
    
    save_response = args.yes
    if save_response is False:
        save_response = input('Save all figures? [y] ')
        if save_response.strip().lower() == 'y':
            save_response = True
        else:
            save_response = False
    
    if args.plot == 'linear_model':
        mids_list = ['CBC7', 'A9BE', 'C0A7', '72CA', 'BC46', '20CA', 'EAC8', '113E', 'E8E4', '603A']

        for mid in mids_list:
            for quant in ['reg', 'excLF']:
                for modelname in ['glm', 'elastic']:
                    print(mid, quant, modelname)
                    plot_bars_K_histlen_perf(modelname, quant, args.source, args.sensor, 'v2', mid)

        plot_bars_mid_quant_bestperf(mids_list, args.source, args.sensor, 'v2')
    
    elif args.plot == 'scatter_1nn':

        mids1_list = np.loadtxt('data/kaiterra/kaiterra_fieldeggid_locations.csv', delimiter=',', dtype=str, skiprows=1, usecols=[0])
        mids2_list = np.loadtxt('data/govdata/govdata_locations.csv', delimiter=',', dtype=str, skiprows=1, usecols=[0])
        
        for mid in tqdm(mids1_list):
            plot_scatter_1nn(mid, 'kaiterra', args.sensor)
        for mid in tqdm(mids2_list):
            plot_scatter_1nn(mid, 'govdata', args.sensor)            

    elif args.plot == 'cluster_freqbuckets':
        from itertools import chain
        
        mids1_list = np.loadtxt('data/kaiterra/kaiterra_fieldeggid_locations.csv', delimiter=',', dtype=str, skiprows=1, usecols=[0])
        mids2_list = np.loadtxt('data/govdata/govdata_locations.csv', delimiter=',', dtype=str, skiprows=1, usecols=[0])
        
        for mid in tqdm(chain(mids1_list, mids2_list), total=len(mids1_list) + len(mids2_list)):
            plot_cluster_freqbuckets('output/freq_components/location_wise/', 3, monitor_id=mid, save=save_response)
            flist = glob.glob(os.path.join('output/freq_components/location_wise/', '{}_{}_*_freqbuckets_filter_thres03CPD_lowpass.csv'.format(mid, args.sensor)))
            flist.sort()

    elif args.plot == 'hotspots_temporal':

        fpath_list = glob.glob('output/hotspots/{}/{}/temporal/*/*/table_*.csv'.format(args.source, args.sensor))
        fpath_list.sort()
        
        if args.disp:
            plot_hotspots_temporal_binary(fpath_list[0], args.source, args.sensor, save=save_response, disp=args.disp)
        else:
            for count, fpath in enumerate(fpath_list, 1):
                print('{}/{} {}'.format(count, len(fpath_list), fpath))
                plot_hotspots_temporal_binary(fpath, args.source, args.sensor, save=save_response, disp=args.disp)
