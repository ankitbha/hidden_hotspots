
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os


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

    save_response = input('Save all figures? [y] ')
    if save_response.strip().lower() == 'y':
        save_response = True
    else:
        save_response = False

    # mids_list = ['CBC7', 'A9BE', 'C0A7', '72CA', 'BC46', '20CA', 'EAC8', '113E', 'E8E4', '603A']
    mids1_list = np.loadtxt('data/kaiterra/kaiterra_fieldeggid_locations.csv', delimiter=',', dtype=str, skiprows=1, usecols=[0])
    mids2_list = np.loadtxt('data/govdata/govdata_locations.csv', delimiter=',', dtype=str, skiprows=1, usecols=[0])
    
    # for mid in mids_list:
    #     for quant in ['reg', 'excLF']:
    #         for modelname in ['glm', 'elastic']:
    #             print(mid, quant, modelname)
    #             plot_bars_K_histlen_perf(modelname, quant, 'kaiterra', 'pm25', 'v2', mid)

    # plot_bars_mid_quant_bestperf(mids_list, 'kaiterra', 'pm25', 'v2')
    
    # for mid in mids1_list:
    #     print(mid)
    #     plot_scatter_1nn(mid, 'kaiterra', 'pm25')
    
    # for mid in mids2_list:
    #     print(mid)
    #     plot_scatter_1nn(mid, 'govdata', 'pm25')

    from itertools import chain
    for mid in chain(mids1_list, mids2_list):
        # print(mid)
        plot_cluster_freqbuckets('figures/freq_components/location_wise/', 3, monitor_id=mid, save=save_response)
        # flist = glob.glob(os.path.join('figures/freq_components/location_wise/', '{}_pm25_*_freqbuckets_filter_thres03CPD_lowpass.csv'.format(mid)))
        # flist.sort()
        # print(mid, len(flist))
        # print(flist)
