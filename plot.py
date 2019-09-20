
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

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


if __name__ == '__main__':

    mids_list = ['CBC7', 'A9BE', 'C0A7', '72CA', 'BC46', '20CA', 'EAC8', '113E', 'E8E4', '603A']
    
    # for mid in mids_list:
    #     for quant in ['reg', 'excLF']:
    #         for modelname in ['glm', 'elastic']:
    #             print(mid, quant, modelname)
    #             plot_bars_K_histlen_perf(modelname, quant, 'kaiterra', 'pm25', 'v2', mid)
    plot_bars_mid_quant_bestperf(mids_list, 'kaiterra', 'pm25', 'v2')
