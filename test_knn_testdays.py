
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.autograd import Variable
from tqdm import tqdm

from geopy import distance
from operator import itemgetter
from datasets import create_dataset_knn_testdays
from kNearestNeighbors import get_latlondict_kaiterra, get_latlondict_gov, get_bearing
from nets import Series


def prettyprint_args(ns, outfile=sys.stdout):
    print('\nInput argument --', file=outfile)

    for k,v in ns.__dict__.items():
        print('{}: {}'.format(k,v), file=outfile)

    print(file=outfile)
    return


def test(models, version, dataset_df, test_begin):

    test_series = dataset_df.iloc[test_begin:,:]

    y = test_series.values[:,0] * 100
    y_pred = np.empty_like(y) * np.nan
    
    lstm_state = models[0].init_lstms(device=device, batch=1)

    dividefactor = {'v1':1, 'v2':3}
    
    for ind in range(test_series.shape[0]):
        
        # determine no of available sensors at this instant
        vals = test_series.iloc[ind,1:]
        locs, = np.where(~np.isnan(vals))
        nfeatures = len(locs)
        assert nfeatures % dividefactor[version] == 0
        if nfeatures == 0:
            # reset LSTM whenever there is a discontinuity
            lstm_state = models[0].init_lstms(device=device, batch=1)
            continue

        K = nfeatures // dividefactor[version]
        
        point_seq = np.empty((1, nfeatures, 1))
        point_seq[0, :, 0] = vals.values[locs]
        point_label = np.ones((1, 1)) * test_series.iloc[ind,0]
        
        point_seq = np.transpose(point_seq, [2, 0, 1])
        point_seq = Variable(torch.from_numpy(point_seq), requires_grad=True).to(device).double()
        point_label = torch.from_numpy(point_label).unsqueeze(2).to(device)

        pred, lstm_state = models[K-1](point_seq, lstm_state)

        y_pred[ind] = pred.detach().squeeze().cpu().numpy() * 100.0
        
    return y, y_pred



if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Spatio-temporal LSTM for air quality prediction (testing only)')
    parser.add_argument('datafilepath', help='Input data dump (kaiterra_fieldeggid_*_panel.csv or govdata_*_panel.csv)')
    parser.add_argument('source', choices=('kaiterra', 'govdata'), help='Source of the data')
    parser.add_argument('knn_version', choices=('v1', 'v2'), help='Version 1 or 2')
    parser.add_argument('--testdays', required=True, help='File containing test days')
    parser.add_argument('--sensor', choices=('pm25', 'pm10'), default='pm25', help='Type of sensory data')
    # parser.add_argument('maxneighbors', type=int, choices=range(1,11), help='Max number of nearest neighbors to use')
    # parser.add_argument('--monitorid', help='Sensor at which location the test performance should be reported')
    # parser.add_argument('--history', dest='histlen', type=int, default=32, help='Length of history used in training (to get test split)')
    # parser.add_argument('--stride', type=int, default=1, help='Length of stride through data')
    parser.add_argument('--hidden', type=int, default=256, help='Hidden layer size')
    args = parser.parse_args()

    prettyprint_args(args)

    latlondict = get_latlondict_kaiterra() if args.source == 'kaiterra' else get_latlondict_gov()

    maxneighbors = 10
    
    colordict = {'v1':('r', 'm'), 'v2':('b', 'c')}
    
    # use cuda if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # savesuffix
    savesuffix = os.path.splitext(args.testdays)[0].rsplit('_', 1)[1]
    
    # get dataset (THIS IS BECAUSE OF SEPARATE FORMATS OF INPUT FILES,
    # WHICH MUST BE CHANGED)
    if args.source == 'kaiterra':
        df_all = pd.read_csv(args.datafilepath, usecols=[0,1,5], index_col=[0,1], parse_dates=True, squeeze=True)
    else:
        df_all = pd.read_csv(args.datafilepath, usecols=[0,1,2], index_col=[0,1], parse_dates=True, squeeze=True)
        
    # load all models for all values of K
    models = []
    numsegments_dict = {'v1':lambda K: K+1, 'v2':lambda K: 3*K+1}
    
    sys.stdout.write('Loading all the models ... ')
    for K in tqdm(range(1, maxneighbors+1)):
        
        model = Series(batchsize=1,
                       historylen=1,
                       numsegments=numsegments_dict[args.knn_version](K),
                       hiddensize=args.hidden).to(device).double()
        
        # load saved model parameters
        savepath = 'models/model_{}_{}_K{:02d}_{}.pth'.format(args.source, args.sensor, K, args.knn_version)
        
        if not os.path.exists(savepath):
            print('No saved model available for K = {}! Please run training script for that with these parameters first.'.format(K), file=sys.stderr)
            raise sys.exit(-1)
        
        model.load_state_dict(torch.load(savepath))
        model.eval()
        models.append(model)
    
    print('done.')
    
    print('Beginning testing ...')
    
    # compute test for every day in the testdays
    testdays_table = pd.read_csv(args.testdays, index_col=[0], parse_dates=True)

    errors_allstats_0 = np.empty((testdays_table.shape[0], 8)) * np.nan
    errors_allstats_bl1 = np.empty((testdays_table.shape[0], 8)) * np.nan
    errors_allstats_bl2 = np.empty((testdays_table.shape[0], 8)) * np.nan
    errors_allstats_bl3 = np.empty((testdays_table.shape[0], 8)) * np.nan
    
    # for count, monitorid in enumerate(MONITORIDS, 1):
    for dayno, testrow in enumerate(testdays_table.iterrows()):

        testdate, testrow = testrow
        
        # for every testing location
        testlocs = testrow[testrow == 0].index
        inputlocs = testrow[testrow == 1].index

        #print(count, monitorid)
        print('Day {}/{}: {}, Num input locs: {}, Num test locs: {}'.format(dayno + 1, testdays_table.shape[0], testdate, inputlocs.size, testlocs.size))

        if testlocs.size == 0 or inputlocs.size == 0:
            continue
        
        # make the dataset for this set of input locs
        tsindex = pd.date_range(testdate, periods=96, freq='15T', name='Timestamp')
        df_testday = df_all.loc[(slice(None), tsindex)]
        
        # create new empty dataframe with 3 columns for every sensor
        # since we are also storing distance and heading (theta)
        numneighbors = len(inputlocs) if len(inputlocs) <= 10 else 10
        if args.knn_version == 'v1':
            dataset_knn = np.empty((len(tsindex), numneighbors + 1)) * np.nan
        else:
            dataset_knn = np.empty((len(tsindex), 3*numneighbors + 1)) * np.nan
        
        mape_day_list = []
        rmse_day_list = []
        
        for testloc in tqdm(testlocs):
            
            # set first column to be the target data
            dataset_knn[:,0] = df_testday.loc[(testloc, slice(None))].values / 100.0
            
            # compute and store the sorted distances of every monitor
            # from testloc as well as the compass bearings
            dists_bearings_list = []
            for inputloc in inputlocs[:numneighbors]:
                dist = distance.distance(latlondict[inputloc], latlondict[testloc]).meters
                bearing = get_bearing(latlondict[testloc], latlondict[inputloc])
                dists_bearings_list.append((inputloc, dist, bearing))
            
            # sort by distance
            dists_bearings_list.sort(key=itemgetter(1))

            # fill up remaining columns
            count = 1
            if args.knn_version == 'v1':
                for tup in dists_bearings_list:
                    vals = df_testday.loc[(tup[0], slice(None))].values
                    dataset_knn[:,count] = (vals * 1e7) / (tup[1]**2)
                    count += 1
                dataset_knn[:,1:] /= 100.0
            else:
                for tup in dists_bearings_list:
                    vals = df_testday.loc[(tup[0], slice(None))].values
                    nanlocs = np.isnan(vals)
                    dataset_knn[:,count] = vals
                    dataset_knn[:,count+1] = tup[1]
                    dataset_knn[nanlocs,count+1] = np.nan
                    dataset_knn[:,count+2] = tup[2]
                    dataset_knn[nanlocs,count+2] = np.nan
                    count += 3
                dataset_knn[:,1::3] /= 100.0
                dataset_knn[:,2::3] /= 1000.0
                dataset_knn[:,3::3] /= 100.0
            
            dataset_df = pd.DataFrame(dataset_knn, index=tsindex)
            
            # neural network testing
            y, y_pred = test(models, args.knn_version, dataset_df, 0)
            
            # simple baselines
            invdists = 1 / np.array([tup[1] for tup in dists_bearings_list])
            inpdata = dataset_knn[:,1:] if args.knn_version == 'v1' else dataset_knn[:,1::3]
            inpdata = np.ma.masked_invalid(inpdata)
            y_pred_bl1 = inpdata.mean(axis=1) * 100.0
            y_pred_bl2 = np.average(inpdata, axis=1, weights=invdists) * 100.0
            y_pred_bl3 = np.average(inpdata, axis=1, weights=invdists**2) * 100.0
            
            # compute errors and plot the real vs predicted
            rmse_0 = np.sqrt(np.nanmean((y - y_pred)**2))
            rmse_1 = np.sqrt(np.nanmean((y - y_pred_bl1)**2))
            rmse_2 = np.sqrt(np.nanmean((y - y_pred_bl2)**2))
            rmse_3 = np.sqrt(np.nanmean((y - y_pred_bl3)**2))
            mape_0 = np.mean(np.fabs(np.ma.masked_invalid(y) - np.ma.masked_invalid(y_pred)) / np.ma.masked_equal(y, 0)) * 100.0
            mape_1 = np.mean(np.fabs(np.ma.masked_invalid(y) - np.ma.masked_invalid(y_pred_bl1)) / np.ma.masked_equal(y, 0)) * 100.0
            mape_2 = np.mean(np.fabs(np.ma.masked_invalid(y) - np.ma.masked_invalid(y_pred_bl2)) / np.ma.masked_equal(y, 0)) * 100.0
            mape_3 = np.mean(np.fabs(np.ma.masked_invalid(y) - np.ma.masked_invalid(y_pred_bl3)) / np.ma.masked_equal(y, 0)) * 100.0
            
            mape_day_list.append((mape_0, mape_1, mape_2, mape_3))
            rmse_day_list.append((rmse_0, rmse_1, rmse_2, rmse_3))
            
            fig = plt.figure(figsize=(14,5))
            ax = fig.add_subplot(111)
            fig.suptitle('Test day: {}'.format(testdate))
            ax.set_title('Monitor: {}, Input locs: {}, RMSE: {:.2f}, MAPE: {:.2f}'.format(testloc, len(inputlocs), rmse_0, mape_0))
            
            # for k in range(K):
            #     ax.plot(x_pm[:, k], color='#AAAAAA')
            ax.plot(y, label='Real')
            ax.plot(y_pred, label='Predicted')
            ax.plot(y_pred_bl1, alpha=0.5, ls='--', label='Simple average')
            ax.plot(y_pred_bl2, alpha=0.5, ls='--', label='Inv dist wtd average')
            ax.plot(y_pred_bl3, alpha=0.5, ls='--', label='Inv distsq wtd average')
            ax.set_xlabel('Time (test period)')
            ax.set_ylabel('{} conc'.format(args.sensor))
            ax.legend()
            # fig.tight_layout()
            # plt.show()
            savedf = pd.DataFrame(data=np.hstack((y[:,None], y_pred[:,None], y_pred_bl1[:,None], y_pred_bl2[:,None], y_pred_bl3[:,None])),
                                  index=tsindex,
                                  columns=['Actual','Predicted', 'Simple spatial average', 'Inv dist weighted spatial average', 'Inv dist sq weighted spatial average'])
            savepath_noext = 'plots/test_{}_{}_{}_testdays_{}_{}_{}'.format(args.source, args.sensor, args.knn_version, savesuffix, testdate.strftime('%Y_%m_%d'), testloc)
            savedf.to_csv(savepath_noext + '.txt')
            fig.savefig(savepath_noext + '.pdf')
            fig.savefig(savepath_noext + '.png')
            plt.close(fig)

        mape_day_list = np.asarray(mape_day_list)
        rmse_day_list = np.asarray(rmse_day_list)

        plt.rc('font', size=20)
        fig = plt.figure(figsize=(12,9))
        fig.suptitle('Test day: {}, Test locs: {}, Input locs: {}'.format(testdate, len(testlocs), len(inputlocs)))
        ax1 = fig.add_subplot(211)
        ax1.plot(mape_day_list[:,0], colordict[args.knn_version][0] + '.-', label='max-K-NN')
        ax1.plot(mape_day_list[:,1], ls='--', c='#606060', label='Simple avg')
        ax1.plot(mape_day_list[:,2], ls='--', c='#A0A0A0', label='Inv dist wtd avg')
        ax1.plot(mape_day_list[:,3], ls='--', c='#E0E0E0', label='Inv dist sq wtd avg')
        ax1.set_title('MAPE of prediction at the test locs')
        ax1.set_xlabel('Test location')
        ax1.set_ylabel('MAPE')
        ax1.set_xticks(np.arange(len(testlocs)))
        ax1.tick_params(labelbottom=0)
        ax1.legend(ncol=4, fontsize='x-small')
        
        ax2 = fig.add_subplot(212, sharex=ax1)
        ax2.set_title('RMSE of prediction at the test locs')
        ax2.plot(rmse_day_list[:,0], colordict[args.knn_version][1] + '.-', label='max-K-NN')
        ax2.plot(rmse_day_list[:,1], ls='--', c='#606060', label='Simple avg')
        ax2.plot(rmse_day_list[:,2], ls='--', c='#A0A0A0', label='Inv dist wtd avg')
        ax2.plot(rmse_day_list[:,3], ls='--', c='#E0E0E0', label='Inv dist sq wtd avg')
        ax2.set_xlabel('Test location')
        ax2.set_ylabel('RMSE')
        ax2.set_xticks(np.arange(len(testlocs)))
        ax2.set_xticklabels(testlocs)
        ax2.legend(ncol=4, fontsize='x-small')

        errors_locs_df = pd.DataFrame(data=np.hstack((mape_day_list, rmse_day_list)),
                                      index=testlocs,
                                      columns=['mape_0', 'mape_bl1', 'mape_bl2', 'mape_bl3', 'rmse_0', 'rmse_bl1', 'rmse_bl2', 'rmse_bl3'])
        savepath_noext = 'results/errors_{}_{}_{}_testdays_{}_{}'.format(args.source, args.sensor, args.knn_version, savesuffix, testdate.strftime('%Y_%m_%d'))
        errors_locs_df.to_csv(savepath_noext + '.txt')
        fig.savefig(savepath_noext + '.pdf')
        fig.savefig(savepath_noext + '.png')
        plt.close(fig)

        mape_day_mean = mape_day_list.mean(axis=0)
        mape_day_min = mape_day_list.min(axis=0)
        mape_day_max = mape_day_list.max(axis=0)
        mape_day_std = mape_day_list.std(axis=0)

        rmse_day_mean = rmse_day_list.mean(axis=0)
        rmse_day_min = rmse_day_list.min(axis=0)
        rmse_day_max = rmse_day_list.max(axis=0)
        rmse_day_std = rmse_day_list.std(axis=0)

        errors_allstats_0[dayno,:] = (mape_day_mean[0], mape_day_min[0], mape_day_max[0], mape_day_std[0],
                                      rmse_day_mean[0], rmse_day_min[0], rmse_day_max[0], rmse_day_std[0])
        
        errors_allstats_bl1[dayno,:] = (mape_day_mean[1], mape_day_min[1], mape_day_max[1], mape_day_std[1],
                                        rmse_day_mean[1], rmse_day_min[1], rmse_day_max[1], rmse_day_std[1])
        
        errors_allstats_bl2[dayno,:] = (mape_day_mean[2], mape_day_min[2], mape_day_max[2], mape_day_std[2],
                                        rmse_day_mean[2], rmse_day_min[2], rmse_day_max[2], rmse_day_std[2])
        
        errors_allstats_bl3[dayno,:] = (mape_day_mean[3], mape_day_min[3], mape_day_max[3], mape_day_std[3],
                                        rmse_day_mean[3], rmse_day_min[3], rmse_day_max[3], rmse_day_std[3])


    
    colnameslist = ['mape_mean', 'mape_min', 'mape_max', 'mape_std', 'rmse_mean', 'rmse_min', 'rmse_max', 'rmse_std']
    
    errors_allstats_0_df = pd.DataFrame(data=errors_allstats_0, index=testdays_table.index, columns=colnameslist)
    errors_allstats_0_df.to_csv('results/errors_{}_{}_{}_testdays_{}_alldays_0.txt'.format(args.source, args.sensor, args.knn_version, savesuffix))

    errors_allstats_bl1_df = pd.DataFrame(data=errors_allstats_bl1, index=testdays_table.index, columns=colnameslist)
    errors_allstats_bl1_df.to_csv('results/errors_{}_{}_{}_testdays_{}_alldays_bl1.txt'.format(args.source, args.sensor, args.knn_version, savesuffix))

    errors_allstats_bl2_df = pd.DataFrame(data=errors_allstats_bl2, index=testdays_table.index, columns=colnameslist)
    errors_allstats_bl2_df.to_csv('results/errors_{}_{}_{}_testdays_{}_alldays_bl2.txt'.format(args.source, args.sensor, args.knn_version, savesuffix))

    errors_allstats_bl3_df = pd.DataFrame(data=errors_allstats_bl3, index=testdays_table.index, columns=colnameslist)
    errors_allstats_bl3_df.to_csv('results/errors_{}_{}_{}_testdays_{}_alldays_bl3.txt'.format(args.source, args.sensor, args.knn_version, savesuffix))
