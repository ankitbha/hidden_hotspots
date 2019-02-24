
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import create_dataset_knn_sensor
from nets import Series
from torch.autograd import Variable

def prettyprint_args(ns, outfile=sys.stdout):
    print('\nInput argument --', file=outfile)

    for k,v in ns.__dict__.items():
        print('{}: {}'.format(k,v), file=outfile)

    print(file=outfile)
    return


def frac_type(arg):
    try:
        val = float(arg)
        if val <= 0 or val >= 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError('train-test split should be in (0,1)')
    return


def test(models, version, dataset_df, test_begin):

    test_series = dataset_df.iloc[test_begin:,:]

    nfeatures = test_series.shape[1] - 1
    
    y = test_series.values[:,0] * 100
    y_pred = np.empty_like(y) * np.nan
    
    lstm_state = models[0].init_lstms(device=device, batch=1)

    dividefactor = {'v1':1, 'v2':3}
    
    for ind in range(test_series.shape[0]):
        
        # determine no of available sensors at this instant
        vals = test_series.iloc[ind,1:]
        locs, = np.where(~np.isnan(vals))
        nfeatures = len(locs)
        if nfeatures == 0:
            # reset LSTM whenever there is a discontinuity
            lstm_state = models[0].init_lstms(device=device, batch=1)
            continue

        point_seq = np.empty((1, nfeatures, 1))
        point_seq[0, :, 0] = vals.values[locs]
        point_label = np.ones((1, 1)) * test_series.iloc[ind,0]
        
        point_seq = np.transpose(point_seq, [2, 0, 1])
        point_seq = Variable(torch.from_numpy(point_seq), requires_grad=True).to(device).double()
        point_label = torch.from_numpy(point_label).unsqueeze(2).to(device)

        K = int(nfeatures / dividefactor[version])
        pred, lstm_state = models[K-1](point_seq, lstm_state)

        y_pred[ind] = pred.detach().squeeze().cpu().numpy() * 100.0
        
    return y, y_pred


# def test(models, version, dataset_df_v2, test_begin):

#     test_series = dataset_df_v2.iloc[test_begin:,:]
    
#     y = test_series.values[:,0] * 100
#     y_pred = np.empty_like(y) * np.nan
    
#     lstm_state = models[0].init_lstms(device=device, batch=1)
        
#     for ind in range(test_series.shape[0]):
        
#         # determine no of available sensors at this instant
#         vals = test_series.iloc[ind,1::3]
#         locs, = np.where(~np.isnan(vals))
#         K = len(locs)
#         if K == 0:
#             # reset LSTM whenever there is a discontinuity
#             lstm_state = model[0].init_lstms(device=device, batch=1)
#             continue

#         dists = test_series.iloc[ind,2::3]
        
#         if version == 'v1':
#             point_seq = np.empty((1, K, 1))
#             point_seq[0, :, 0] = (vals.values[locs] * 1e7) / (dists.values[locs]**2)
#         else:
#             bearings = test_series.iloc[ind,3::3]
#             point_seq = np.empty((1, 3*K, 1))
#             point_seq[0, ::3, 0] = vals.values[locs]
#             point_seq[0, 1::3, 0] = dists.values[locs]
#             point_seq[0, 2::3, 0] = bearings.values[locs]
        
#         point_label = np.ones((1, 1)) * test_series.iloc[ind,0]
        
#         point_seq = np.transpose(point_seq, [2, 0, 1])
#         point_seq = Variable(torch.from_numpy(point_seq), requires_grad=True).to(device).double()
#         point_label = torch.from_numpy(point_label).unsqueeze(2).to(device)

#         pred, lstm_state = models[K-1](point_seq, lstm_state)

#         y_pred[ind] = pred.detach().squeeze().cpu().numpy() * 100.0
        
#     return y, y_pred


def test_K(model, dataset_df, test_begin):
    
    test_series = dataset_df.iloc[test_begin:,:]

    nfeatures = test_series.shape[1] - 1

    y = test_series.values[:,0] * 100
    y_pred = np.empty_like(y) * np.nan
    
    lstm_state = model.init_lstms(device=device, batch=1)
        
    for ind in range(test_series.shape[0]):
        
        if test_series.iloc[ind,1:].isnull().any():
            # reset LSTM whenever there is a discontinuity
            lstm_state = model.init_lstms(device=device, batch=1)
            continue
        
        point_seq = np.empty((1, nfeatures, 1))
        point_seq[0, :, 0] = test_series.iloc[ind,1:]
        point_label = np.ones((1, 1)) * test_series.iloc[ind,0]
        
        point_seq = np.transpose(point_seq, [2, 0, 1])
        point_seq = Variable(torch.from_numpy(point_seq), requires_grad=True).to(device).double()
        point_label = torch.from_numpy(point_label).unsqueeze(2).to(device)
        
        pred, lstm_state = model(point_seq, lstm_state)
        
        y_pred[ind] = pred.detach().squeeze().cpu().numpy() * 100.0
    
    return y, y_pred


# def test_K(model, version, dataset_df_v2, test_begin):
    
#     test_series = dataset_df_v2.iloc[test_begin:,:]

#     targets = test_series.values[:,0]
#     vals = test_series.values[:,1::3]
#     dists = test_series.values[:,2::3]
#     bearings = test_series.values[:,3::3]

#     K = vals.shape[1]

#     if version == 'v1':
#         inp = vals * 1e7 / (dists**2)
#         nfeatures = K
#     else:
#         inp = test_series.values[:,1:]
#         nfeatures = 3*K
    
#     y = targets * 100
#     y_pred = np.empty_like(y) * np.nan
    
#     lstm_state = model.init_lstms(device=device, batch=1)
        
#     for ind in range(test_series.shape[0]):
        
#         if test_series.iloc[ind,:].isnull().any():
#             # reset LSTM whenever there is a discontinuity
#             lstm_state = model.init_lstms(device=device, batch=1)
#             continue
        
#         point_seq = np.empty((1, nfeatures, 1))
#         point_seq[0, :, 0] = inp[ind,:]
#         point_label = np.ones((1, 1)) * test_series.iloc[ind,0]
        
#         point_seq = np.transpose(point_seq, [2, 0, 1])
#         point_seq = Variable(torch.from_numpy(point_seq), requires_grad=True).to(device).double()
#         point_label = torch.from_numpy(point_label).unsqueeze(2).to(device)
        
#         pred, lstm_state = model(point_seq, lstm_state)
        
#         y_pred[ind] = pred.detach().squeeze().cpu().numpy() * 100.0
    
#     return y, y_pred
    


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Spatio-temporal LSTM for air quality prediction (testing only)')
    parser.add_argument('maxneighbors', type=int, choices=range(1,11), help='Max number of nearest neighbors to use')
    parser.add_argument('knn_version', choices=('v1', 'v2'), help='Version 1 or 2')
    parser.add_argument('--fieldeggid', help='Sensor at which location the test performance should be reported')
    parser.add_argument('--data-version', default='2019_Feb_05', dest='datesuffix', help='Version of raw data to use')
    parser.add_argument('--sensor', choices=('pm25', 'pm10'), default='pm25', help='Type of sensory data')
    parser.add_argument('--history', type=int, default=32, dest='histlen',
                        help='Length of history that was used in training (important to ensure correct train-test split)')
    parser.add_argument('--stride', type=int, default=1, help='Length of stride through data')
    parser.add_argument('--hidden', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--split', type=frac_type, default=0.8, help='Train-test split fraction')
    args = parser.parse_args()
    
    prettyprint_args(args)
    
    # use cuda if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # list of fieldeggids
    if args.fieldeggid == None:
        datapath = 'datasets/knn_{}_{}/'.format(args.knn_version, args.datesuffix)
        FIELDEGGIDS = set((fname.split('_')[2] for fname in os.listdir(datapath)))
        FIELDEGGIDS = sorted(FIELDEGGIDS)
    else:
        FIELDEGGIDS = [args.fieldeggid]
    
    # load all models for all values of K
    models = []
    
    # load all the models
    sys.stdout.write('Loading all the models ... ')
    for K in range(1, args.maxneighbors+1):
        numsegments_dict = {'v1':K+1, 'v2':3*K+1}
        model = Series(batchsize=1,
                       historylen=1,
                       numsegments=numsegments_dict[args.knn_version],
                       hiddensize=args.hidden).to(device).double()
        
        # load saved model parameters
        savepath = 'models/model_K{:02d}_{}_{}_{}.pth'.format(K, args.knn_version, args.datesuffix, args.sensor)
        
        if not os.path.exists(savepath):
            print('No saved model available for K = {}! Please run training script for that with these parameters first.'.format(K), file=sys.stderr)
            raise sys.exit(-1)
        
        model.load_state_dict(torch.load(savepath))
        model.eval()
        models.append(model)
    print('done.')


    print('Beginning testing ...')

    # store the errors for table
    errorvals = []
    
    for count, fieldeggid in enumerate(FIELDEGGIDS, 1):

        errors = []
        
        print(count, fieldeggid)

        errors.append(fieldeggid)

        sys.stdout.write('Computing test period ... ')
        test_begin_max = 0
        for K in range(1, args.maxneighbors+1):
            # load test data. Hist len is important, since the train-test
            # split depends on it
            df, _, testrefs = create_dataset_knn_sensor(K, fieldeggid, args.knn_version, args.datesuffix, args.sensor, args.stride, args.histlen, args.split)
            
            test_begin = testrefs[0][1]
            if test_begin > test_begin_max:
                test_begin_max = test_begin

        print('done.')

        errors.extend([df.index[test_begin_max], df.index[-1]])
        
        y, y_pred = test_K(model, df, test_begin_max)

        errors.append(np.nanmean(y))
        
        # compute errors and plot the real vs predicted
        rmse = np.sqrt(np.nanmean((y - y_pred)**2))
        mape = np.mean(np.fabs(np.ma.masked_invalid(y) - np.ma.masked_invalid(y_pred)) / np.ma.masked_equal(y, 0)) * 100.0

        errors.extend([rmse, mape])
        
        # x_pm = df_v2.values[test_begin:, 1::3] * 100.0
        
        fig = plt.figure(figsize=(14,5))
        ax = fig.add_subplot(111)
        fig.suptitle('Sensor: {}, K: {}, RMSE: {:.2f}, MAPE: {:.2f}'.format(fieldeggid, args.maxneighbors, rmse, mape))
        ax.set_title('Test start: {}, test end: {}'.format(df.index[test_begin_max], df.index[-1]))
        # for k in range(K):
        #     ax.plot(x_pm[:, k], color='#AAAAAA')
        ax.plot(y, label='Real')
        ax.plot(y_pred, label='Predicted')
        ax.set_xlabel('Time (test period)')
        ax.set_ylabel('PM2.5 conc')
        ax.legend()
        # fig.tight_layout()
        # plt.show()
        savedf = pd.DataFrame(data=np.hstack((y[:,None], y_pred[:,None])),
                              index=df.index[test_begin_max:],
                              columns=['Actual','Predicted'])
        savedf.to_csv('plots/test_{}_K{:02d}_{}_{}_{}.txt'.format(fieldeggid, args.maxneighbors, args.knn_version, args.datesuffix, args.sensor))
        fig.savefig('plots/test_{}_K{:02d}_{}_{}_{}.pdf'.format(fieldeggid, args.maxneighbors, args.knn_version, args.datesuffix, args.sensor))
        fig.savefig('plots/test_{}_K{:02d}_{}_{}_{}.png'.format(fieldeggid, args.maxneighbors, args.knn_version, args.datesuffix, args.sensor))
        plt.close(fig)
    
        # test by combining all the models
        # df_v2, _, _ = create_dataset_knn_sensor(K, fieldeggid, 'v2', args.datesuffix, args.sensor, args.stride, args.histlen, args.split)
        
        y, y_pred = test(models, args.knn_version, df, test_begin_max)
        
        # compute errors and plot the real vs predicted
        rmse = np.sqrt(np.nanmean((y - y_pred)**2))
        mape = np.mean(np.fabs(np.ma.masked_invalid(y) - np.ma.masked_invalid(y_pred)) / np.ma.masked_equal(y, 0)) * 100.0

        errors.extend([rmse, mape])
        
        # x_pm = df_v2.values[test_begin_max:, 1::3] * 100.0
        
        fig = plt.figure(figsize=(14,5))
        ax = fig.add_subplot(111)
        fig.suptitle('Sensor: {}, max K: {}, RMSE: {:.2f}, MAPE: {:.2f}'.format(fieldeggid, args.maxneighbors, rmse, mape))
        ax.set_title('Test start: {}, test end: {}'.format(df.index[test_begin_max], df.index[-1]))
        ax.plot(y, label='Real')
        ax.plot(y_pred, label='Predicted')
        ax.set_xlabel('Time (test period)')
        ax.set_ylabel('PM2.5 conc')
        ax.legend()

        savedf = pd.DataFrame(data=np.hstack((y[:,None], y_pred[:,None])),
                              index=df.index[test_begin_max:],
                              columns=['Actual','Predicted'])
        savedf.to_csv('plots/test_{}_maxK{:02d}_{}_{}_{}.txt'.format(fieldeggid, args.maxneighbors, args.knn_version, args.datesuffix, args.sensor))
        fig.savefig('plots/test_{}_maxK{:02d}_{}_{}_{}.pdf'.format(fieldeggid, args.maxneighbors, args.knn_version, args.datesuffix, args.sensor))
        fig.savefig('plots/test_{}_maxK{:02d}_{}_{}_{}.png'.format(fieldeggid, args.maxneighbors, args.knn_version, args.datesuffix, args.sensor))
        plt.close(fig)

        errorvals.append(errors)

    # save the error table
    if args.fieldeggid == None:
        with open('results/errors_maxK{:02d}_{}_{}_{}.csv'.format(args.maxneighbors, args.knn_version, args.datesuffix, args.sensor), 'w') as fout:
            fout.write('Field egg ID,test start,test end,mean,single RMSE,single MAPE,combined RMSE,combined MAPE\n')
            for errors in errorvals:
                fout.write(','.join(errors[0:3] + ['{:.4f}'.format(err) for err in errors[3:]]) + '\n')
