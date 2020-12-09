
# some ideas on computing baseline

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import interpolate


def get_extrema(splinefunc):
    """WORKS ONLY FOR QUADRATIC SPLINES!! 

    This function can actually be made to work for any spline, but
    right now only quadratic is supported. There is no way to check if
    the provided spline function is a quadratic one, hence it is
    imperative upon the caller to make sure the arguments are correct,
    else the return value may be garbage.

    """

    # get the extrema of a quadratic spline (= zeros of the
    # derivative linear spline)
    deriv = splinefunc.derivative()
    knots_x = splinefunc.get_knots()
    knots_y = deriv.get_coeffs()
    extrema_list = []

    # (1) first check, is the deriv zero at any of the knots?
    extrema_list.extend(knots_x[knots_y==0])

    # (2) in a quadratic spline, there is at most one local extremum
    # between two knots, hence check for changing signs in the
    # derivative of the spline between successive pairs of knots
    extrema_intervals = np.arange(len(knots_x)-1)[np.diff(knots_y > 0)]
    for ii in extrema_intervals:
        x1, x2 = knots_x[ii], knots_x[ii+1]
        y1, y2 = deriv(x1), deriv(x2)
        x = x1 - (x2 - x1)*y1/(y2 - y1)
        extrema_list.append(x)
    
    extrema_list.sort()

    return extrema_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('sensor', choices=['pm25', 'pm10'], help='Type of sensor data')
    parser.add_argument('--mid', help='A particular monitor location (leave blank for all)')
    parser.add_argument('--odir', '-o', default='output', help='Output directory')

    megroup = parser.add_mutually_exclusive_group()
    megroup.add_argument('--week-start', '-S', default=0, type=int, choices=np.arange(7),
                         help='What day does the week start? (Mon = 0, ... Sun = 6)')
    megroup.add_argument('--weekdays', action='store_true', help='Only weekdays')
    megroup.add_argument('--weekends', action='store_true', help='Only weekends')

    parser.add_argument('--length', '-l', type=int,
                        help='# of days to average for baseline (# of weeks in case of weekdays/weekends option)')

    args = parser.parse_args()

    panel = pd.read_csv('data/combined/combined_1H_20180501_20191001_IMPUTED.csv', index_col=[0,1], parse_dates=True)

    #args.mid = 'A9BE'
    mid_list = None
    
    if args.mid is not None:
        mid_list = [args.mid]
    else:
        mid_list = panel.index.levels[0]

    # for each monitor location
    for count, mid in enumerate(mid_list, 1):

        print(os.linesep + '******** {}/{} {} ********'.format(count, len(mid_list), mid) + os.linesep)

        data = panel.loc[mid,args.sensor]

        # all days or 'weekdays' or 'weekends' 
        startday = 0
        length = None
        saveprefix = None
        if args.weekdays:

            # select only weekdays in the data
            data = data.loc[data.index.weekday < 5]

            # start on Mon
            while data.index[startday*24].weekday() != 0:
                startday += 1

            # length is a multiple of 5 (args.length is in weeks in this
            # case)
            length = 5 if args.length is None else args.length * 5

            saveprefix = 'weekday'
        elif args.weekends:

            # select only weekends in the data
            data = data.loc[data.index.weekday >= 5]

            # start on Sat
            while data.index[startday*24].weekday() != 5:
                startday += 1

            # length is a multiple of 2 (args.length is in weeks in this
            # case)
            length = 2 if args.length is None else args.length * 2

            saveprefix = 'weekend'
        else:
            # start the week on the given start weekday
            while data.index[startday*24].weekday() != args.week_start:
                startday += 1

            # length is in number of days 
            length = 7 if args.length is None else args.length

            saveprefix = 'weekstart{}'.format(args.week_start)

        mat = data.values.reshape(-1, 24)

        # the results will contain baseline prediction error and some
        # other stats for each day of prediction
        results_df = pd.DataFrame(index=data.index[np.arange(0, data.index.size, 24)],
                                  columns=['Spline', 'RMSE', 'MAPE', 'Knots', 'Extrema', 'First_Max'])

        savepath = os.path.join(args.odir, 'baseline', args.sensor, saveprefix, mid)
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        print('First day in data:', data.index[0], data.index[0].day_name())
        print('Start day:', startday)
        print('Starting on:', data.index[startday*24], data.index[startday*24].day_name())
        print('Averaging length (days):', length)
        print('Mode:', saveprefix)
        print('Total count:', (mat.shape[0]-startday)//length)

        for s_ii in tqdm(range(startday, mat.shape[0]-length, length)):

            # take one week's data at a time and compute average series
            submat = mat[s_ii:s_ii+length,:]

            avg = submat.mean(axis=0)

            # fit some curves to it
            x = np.arange(avg.size)
            xx = np.linspace(x[0], x[-1], 1000)

            # smoothing factor (used to pick no of knots in spline)
            smooth = 4*avg.size

            f1 = interpolate.UnivariateSpline(x, avg, k=2, s=smooth)
            #f2 = interpolate.UnivariateSpline(x, avg, k=3, s=smooth)

            # avg rmse in the entire week
            rmse = np.sqrt(np.mean((submat - avg)**2, axis=1))
            rmse1 = np.sqrt(np.mean((submat - f1(x))**2, axis=1))
            #rmse2 = np.sqrt(np.mean((submat - f2(x))**2, axis=1))

            # avg mape
            mape = np.mean(np.abs((submat - avg) / submat), axis=1)
            mape1 = np.ma.masked_invalid(np.abs((submat - f1(x)) / submat)).mean(axis=1)
            #mape2 = np.ma.masked_invalid(np.abs((submat - f2(x)) / submat)).mean(axis=1)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x, avg, 'ro', ms=5)
            ax.plot(xx, f1(xx), 'g', lw=2, label='2-deg spline, rmse={:.2f}, mape={:.0f}%'.format(rmse1.mean(), mape1.mean()*100))
            #ax.plot(xx, f2(xx), 'b', lw=2, label='3-deg spline, rmse={:.2f}, mape={:.0f}%'.format(rmse2.mean(), mape2.mean()*100))
            fig.suptitle('{}, {}, {}'.format(mid, args.sensor, data.index[s_ii*24].strftime('%Y-%m-%d')))
            ax.set_title('simple avg rmse={:.2f} mape={:.0f}%'.format(rmse.mean(), mape.mean()*100))
            ax.legend(loc=0, fontsize='small')
            fig.savefig(os.path.join(savepath, 'N{:02d}_sday{:03d}.png'.format(length, s_ii)))
            plt.close(fig)
            #plt.show()

            index_slice = results_df.index[s_ii:s_ii+length]
            results_df.loc[index_slice, 'Spline'] = 'sday{:03d}_N{:02d}'.format(s_ii, length)
            results_df.loc[index_slice, 'Knots'] = len(f1.get_knots())
            extrema = get_extrema(f1)
            results_df.loc[index_slice, 'Extrema'] = len(extrema)
            deriv2 = f1.derivative().derivative()
            results_df.loc[index_slice, 'First_Max'] = (deriv2(extrema[0]) < 0) if len(extrema) > 0 else False
            results_df.loc[index_slice, 'RMSE'] = rmse1.round(2)
            results_df.loc[index_slice, 'MAPE'] = (mape1*100).round(2)

        results_df.to_csv(os.path.join(os.path.dirname(savepath), 'results_{}_{}_N{:02d}.csv'.format(mid, args.sensor, length)))

        # also plot CDFs of RMSE and MAPE (essentially a better
        # representation of the distribution of errors over all the
        # days than a histogram or a table)
        hist_rmse, bins_rmse = np.histogram(results_df.RMSE[results_df.RMSE.notna()].values, bins=20)
        cmf_rmse = hist_rmse.cumsum()
        hist_mape, bins_mape = np.histogram(results_df.MAPE[results_df.MAPE.notna()].values, bins=20)
        cmf_mape = hist_mape.cumsum()
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot((bins_rmse[:-1] + bins_rmse[1:])/2, cmf_rmse/cmf_rmse[-1], 'b-', lw=2)
        ax1.grid()
        ax1.set_title(r'RMSE ($\mu g/m^3$)')
        ax2 = fig.add_subplot(212)
        ax2.plot((bins_mape[:-1] + bins_mape[1:])/2, cmf_mape/cmf_mape[-1], 'r-', lw=2)
        ax2.grid()
        ax2.set_title(r'MAPE (%)')
        fig.suptitle('Baseline pred errors, {}, {}'.format(mid, args.sensor))
        fig.subplots_adjust(hspace=0.3)
        fig.savefig(os.path.join(os.path.dirname(savepath), 'results_{}_{}_N{:02d}.png'.format(mid, args.sensor, length)))
        plt.close()
        
        # also plot some correlations just for the fun of it --
        # RMSE/MAPE vs no of extrema and first_max
        
