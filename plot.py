
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import pytz
import glob
import sys
import os

from datasets import get_data
from tqdm import tqdm
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

plt.rc('ps', useafm=True)
plt.rc('pdf', use14corefonts=True)

import tilemapbase

tilemapbase.init()
tilemapbase.start_logging()


def plot_availability(sensor, res, start_date=None, end_date=None, save=False, disp=True):

    # if source in ('kaiterra', 'govdata'):
    #     data = get_data('data', source, sensor, res, start_date, end_date)
    # elif source == 'combined':
    data_kai = get_data('data', 'kaiterra', sensor, res, start_date, end_date)
    data_gov = get_data('data', 'govdata', sensor, res, start_date, end_date)
    data = pd.concat([data_kai, data_gov], axis=0, sort=False)
    # else:
    #     print('Cannot recognize source name.')
    #     return

    # display availability based on percent counts
    grouped_kai = data_kai.groupby(level=0)
    validfracs_kai = (grouped_kai.count() / grouped_kai.size())
    validfracs_kai.sort_values(ascending=True, inplace=True)

    grouped_gov = data_gov.groupby(level=0)
    validfracs_gov = (grouped_gov.count() / grouped_gov.size())
    validfracs_gov.sort_values(ascending=True, inplace=True)

    grouped = data.groupby(level=0)
    validfracs = (grouped.count() / grouped.size())
    validfracs.sort_values(ascending=True, inplace=True)

    # qt1 = np.percentile(validfracs.values, 25)
    # qt2 = np.percentile(validfracs.values, 50)
    # qt3 = np.percentile(validfracs.values, 75)

    plt.rc('font', size=40)
    plt.rc('ps', useafm=True)
    plt.rc('pdf', use14corefonts=True)
    
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    ax.plot(validfracs_kai.values, np.arange(1, validfracs_kai.size+1)/validfracs_kai.size, lw=2, label='Sensors')
    ax.plot(validfracs_gov.values, np.arange(1, validfracs_gov.size+1)/validfracs_gov.size, lw=2, label='Govt')
    ax.plot(validfracs.values, np.arange(1, validfracs.size+1)/validfracs.size, lw=2, label='All')
    # xmin, xmax = ax.get_xlim()
    # plt.vlines([qt1], xmin, xmax, colors='r', linestyles='--', lw=2, label=r'$Q_1$')
    # plt.vlines([qt2], xmin, xmax, colors='b', linestyles='--', lw=2, label=r'$Q_2$')
    # plt.vlines([qt3], xmin, xmax, colors='g', linestyles='--', lw=2, label=r'$Q_3$')
    # plt.xlim(xmin, xmax)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4, prune='both'))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5, prune='both'))
    ax.tick_params(length=20, pad=10)
    #plt.xlabel('The {} sensors in our deployment'.format(validfracs.size), labelpad=20)
    ax.set_xlabel('Fraction of data availability (%)', labelpad=20)
    #plt.ylabel('Available data fraction', labelpad=20)
    ax.legend(ncol=1, fancybox=True, fontsize='small')
    fig.tight_layout()
    #plt.savefig('data/datagaps_kaiterra_fieldeggs_2019_Feb_28_bar.eps')

    if save:
        fig.savefig('figures/avail_{}_{}_{}.eps'.format(sensor, start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')))
        #fig.savefig('figures/avail_{}_{}_{}.pdf'.format(sensor, start_date, end_date))

    if disp:
        plt.show()
    
    plt.close(fig)

    pass


def plot_hotspot_profile(fpath, start_dt=None, end_dt=None, save=False, disp=True):

    table = pd.read_csv(fpath, index_col=[0], parse_dates=True)

    # start and end dates for data
    if start_dt is not None:
        start_dt = start_dt.strftime('%Y%m%d')

    if end_dt is not None:
        end_dt = end_dt.strftime('%Y%m%d')

    subdir = 'profiles_{}_{}'.format(start_dt, end_dt)
    savedir = os.path.join(os.path.dirname(fpath), subdir)
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    table = table.loc[start_dt:end_dt,:]

    X = np.arange(1, 15)
    hotspot_keys = ['001', '009', '010', '011', '019', '090', '091', '099', '100', '110', '190', '900', '910', '990']
    hotspot_names = ['jlow', 'jhigh', 'slow', 'slow-jlow', 'slow-jhigh', 'shigh', 'shigh-jlow', 'shigh-jhigh',
                     'tlow', 'tlow-slow', 'tlow-shigh', 'thigh', 'thigh-slow', 'thigh-shigh']
    totalcounts = pd.DataFrame(index=hotspot_keys, columns=table.columns, data=0)
    totalcounts.insert(0, 'hotspot_name', hotspot_names)

    plt.rc('font', size=16)

    for mid in tqdm(table.columns):
        for code in table[mid]:
            if not np.isnan(code):
                t = int(code // 100)
                j = int(code % 10)
                if t & j:
                    code -= j
                totalcounts.loc['{:03.0f}'.format(code),mid] += 1
        total = totalcounts[mid].sum()
        fig = plt.figure(figsize=(8,3))
        ax = fig.add_subplot(111)
        fig.suptitle('Hotspot profile at {} - Total {}'.format(mid, total))
        (totalcounts[mid] / total).plot(kind='bar', rot=30)
        ax.grid(axis='y')
        ax.set_ylim(0, 1.1)
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        fig.savefig(os.path.join(savedir, mid + '_profile_{:04d}.png'.format(total)))

        if disp:
            plt.show()
        plt.close(fig)

    totalcounts.index.name = 'hotspot_type'
    totalcounts.to_csv(os.path.join(savedir, 'totalcounts.csv'))

    return


def plot_locs(tile=tilemapbase.tiles.Stamen_Toner_Background, labels=False, save=False, disp=True):
    
    # get locations info and initialize map parameters
    locs_kai = pd.read_csv('data/kaiterra/kaiterra_locations.csv', index_col=[0])
    locs_kai['Type'] = 'Kaiterra'
    locs_gov = pd.read_csv('data/govdata/govdata_locations.csv', index_col=[0])
    locs_gov['Type'] = 'Govt'
    locs = pd.merge(locs_kai, locs_gov, how='outer', on=['Monitor ID', 'Latitude', 'Longitude', 'Location', 'Type'], copy=False)
    lon_min, lat_min = locs.Longitude.min(), locs.Latitude.min()
    lon_max, lat_max = locs.Longitude.max(), locs.Latitude.max()
    lon_center, lat_center = locs.Longitude.mean(), locs.Latitude.mean()
    lat_pad = 1.1 * max(lat_center - lat_min, lat_max - lat_center)
    lon_pad = 1.1 * max(lon_center - lon_min, lon_max - lon_center)
    extent = tilemapbase.Extent.from_lonlat(lon_center - lon_pad,
                                            lon_center + lon_pad, 
                                            lat_center - lat_pad,
                                            lat_center + lat_pad)
    extent_3857 = extent.to_project_3857()
    color_dict = {'Kaiterra' : 'r', 'Govt' : 'b'}

    plt.rc('font', size=20)

    fig, ax = plt.subplots(figsize=(12,12), dpi=300)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plotter = tilemapbase.Plotter(extent, tile, width=600)
    #plotter = tilemapbase.Plotter(extent, tile, zoom=13)
    plotter.plot(ax, tile)

    #X_kai, Y_kai = zip(*map(tilemapbase.project, locs.Longitude[locs['Type']=='Kaiterra'], locs.Latitude[locs['Type']=='Kaiterra']))
    #ax.scatter(X_kai, Y_kai, marker='.', color='r', s=400, label='Our sensors')

    #X_gov, Y_gov = zip(*map(tilemapbase.project, locs.Longitude[locs['Type']=='Govt'], locs.Latitude[locs['Type']=='Govt']))
    #ax.scatter(X_gov, Y_gov, marker='.', color='b', s=400, label='CPCB/DPCC/IMD')

    for row in locs.itertuples():
        x, y = tilemapbase.mapping.project(row.Longitude, row.Latitude)
        if row.Type == 'Kaiterra':
            obj1 = ax.scatter(x, y, marker='.', color='r', s=200, label='Our sensors')
        else:
            obj2 = ax.scatter(x, y, marker='.', color='b', s=200, label='CPCB/DPCC/IMD')

        if labels:
            ax.text(x, y, row.Index, fontsize=3)

    # custom locations of hotspots
    hotspots_list = []
    #hotspots_list.append((28.66541706460955, 77.23223216488675, 'IGDTUW'))
    #hotspots_list.append((28.628056, 77.246667, 'ITO Crossing'))

    for lon, lat, name in hotspots_list:
        x, y = tilemapbase.project(lon, lat)
        ax.scatter(x, y, marker='*', color='g', s=200, label=name)
        ax.text(x, y, name, fontsize=8)

    ax.legend((obj1, obj2), (obj1.get_label(), obj2.get_label()), loc='lower right', ncol=2)
    #ax.legend(loc='lower right', ncol=2)

    if save:
        if labels:
            fig.savefig('figures/locs_map_labels.pdf', dpi=300)
        else:
            fig.savefig('figures/locs_map.pdf', dpi=300)
    if disp:
        plt.show()

    plt.close(fig)

    return

def plot_values_map(source, sensor, res, start_date=None, end_date=None, save=False, disp=True):
    """Show air quality values on a map, as a motivation for detailed hotspot study. """

    # get sensor data
    if start_date is None:
        start_date = '20180501'
    if end_date is None:
        end_date = '20200201'

    data = None
    if source == 'kaiterra':
        data = pd.read_csv('data/kaiterra/kaiterra_fieldeggid_{}_{}_{}_panel.csv'.format(res, start_date, end_date), 
                           index_col=[0,1], parse_dates=True)
    elif source == 'govdata':
        data = pd.read_csv('data/govdata/govdata_{}_{}_{}.csv'.format(res, start_date, end_date), 
                           index_col=[0,1], parse_dates=True)
    elif source == 'combined':
        data_kai = pd.read_csv('data/kaiterra/kaiterra_fieldeggid_{}_{}_{}_panel.csv'.format(res, start_date, end_date), 
                               index_col=[0,1], parse_dates=True)
        data_gov = pd.read_csv('data/govdata/govdata_{}_{}_{}.csv'.format(res, start_date, end_date), 
                               index_col=[0,1], parse_dates=True)
        data = pd.concat([data_kai, data_gov], axis=0, sort=False)
    else:
        print('Cannot recognize source name.')
        return

    # get locations info and initialize map parameters
    locs_kai = pd.read_csv('data/kaiterra/kaiterra_locations.csv', index_col=[0])
    locs_kai['Type'] = 'Kaiterra'
    locs_gov = pd.read_csv('data/govdata/govdata_locations.csv', index_col=[0])
    locs_gov['Type'] = 'Govt'
    locs = pd.merge(locs_kai, locs_gov, how='outer', on=['Monitor ID', 'Latitude', 'Longitude', 'Location', 'Type'], copy=False)
    lat_lims = locs.Latitude.min(), locs.Latitude.max()
    lon_lims = locs.Longitude.min(), locs.Longitude.max()
    lon_center, lat_center = locs.Longitude.mean(), locs.Latitude.mean()
    lat_pad = 1.1 * max(lat_center - lat_lims[0], lat_lims[1] - lat_center)
    lon_pad = 1.1 * max(lon_center - lon_lims[0], lon_lims[1] - lon_center)
    extent = tilemapbase.Extent.from_lonlat(lon_center - lon_pad,
                                            lon_center + lon_pad, 
                                            lat_center - lat_pad,
                                            lat_center + lat_pad)
    extent_proj = extent.to_project_3857
    tile = tilemapbase.tiles.Stamen_Toner_Background
    color_dict = {'Kaiterra' : 'r', 'Govt' : 'b'}
    
    # formula for computing marker size proportional to the pm value
    ms_min, ms_max = 10, 300
    pm_min, pm_max = data[sensor].min(), data[sensor].max()

    plt.rc('font', size=20)

    colors = ['r', 'b']
    #pm_values = np.linspace(pm_min, pm_max, 21)[1:11:2]
    pm_values = np.arange(50, pm_max/2, 100)
    
    fig = plt.figure(figsize=(6,2))
    ax = fig.add_subplot(111)

    for x, pm in enumerate(pm_values, 1):
        ms = (pm - pm_min) * (ms_max - ms_min) / (pm_max - pm_min) + ms_min
        ax.scatter(x, 1, marker='.', alpha=0.4, color=colors[x%2], s=ms**2, edgecolors='none')
    ax.set_xticks(np.arange(1, len(pm_values)+1))
    ax.set_xticklabels(['{:.0f}'.format(pm) for pm in pm_values])
    ax.set_xlim([0, len(pm_values)+1])
    ax.yaxis.set_visible(False)
    ax.tick_params(bottom=0)

    fig.subplots_adjust(left=0.01, right=0.98, bottom=0.17, top=0.99)
    if save:
        savedir = 'figures/pm_map/{}/{}/'.format(sensor, source)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        save_prefix = '{}_{}_{}_legend'.format(source, sensor, res)
        fig.savefig(savedir + save_prefix + '.pdf')
        fig.savefig(savedir + save_prefix + '.png')
    if disp:
        plt.show()
    plt.close(fig)

    # group by date
    grouped = data.groupby(level=1)

    for date, group in tqdm(grouped, total=data.index.levels[1].size, desc='Plotting on map'):
        fig, ax = plt.subplots(figsize=(12,12), dpi=100)

        if 'H' in res:
            date_fmt = date.strftime('%Y-%m-%d %H hours')
        elif res == '1D' or res == '1W':
            date_fmt = date.strftime('%Y-%m-%d')
        elif res == '1M':
            date_fmt = date.strftime('%Y-%m')
        fig.suptitle('{}, res {}'.format(date_fmt, res))
        ax.text(0.03, 0.97, r'min {:.0f} $\mu g/m^3$'.format(group[sensor].min()),
                horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
                color='r', fontweight='bold')
        ax.text(0.97, 0.97, r'max {:.0f} $\mu g/m^3$'.format(group[sensor].max()),
                horizontalalignment='right', verticalalignment='center', transform=ax.transAxes,
                color='r', fontweight='bold')
        plotter = tilemapbase.Plotter(extent, tile, width=600)
        plotter.plot(ax, tile)

        for ((mid, ts), pm) in group[sensor].iteritems():
            # size of marker should be proportional to "pm" value; we
            # make a linear relationship between pm value and marker
            # size
            x, y = tilemapbase.project(locs.loc[mid].Longitude, locs.loc[mid].Latitude)
            ms = (pm - pm_min) * (ms_max - ms_min) / (pm_max - pm_min) + ms_min
            ax.scatter(x, y, marker='.', alpha=0.4, color=color_dict[locs.loc[mid].Type], s=ms**2, edgecolors='none')

        fig.subplots_adjust(left=0.06, right=0.995, bottom=0.04, top=0.95)

        if save:
            savedir = 'figures/pm_map/{}/{}/{}/'.format(sensor, source, res)
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            save_prefix = '{}_{}_{}_{}'.format(source, sensor, res, date.strftime('%Y%m%dT%H'))
            fig.savefig(savedir + save_prefix + '.pdf')
            fig.savefig(savedir + save_prefix + '.png')
        if disp:
            plt.show()
            break
        
        plt.close(fig)

    plt.close('all')

    return

def plot_hotspots_map(fpath, save=False, disp=False):

    """ Show total count of hotspots at each location on a map. """

    import tilemapbase
    tilemapbase.start_logging()

    # hotspot parameters
    _, h_type, a, b = os.path.splitext(os.path.basename(fpath))[0].split('_')

    a_str, a_val = a[:2], int(a[2:])
    b_str, b_val = b[:2], int(b[2:])

    tres, sres = os.path.basename(os.path.dirname(fpath)).split('_')

    table = pd.read_csv(fpath, index_col=[0], parse_dates=True, dtype=np.float16)

    pass


def plot_hotspots_totalcount_tres(inpdir, source, sensor, h_type, sres, save=False, disp=False):
    """ Show total count of hotspots at each time. """

    def pick_table(tres):
        choices = glob.glob(os.path.join(inpdir, source, sensor, 'temporal', h_type, 'tres{}_sres{}'.format(tres, sres),
                                         'table_{}_*.csv'.format(h_type)))
        choices.sort()
        if len(choices) == 0:
            return None
        
        table = choices[0]
        if len(choices) > 1:
            print('I found these tables for {}.'.format(tres))
            for (ii, fpath) in enumerate(choices):
                print(ii, fpath)
            jj = 0
            while True:
                try:
                    jj = int(input('Which one do you want to plot? '))
                    if jj < 0 or jj >= len(choices):
                        raise ValueError('Input outside limits')
                    break
                except ValueError:
                    print('Invalid response. Try again.')
            table = choices[jj]

        print('Reading in this table for tres {}:'.format(tres), table)
        table_df = pd.read_csv(table, index_col=[0], parse_dates=True, dtype=np.float16)
        return table_df
    # hotspot parametersn
    # _, h_type, a, b = os.path.splitext(os.path.basename(fpath))[0].split('_')

    # a_str, a_val = a[:2], int(a[2:])
    # b_str, b_val = b[:2], int(b[2:])

    # tres, sres = os.path.basename(os.path.dirname(fpath)).split('_')

    # table = pd.read_csv(fpath, index_col=[0], parse_dates=True, dtype=np.float16)

    #title_str = '{}, {}, {}, sres {}, {} {}, {} {}'.format(source, sensor, h_type, sres, a[0], a[1], b[0], b[1])
    savename = 'hotspots_temporal_totalcount_{}_{}_{}_sres{}'.format(source, sensor, h_type, sres)

    table_1H = pick_table('1H')
    if table_1H is None:
        print('No tables found for 1H!', file=sys.stderr)
        return
    table_1D = pick_table('1D')
    if table_1D is None:
        print('No tables found for 1D!', file=sys.stderr)
        return
    table_1W = pick_table('1W')
    if table_1W is None:
        print('No tables found for 1W!', file=sys.stderr)
        return
    
    totalcount_1H = table_1H.sum(axis=1).fillna(0).astype(int)
    totalcount_1H.name = 'hotspot_count_1H'

    totalcount_1D = table_1D.sum(axis=1).fillna(0).astype(int)
    totalcount_1D.name = 'hotspot_count_1D'

    totalcount_1W = table_1W.sum(axis=1).fillna(0).astype(int)
    totalcount_1W.name = 'hotspot_count_1W'
    
    fig = plt.figure(figsize=(4,6))
    #fig.suptitle(title_str)

    ax1 = fig.add_subplot(311)
    ax1.vlines(totalcount_1H.index, 0, totalcount_1H.values, colors='b')
    ax1.plot(totalcount_1H.index[totalcount_1H.values >= 1], totalcount_1H.values[totalcount_1H.values >= 1], ls='none', marker='.', c='k')
    ax1.tick_params(bottom=0, labelbottom=0)
    ax1.grid(axis='y')
    ax1.set_title('1H time resolution')
    ax1.set_ylabel('Total count')
    
    ax2 = fig.add_subplot(312, sharex=ax1, sharey=ax1)
    ax2.vlines(totalcount_1D.index, 0, totalcount_1D.values, colors='r')
    ax2.plot(totalcount_1D.index[totalcount_1D.values >= 1], totalcount_1D.values[totalcount_1D.values >= 1], ls='none', marker='.', c='k')
    ax2.tick_params(bottom=0, labelbottom=0)
    ax2.grid(axis='y')
    ax2.set_title('1D time resolution')
    ax2.set_ylabel('Total count')

    ax3 = fig.add_subplot(313, sharex=ax1, sharey=ax1)
    ax3.vlines(totalcount_1W.index, 0, totalcount_1W.values, colors='g')
    ax3.plot(totalcount_1W.index[totalcount_1W.values >= 1], totalcount_1W.values[totalcount_1W.values >= 1], ls='none', marker='.', c='k')
    #ax3.tick_params(axis='y', right=0, left=1, length=10)
    ax3.grid(axis='y')
    ax3.set_title('1W time resolution')
    ax3.set_ylabel('Total count')
    ax3.yaxis.set_major_locator(plt.MultipleLocator(2))

    fig.canvas.draw()

    month_list = ['Jan', 'Feb', 'Mar', 'Apr',
                  'May', 'Jun', 'Jul', 'Aug',
                  'Sep', 'Oct', 'Nov', 'Dec']
    labels_cur = [txt.get_text() for txt in ax3.get_xticklabels()]
    print(labels_cur)
    labels_new = []
    for lbl in labels_cur:
        year, month = lbl.split('-')
        month = int(month)
        if month_list[month-1] == 'Jan':
            labels_new.append('Jan\n' + year)
        else:
            labels_new.append(month_list[month-1])
    ax3.set_xticklabels(labels_new)

    #ax.plot(totalcount)
    #totalcount.stem(ax=ax)
    #ax.yaxis.set_major_locator(plt.MultipleLocator())

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.06, top=0.96, hspace=0.25)

    if disp:
        plt.show()

    if save:
        fig.savefig('figures/' + savename + '.png')
        fig.savefig('figures/' + savename + '.eps')
        fig.savefig('figures/' + savename + '.pdf')
        totalcount_1H.sort_values(ascending=False).to_csv('figures/' + savename + '_1H.csv', header=True)
        totalcount_1D.sort_values(ascending=False).to_csv('figures/' + savename + '_1D.csv', header=True)
        totalcount_1W.sort_values(ascending=False).to_csv('figures/' + savename + '_1W.csv', header=True)

    plt.close(fig)

    return


def plot_hotspots_temporal_binary(fpath, source, sensor, save=False, disp=False):
    """ Show hotspots in a table format based on the binary hotspots table. """

    # hotspot parameters
    _, h_type, a, b = os.path.splitext(os.path.basename(fpath))[0].split('_')

    a_str, a_val = a[:2], int(a[2:])
    b_str, b_val = b[:2], int(b[2:])

    tres, sres = os.path.basename(os.path.dirname(fpath)).split('_')

    table = pd.read_csv(fpath, index_col=[0], parse_dates=True, dtype=np.float16)

    title_str = 'Hotspot temporal {}, {} {}, {} {}, {} {}'.format(h_type, tres, sres, a_str, a_val, b_str, b_val)
    savename = 'hotspots_temporal_{}_{}_{}_{}_{}_{}{}_{}{}'.format(source, sensor, h_type, tres, sres, a_str, a_val, b_str, b_val)

    plt.rc('font', size=20)
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
    parser.add_argument('--source', choices=('kaiterra', 'govdata', 'combined'))
    parser.add_argument('--sensor', choices=('pm25', 'pm10'))
    parser.add_argument('--save', '-s', action='store_true', help='Save all figures')
    parser.add_argument('--disp', '-d', action='store_true', help='Display all figures')
    parser.add_argument('arguments', nargs='*', help='Additional arguments for the plot function')
    args = parser.parse_args()

    # type of plot
    plot_types = ['locs',
                  'availability', 
                  'linear_model', 
                  'scatter_1nn', 
                  'cluster_freqbuckets', 
                  'hotspots_temporal_binary', 
                  'hotspots_temporal_totalcount', 
                  'pm_map', 
                  'hotspot_profile']

    if not args.plot in plot_types:
        print('Specify a plot type from the following.', file=sys.stderr)
        print(plot_types)
        sys.exit(1)
    
    save_response = args.save
    if save_response is False:
        save_response = input('Save all figures? [y] ')
        if save_response.strip().lower() == 'y':
            save_response = True
        else:
            save_response = False

    if args.plot == 'locs':
        tile = tilemapbase.tiles.build_OSM()
        labels = False
        plot_locs(tile, labels, save_response, args.disp)

    elif args.plot == 'availability':
        sensor = args.arguments[0]
        assert sensor in ('pm25', 'pm10')
        res = args.arguments[1]
        start_date = pd.Timestamp(args.arguments[2])
        end_date = pd.Timestamp(args.arguments[3])
        plot_availability(sensor, res, start_date, end_date, save=save_response, disp=args.disp)
    
    elif args.plot == 'linear_model':
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

    elif args.plot == 'hotspots_temporal_binary':

        fpath_list = glob.glob('output/hotspots/{}/{}/temporal/*/*/table_*.csv'.format(args.source, args.sensor))
        fpath_list.sort()
        
        if args.disp:
            plot_hotspots_temporal_binary(fpath_list[0], args.source, args.sensor, save=save_response, disp=args.disp)
        else:
            for count, fpath in enumerate(fpath_list, 1):
                print('{}/{} {}'.format(count, len(fpath_list), fpath))
                plot_hotspots_temporal_binary(fpath, args.source, args.sensor, save=save_response, disp=args.disp)

    elif args.plot == 'hotspots_temporal_totalcount':
        inpdir = 'output/hotspots/'
        h_type = args.arguments[0]
        sres = 0
        plot_hotspots_totalcount_tres(inpdir, args.source, args.sensor, h_type, sres, save=save_response, disp=args.disp)

    elif args.plot == 'pm_map':

        res = args.arguments[0]
        start_date, end_date = None, None

        if len(args.arguments) == 2:
            start_date = args.arguments[1]
        elif len(args.arguments) == 3:
            start_date, end_date = args.arguments[1], args.arguments[2]
        elif len(args.arguments) > 3:
            print('Only max 3 arguments allowed.')
            sys.exit(-1)

        plot_values_map(args.source, args.sensor, res, start_date, end_date, save=save_response, disp=args.disp)

    elif args.plot == 'hotspot_profile':
        fpath = args.arguments[0]

        start_dt = end_dt = None
        nn = 1
        while nn < len(args.arguments):
            opt = args.arguments[nn]
            val = pd.Timestamp(args.arguments[nn+1], tz=pytz.FixedOffset(330))
            if opt == '-S' or opt == '--start-dt':
                start_dt = val
            elif opt == '-E' or opt == '--end-dt':
                end_dt = val
            else:
                print('Option should be either \'-s/--start-dt\' or \'-e/--end-dt\'', file=sys.stderr)
                sys.exit(-1)
            nn += 2
        # mid = None
        # if len(args.arguments) == 2:
        #     mid = args.arguments[1]
        # elif len(args.arguments) > 2:
        #     print('Only two arguments allowed: path to table, and (optionally) monitor ID', file=sys.stderr)
        #     sys.exit(-1)
        plot_hotspot_profile(fpath, start_dt, end_dt, save_response, args.disp)
