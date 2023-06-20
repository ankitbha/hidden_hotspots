import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sens = pd.read_csv('../epod-nyu-delhi-pollution/data/kaiterra/kaiterra_fieldeggid_1H_20180501_20200201_panel.csv')
import datetime
sens['hour_of_day'] = sens['timestamp_round'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S+05:30').hour)

spline = sens.groupby(['field_egg_id', 'hour_of_day']).mean()['pm25'].reset_index()
spline_avg = sens.groupby(['hour_of_day']).mean()['pm25'].reset_index()

from scipy.interpolate import CubicSpline
fields = []
times = []
pm25 = []
for i in np.unique(spline['field_egg_id']):
    s_i = spline[spline['field_egg_id']==i]
    x = s_i['hour_of_day'].values
    y = [t for t in s_i['pm25'].values]
    c1 = CubicSpline(x[:8],y[:8])
    c2 = CubicSpline(x[8:16],y[8:16])
    c3 = CubicSpline(x[16:24],y[16:24])
    ix = [k/100.0 for k in range(2400)]
    iy = list(np.concatenate((c1(ix[:800]),c2(ix[800:1600]),c3(ix[1600:2400]))))
    plt.plot(ix, iy, label=i, linestyle='--')
    fields += [i]*2400
    times += ix
    pm25 += iy
spline_df = pd.DataFrame((fields, times, pm25), columns=['field_egg_id', 'time', 'pm25'])
spline_df.to_csv('splines.csv')
plt.xlim(0,23)
plt.ylabel('PM25 ($\mu g/ m^3$)')
plt.xlabel('Hour of day')
plt.xticks(range(24))
plt.legend(loc="lower left", ncol=5, bbox_to_anchor=(0, -0.60))
plt.savefig('all_splines.png', bbox_inches='tight')
