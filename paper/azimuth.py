import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openalea.maizetrack.phenomenal_display import PALETTE
import os

plantid = 940

# result of tracking (without 3D object) saved by trackedplant method (copied to cachezA17/local_benoit)
dfid = pd.read_csv('data/tracking/{}.csv'.format(plantid))
#dfid = dfid[~dfid['mature']]
#from llorenc, currently copied to  modulor cache
df_tt = pd.read_csv('data/TT_ZA17.csv')

thermaltimes = np.array(df_tt['ThermalTime'])
timestamps = np.array(df_tt['timestamp'])
dfid['tt'] = [thermaltimes[np.argmin(np.abs(timestamps - t))] for t in dfid['timestamp']]

fig, ax = plt.subplots(figsize =(10, 10), dpi=100, subplot_kw={'projection': 'polar'})
fig.canvas.set_window_title(str(plantid))

ranks = sorted([r for r in dfid['rank_tracking'].unique() if r != 0])
for r in ranks[:9]:
    dfr = dfid[dfid['rank_tracking'] == r].sort_values('tt').iloc[:25]
    dfr = dfr[dfr['tt'] - min(dfr['tt']) < 40]
    time = dfr['tt'] - min(dfr['tt'])
    azimuth = dfr['a'] / 360 * 2*np.pi
    ax.plot(azimuth, time, '-', color=PALETTE[r - 1] / 255.)

ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.tick_params(axis='both', which='major', labelsize=20)
plt.legend()

fig.savefig('paper/azimuth_growth', bbox_inches='tight')


fig, ax = plt.subplots(figsize =(10, 10), dpi=100)
ax.tick_params(axis='both', which='major', labelsize=20)  # axis number size
fig.canvas.set_window_title(str(plantid))
dfid2 = dfid[dfid['rank_tracking'] != 0]
profile = []
ranks = dfid2['rank_tracking'].unique()
for r in ranks:
    dfr = dfid2[dfid2['rank_tracking'] == r].sort_values('tt').iloc[:15]
    profile.append(np.median(dfr['a']))
profile[-2] -= 180
plt.plot(ranks, profile, 'k*-', markersize=10)
plt.xlabel('Rank', fontsize=30)
plt.ylabel('Azimuth (°)', fontsize=30)

fig.savefig('paper/azimuth_profile', bbox_inches='tight')

