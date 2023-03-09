import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import datetime
import os

from sklearn.metrics import r2_score


def date_to_timestamp(date):
    return time.mktime(datetime.datetime.strptime(date[:10], "%Y-%m-%d").timetuple())

# availble on NasShare2
df_anot = pd.read_csv('data/rgb_leaf_annotation/all_anot.csv')  # attention 940

# ====== leaf extension validation ==================================

# mature data, 9 plants

res = []

for plantid in [int(p) for p in df_anot['plantid'].unique()]:
    dfid = pd.read_csv('data/tracking/{}.csv'.format(plantid)) # modulor/local_benoit
    df_anotid = df_anot[df_anot['plantid'] == plantid].sort_values('rank')

    if plantid == 940:
        df_anotid = df_anotid.groupby('rank').max('timestamp').reset_index()

    for _, row in df_anotid.iterrows():
        select = dfid[(dfid['rank_annotation'] == row['rank']) &
                             (dfid['timestamp'] == row['timestamp'])]
        if not select.empty:
            select_row = select.iloc[0]
            res.append([row['length'], select_row['l'], select_row['l_extended'], 'mature'])
        else:
            print('{} 1 row with no data'.format(plantid))

# growing data, 10 plants (rank 6, rank 9)
# availble on NasShare2
df_anot2 = pd.read_csv('data/rgb_leaf_growing_annotation/leaf_growing_annotation.csv')

for plantid in [int(p) for p in df_anot2['plantid'].unique()]:
    dfid = pd.read_csv('data/tracking/{}.csv'.format(plantid))
    df_anotid = df_anot2[df_anot2['plantid'] == plantid].sort_values('rank')

    for _, row in df_anotid.iterrows():
        select = dfid[(dfid['rank_annotation'] == row['rank']) &
                             (dfid['timestamp'] == row['timestamp'])]
        if not select.empty:
            select_row = select.iloc[0]
            res.append([row['length_mm'], select_row['l'], select_row['l_extended'], 'growing'])
        else:
            print('{} 1 row with no data'.format(plantid))


res = pd.DataFrame(res, columns=['obs', 'pred_phm', 'pred_ext', 'var'])

# ===== everything in the same graph ==================================================================

# mature + growing
# x = np.concatenate((obs, obs2))
# y = np.concatenate((pred_phm, pred2_phm))
# y_ext = np.concatenate((pred_ext, pred2_ext))

# only_mature
x = res['obs'] / 10
y = res['pred_phm'] / 10
y_ext = res['pred_ext'] / 10

rmse = np.sqrt(np.sum((x - y) ** 2) / len(x))
r2 = r2_score(x, y)
bias = np.mean(x - y)
rmse_ext = np.sqrt(np.sum((x - y_ext) ** 2) / len(x))
r2_ext = r2_score(x, y_ext)
bias_ext = np.mean(x - y_ext)

fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xlim((0, 135))
plt.ylim((0, 135))
plt.plot([-10, 150], [-10, 150], '-', color='k')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('observation (cm)', fontsize=30)
plt.ylabel('prediction (cm)', fontsize=30)
plt.plot(x, y, '^', color='grey', markersize=6, label='Phenomenal') # \n RMSE = {} cm, R² = {}'.format(round(rmse, 1), round(r2, 3)))
plt.plot(x, y_ext, 'o', color='black', markersize=6, label='Phenomenal with leaf extension') # \n RMSE = {} cm, R² = {}'.format(round(rmse_ext, 1), round(r2_ext, 3)))
leg = plt.legend(prop={'size': 16}, loc='upper left', markerscale=2)
leg.get_frame().set_linewidth(0.0)
plt.title('Leaf length', fontsize=35)
#plt.subtitle('Leaf length', fontsize=35, y=0.94)


ax.text(0.3, 0.17, 'n = {} \nBias = {} cm \nRMSE = {} cm \nR² = {}'.format(len(x), round(bias_ext, 2), round(rmse_ext, 2), round(r2_ext, 3)), transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='black')
ax.text(0.67, 0.17, 'n = {} \nBias = {} cm \nRMSE = {} cm \nR² = {}'.format(len(x), round(bias, 2), round(rmse, 1), round(r2, 3)), transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='grey')

fig.savefig('paper/results/extension_validation', bbox_inches='tight')







