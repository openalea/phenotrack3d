"""
Test accuracy of leaf length of growing leaves after phenomenal + tracking
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score
from matplotlib.ticker import MaxNLocator

# available on  NAsshare2
df_anot = pd.read_csv('data/rgb_leaf_growing_annotation/leaf_growing_annotation.csv')

plantids = df_anot['plantid'].unique()

res = []
n = 0
for plantid in plantids:
    # available on modulor local_benoit
    df_pred = pd.read_csv('data/tracking/{}.csv'.format(plantid))
    df_obs = df_anot[df_anot['plantid'] == plantid]

    for _, row_obs in df_obs.iterrows():
        selec = df_pred[(df_pred['rank_tracking'] == row_obs['rank']) & (df_pred['timestamp'] == row_obs['timestamp'])]
        if not selec.empty:
            n += ((selec['rank_tracking'] == selec['rank_annotation']).iloc[0])
            row_pred = selec.iloc[0]
            res.append([plantid, row_obs['rank'], row_obs['length_mm'], row_pred['l'], row_pred['l_extended']])

res = pd.DataFrame(res, columns=['plantid', 'rank', 'obs', 'pred_phm', 'pred_ext'])


for rank in [6, 9]:
    selec = res[res['rank'] == rank]
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.tick_params(axis='both', which='major', labelsize=20) # axis number size
    plt.gca().set_aspect('equal', adjustable='box') # same x y scale
    plt.title('Length of leaf {} \nduring growing phase'.format(rank), fontsize=35)
    plt.xlabel('observation (cm)', fontsize=30)
    plt.ylabel('prediction (cm)', fontsize=30)
    plt.plot([-20, 160], [-20, 160], '-', color='k')
    plt.xlim((0, 130))
    plt.ylim((0, 130))
    x, y = selec['obs'] / 10, selec['pred_ext'] / 10
    plt.plot(x, y, 'k.', markersize=15, alpha=0.3)
    # x2, y2 = selec['obs'] / 10, selec['pred_phm'] / 10
    # plt.plot(x2, y2, '^', color='grey', alpha=0.7, markersize=15)

    a, b = np.polyfit(x, y, 1)
    plt.plot([-10, 3000], a * np.array([-10, 3000]) + b, 'k--',
             label='linear regression \n (a = {}, b = {} cm)'.format(round(a, 2), round(b, 2)))

    leg = plt.legend(prop={'size': 17}, loc='upper left', markerscale=2)
    leg.get_frame().set_linewidth(0.0)

    rmse = np.sqrt(np.sum((x - y) ** 2) / len(x))
    r2 = r2_score(x, y)
    biais = np.mean(y - x)
    ax.text(0.60, 0.22, 'n = {} \nBias = {} cm \nRMSE = {} cm \nR² = {}'.format(len(x), round(biais, 2), round(rmse, 1), round(r2, 3)), transform=ax.transAxes, fontsize=25,
            verticalalignment='top')

    fig.savefig('paper/results/growing_leaf_{}'.format(rank), bbox_inches='tight')




