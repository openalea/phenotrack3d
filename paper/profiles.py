import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score

plantids = [int(f.split('.')[0]) for f in os.listdir('data/tracking') if os.path.isfile('data/tracking/' + f)]

# available in NASHShare2
df_anot = pd.read_csv('data/rgb_leaf_annotation/all_anot.csv')  # attention 940
df_anot_h = pd.read_csv('data/rgb_insertion_profile_annotation/annotation/all_anot.csv')

# ===== combine ligulated data of all plantids in a unique df =================================

df = []
for plantid in plantids:

    # data plantid

    # available in modulor local_benoit
    dfid = pd.read_csv('data/tracking/{}.csv'.format(plantid))
    dfid_mature = dfid[dfid['mature']]

    dfid_mature = dfid_mature[dfid_mature['timestamp'] < 1496509351] # 06-03

    ranks = sorted([r for r in dfid_mature['rank_tracking'].unique() if r != 0])
    for r in ranks:
        dfr = dfid_mature[(dfid_mature['mature']) & (dfid_mature['rank_tracking'] == r)]
        dfr = dfr.sort_values('timestamp')[:15]
        df.append(dfr)

df = pd.concat(df)

# ====== profile + merge with annotation ========================================================================

res = []
for plantid in df['plantid'].unique():

    # length annotation
    df_anotid = df_anot[df_anot['plantid'] == plantid].sort_values('rank')

    # height annotation
    df_anotid_h = df_anot_h[df_anot_h['plantid'] == plantid].sort_values('height_mm')
    df_anotid_h['rank'] = np.arange(1, len(df_anotid_h) + 1)

    selec = df[df['plantid'] == plantid]
    for r in selec['rank_tracking'].unique():
        if r in list(df_anotid['rank']):
            dfr = selec[selec['rank_tracking'] == r]
            obs = df_anotid[df_anotid['rank'] == r].sort_values('length').iloc[-1]['length']
            pred = np.median(dfr['l_extended'])
            res.append([plantid, obs, pred, 'l', r])

        # height
        if r in list(df_anotid_h['rank']):
            dfr = selec[selec['rank_tracking'] == r]
            obs = df_anotid_h[df_anotid_h['rank'] == r].iloc[0]['height_mm']
            pred = np.median(dfr['h'])
            res.append([plantid, obs, pred, 'h', r])

res = pd.DataFrame(res, columns=['plantid', 'obs', 'pred', 'var', 'rank'])

# ===== plot an example for 1 plant ===============================================================================

# length
#for plantid in [1276, 1301, 940]:
plantid = 1301

fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
ax.tick_params(axis='both', which='major', labelsize=20)  # axis number size

# anot
selec = res[(res['plantid'] == plantid) & (res['var'] == 'l')]
plt.plot(selec['rank'], selec['obs'] / 10, 'g--', linewidth=3, label='observation')

# pred
selec = df[df['plantid'] == plantid]
plt.plot(selec['rank_tracking'], selec['l_extended'] / 10, 'k.', markersize=8, label='prediction')
selec = selec.groupby('rank_tracking').median('l_extended').reset_index()
plt.plot(selec['rank_tracking'], selec['l_extended'] / 10, 'k-o', markersize=10, linewidth=3, label='Final median prediction')

plt.legend()
plt.title('Leaf length profile \n example for one plant', fontsize=35)
plt.xlabel('Leaf rank', fontsize=30)
plt.ylabel('Leaf length (cm)', fontsize=30)
leg = plt.legend(prop={'size': 22}, loc='lower center', markerscale=2)
leg.get_frame().set_linewidth(0.0)

fig.savefig('paper/results/profil_example', bbox_inches='tight')

# fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
# ax.tick_params(axis='both', which='major', labelsize=20)  # axis number size
# selec = df[df['plantid'] == plantid]
# plt.plot(selec['rank_tracking'], (selec['h'] + 700) / 10, 'k.', markersize=10, label='Ligulated leaf')
# selec = selec.groupby('rank_tracking').median('l_extended').reset_index()
# plt.plot(selec['rank_tracking'], (selec['h'] + 700) / 10, 'r-*', markersize=10, label='Profile value (median)')
# plt.legend()
# plt.xlabel('Rank', fontsize=30)
# plt.ylabel('Length (cm)', fontsize=30)
# plt.legend(prop={'size': 22}, loc='lower right', markerscale=2)

# insertion height
for plantid in [439, 907]:

    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.tick_params(axis='both', which='major', labelsize=20)  # axis number size

    # anot
    selec = res[(res['plantid'] == plantid) & (res['var'] == 'h')]
    plt.plot(selec['rank'], selec['obs'] / 10, 'g-', linewidth=3, label='observation (ground-truth)')

    # pred
    selec = df[df['plantid'] == plantid]
    selec['h'] += 700
    plt.plot(selec['rank_tracking'], selec['h'] / 10, 'k.', markersize=10)
    selec = selec.groupby('rank_tracking').median('l_extended').reset_index()
    plt.plot(selec['rank_tracking'], selec['h'] / 10, 'r-', markersize=10, linewidth=3, label='prediction (median)')

    plt.legend()
    plt.title(plantid)
    plt.xlabel('Rank', fontsize=30)
    plt.ylabel('Insertion height (cm)', fontsize=30)
    plt.legend(prop={'size': 22}, loc='lower right', markerscale=2)

# ===== plot validation of profile length ============================================================================

selec = res[res['var'] == 'l']
x, y = np.array(selec['obs']) / 10, np.array(selec['pred']) / 10
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
ax.tick_params(axis='both', which='major', labelsize=20) # axis number size
plt.gca().set_aspect('equal', adjustable='box') # same x y scale
plt.title('Length profile', fontsize=35)
plt.xlabel('observation (cm)', fontsize=30)
plt.ylabel('prediction (cm)', fontsize=30)
plt.plot([-20, 300], [-20, 300], '-', color='black')
plt.xlim((0, 130))
plt.ylim((0, 130))
plt.plot(x, y, 'ro', markersize=8, alpha=0.5)

# add results without leaf extension
df_ext = pd.read_csv('paper/leaf_ext.csv').iloc[:-1]
x2, y2 = df_ext['obs'] / 10, df_ext['pred_phm'] / 10
plt.plot(x2, y2, '.', color='grey', markersize=8, alpha=0.5)

a, b = np.polyfit(x, y, 1)
plt.plot([-10, 3000], a*np.array([-10, 3000]) + b, 'r--', label=f'y = {a:.{2}f}x {b:+.{2}f}')

a, b = np.polyfit(x2, y2, 1)
plt.plot([-10, 3000], a*np.array([-10, 3000]) + b, '--', color='grey', label=f'y = {a:.{2}f}x {b:+.{2}f}')

leg = plt.legend(prop={'size': 22}, loc='upper left', markerscale=2)
leg.get_frame().set_linewidth(0.0)
leg.get_texts()[1].set_color('grey')

rmse = np.sqrt(np.sum((x - y) ** 2) / len(x))
r2 = r2_score(x, y)
biais = np.mean(y - x)
mape = 100 * np.mean(np.abs((x - y) / x))
ax.text(0.60, 0.26, 'n = {} \nBias = {} cm\nRMSE = {} cm \nMAPE = {} % \nR² = {}'.format(
    len(x), round(biais, 2), round(rmse, 2), round(mape, 1), round(r2, 3)), transform=ax.transAxes, fontsize=25,
        verticalalignment='top')
rmse2 = np.sqrt(np.sum((x2 - y2) ** 2) / len(x2))
r2_2 = r2_score(x2, y2)
bias2 = np.mean(y2 - x2)
mape2 = 100 * np.mean(np.abs((x2 - y2) / x2))
ax.text(0.05, 0.75, 'n = {} \nBias = {} cm \nRMSE = {} cm \nMAPE = {} % \nR² = {}'.format(
    len(x2), round(bias2, 2), round(rmse2, 1), round(mape2, 1), round(r2_2, 3)),
        transform=ax.transAxes, fontsize=20, verticalalignment='top', color='grey')

fig.savefig('paper/results/profil_l', bbox_inches='tight')

# ===== plot validation of profile height ============================================================================

selec = res[res['var'] == 'h']
x, y = np.array(selec['obs']) / 10, (np.array(selec['pred']) + 700) / 10
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
ax.tick_params(axis='both', which='major', labelsize=20) # axis number size
plt.gca().set_aspect('equal', adjustable='box') # same x y scale
plt.title('Insertion height profile', fontsize=35)
plt.xlabel('observation (cm)', fontsize=30)
plt.ylabel('prediction (cm)', fontsize=30)
plt.plot([-20, 300], [-20, 300], '-', color='k')
plt.xlim((-10, 210))
plt.ylim((-10, 210))
plt.plot(x, y, 'ro', markersize=8, alpha=0.5)

a, b = np.polyfit(x, y, 1)
plt.plot([-10, 3000], a*np.array([-10, 3000]) + b, 'r--', label=f'y = {a:.{2}f}x {b:+.{2}f}')

leg = plt.legend(prop={'size': 22}, loc='upper left', markerscale=2)
leg.get_frame().set_linewidth(0.0)

rmse = np.sqrt(np.sum((x - y) ** 2) / len(x))
r2 = r2_score(x, y)
biais = np.mean(y - x)
mape = 100 * np.mean(np.abs((x - y) / x))
ax.text(0.60, 0.26, 'n = {} \nBias = {} cm\nRMSE = {} cm \nMAPE = {} % \nR² = {}'.format(
    len(x), round(biais, 2), round(rmse, 2), round(mape, 1), round(r2, 3)), transform=ax.transAxes, fontsize=25,
        verticalalignment='top')

fig.savefig('paper/results/profil_h', bbox_inches='tight')



# ===== for review, juste change fig title ========================================================================

fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
ax.tick_params(axis='both', which='major', labelsize=20) # axis number size
# plt.gca().set_aspect('equal', adjustable='box') # same x y scale
plt.title('Ligulated leaf length = f(rank)\n- example for one plant', fontsize=26)


