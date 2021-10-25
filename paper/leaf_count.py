import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import os
from matplotlib.ticker import MaxNLocator

from sklearn.metrics import r2_score
from scipy.interpolate import interp1d

from openalea.maizetrack.stem_correction import savgol_smoothing_function

from openalea.maizetrack.stem_correction import stem_height_smoothing


def date_to_timestamp(date):
    return time.mktime(datetime.datetime.strptime(date[:10], "%Y-%m-%d").timetuple())

#from llorenc, currently copied to  modulor cache
df_tt = pd.read_csv('TT_ZA17.csv')
f_tt = interp1d(df_tt['timestamp'], df_tt['ThermalTime'])

#from llorenc, currently copied to  modulor cache
df_pheno = pd.read_csv('pheno_ZA17.csv')
df_pheno['timestamp'] = df_pheno['resultdate'].map(date_to_timestamp)

# available in modulor local_benoit
folder = 'data/tracking/'
folder = 'data/tracking/results_tracking_360/'
#folder = 'data/tracking/gxe/'
plantids = [int(f.split('.')[0]) for f in os.listdir(folder) if os.path.isfile(folder + f)]
obs_visi, pred_visi, obs_ligul, pred_ligul, pred_visi_smooth, pred_ligul_smooth = [], [], [], [], [], []
pred_visi_smooth2 = []
pred_ligul2 = []

res_emergence = []

for plantid in plantids:

    print(plantid)

    # ===== data ===================================================================================

    dfid = pd.read_csv(folder + '{}.csv'.format(plantid))
    dfid_mature = dfid[dfid['mature']]

    # ===========================================================================================
    # ==== VISIBLE STAGE ========================================================================
    # ===========================================================================================

    # ===== brut visible stage ==================================================================

    #select = df_phenoid[(df_phenoid['observationcode'] == 'panicule_deployee') & (df_phenoid['observation'] == 1)]
    df_phenoid = df_pheno[df_pheno['Pot'] == plantid]
    select = df_phenoid[(df_phenoid['observationcode'] == 'panicule_visi') & (df_phenoid['observation'] == 1)]
    t_max = max(dfid['timestamp'])
    if not select.empty:
        t_max = min(select['timestamp'])

    timestamps = [t for t in dfid['timestamp'].unique() if t < t_max]
    df_visi = []
    for t in timestamps:

        dfidt = dfid[dfid['timestamp'] == t]

        mature_ranks = dfidt[dfidt['mature']]['rank_tracking']
        if mature_ranks.empty:
            n_mature = 0
        else:
            n_mature = max(mature_ranks)
        n_growing = len(dfidt[dfidt['mature'] == False])

        df_visi.append([t, n_mature + n_growing])
    df_visi = pd.DataFrame(df_visi, columns=['t', 'n'])

    # ===== visible emergence timing (06/10 idee Christian) ===========================================

    median_visi = df_visi.groupby('n').median('t').reset_index()

    # add missing n values (interpolation)
    for n in range(min(median_visi['n']) + 1, max(median_visi['n'])):
        if n not in list(median_visi['n']):
            t1 = median_visi[median_visi['n'] < n].sort_values('n').iloc[-1]['t']
            t2 = median_visi[median_visi['n'] > n].sort_values('n').iloc[0]['t']
            median_visi = median_visi.append({'n': n, 't': (t1 + t2) / 2}, ignore_index=True)

    # compute timing of leaf appearance (except for first and last n)
    emerg_visi = []
    for n in sorted(list(median_visi['n'].unique()))[1:-1]:
        t = np.mean(median_visi[median_visi['n'].isin([n - 1, n])]['t'])
        emerg_visi.append([plantid, t, n, 'visi'])
    emerg_visi = pd.DataFrame(emerg_visi, columns=['plantid', 't', 'n', 'type'])

    # force monotony : t = f(n) can only increase
    emerg_visi = emerg_visi.sort_values('n')
    for i_row, row in emerg_visi.iterrows():
        previous = emerg_visi[emerg_visi['n'] < row['n']]
        if not previous.empty:
            emerg_visi.iloc[i_row, emerg_visi.columns.get_loc('t')] = max(row['t'], max(previous['t']))

    res_emergence.append(emerg_visi)

    # ===========================================================================================
    # ==== LIGULATED STAGE ======================================================================
    # ===========================================================================================

    # ===== profile height used for ligulated stage =================================================

    ranks = sorted([r for r in dfid_mature['rank_tracking'].unique() if r != 0])
    height_profile = {}
    for r in ranks:
        dfr = dfid_mature[dfid_mature['rank_tracking'] == r]
        dfr = dfr.sort_values('timestamp')[:15]
        height_profile[r] = np.median(dfr['h']) + 720

    # ===== stem height used for ligulated stage =====================================================

    dfs = []
    folder_dl = 'local_cache/cache_ZA17/collars_voxel4_tol1_notop_vis4_minpix100'
    #df_tt = pd.read_csv('TT_ZA17.csv')
    all_files = [folder_dl + '/' + rep + '/' + f for rep in os.listdir(folder_dl) for f in os.listdir(folder_dl + '/' + rep)]
    plantid_files = [f for f in all_files if int(f.split('/')[-1][:4]) == plantid]
    for f in plantid_files:
        daydate = f.split('/')[-2]
        timestamp = df_tt[df_tt['daydate'] == daydate].iloc[0]['timestamp']
        dft = pd.read_csv(f)
        dft['t'] = timestamp
        dfs.append(dft)

    if dfs == []:
        print('no collar data {}'.format(plantid))
    else:

        df_collars = pd.concat(dfs)

        df_stem = df_collars.sort_values('z_phenomenal', ascending=False).drop_duplicates(['t']).sort_values('t')

        f_smoothing = stem_height_smoothing(np.array(df_stem['t']), np.array(df_stem['z_phenomenal']))
        df_stem['z_smooth'] = [f_smoothing(t) for t in df_stem['t']]

        # ===== ligulated emergence timing =============================================================================

        T = np.linspace(min(df_stem['t']), max(df_stem['t']), 3000)
        Z = np.array([f_smoothing(ti) for ti in T])

        emerg_ligu = []
        ranks = sorted(height_profile.keys())
        for r in ranks[1:]:
            t = T[np.argmin(np.abs(Z - height_profile[r - 1]))]
            emerg_ligu.append([plantid, t, r, 'ligu'])
        emerg_ligu = pd.DataFrame(emerg_ligu, columns=['plantid', 't', 'n', 'type'])

        res_emergence.append(emerg_ligu)

    # ===================================================================================================
    # ===================================================================================================
    # ===================================================================================================

    # ===== plot example for one plant ==================================================================

    #if plantid == 461:
    if False:

        # VISI

        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        ax.tick_params(axis='both', which='major', labelsize=20)  # axis number size
        plt.title(plantid)
        plt.xlabel('Thermal time $(day_{20°C})$', fontsize=30)
        plt.ylabel('Visible leaf stage', fontsize=30)

        # pred
        plt.plot([f_tt(t) for t in df_visi['t']], df_visi['n'], 'k.', markersize=15, label='raw prediction')
        plt.plot([f_tt(t) for t in emerg_visi['t']], emerg_visi['n'], 'r.-', markersize=15, label='prediction after smoothing')

        # anot
        # df_phenoid = df_pheno[df_pheno['Pot'] == plantid]
        # anot = df_phenoid[df_phenoid['observationcode'] == 'visi_f']
        # plt.plot([f_tt(t) for t in anot['timestamp']], anot['observation'], 'g^', markersize=7, label='observation (ground-truth)')

        plt.legend(prop={'size': 20}, loc='lower right', markerscale=2)


    if False:

        # LIGU

        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        ax.tick_params(axis='both', which='major', labelsize=20)  # axis number size
        plt.title(plantid)
        plt.xlabel('Thermal time $(day_{20°C})$', fontsize=30)
        plt.ylabel('Ligulated leaf stage', fontsize=30)

        # pred
        plt.plot([f_tt(t) for t in emerg_ligu['t']], emerg_ligu['n'], 'k.', markersize=15, label='leaf emergence')
        plt.step([f_tt(t) for t in emerg_ligu['t']], emerg_ligu['n'], 'r-', where='post', label='prediction (discrete interpolation)')
        # plt.plot([f_tt(t) for t in emerg_ligu['t']], emerg_ligu['n'], 'b-', label='linear interpolation')

        # anot
        df_phenoid = df_pheno[df_pheno['Pot'] == plantid]
        anot = df_phenoid[df_phenoid['observationcode'] == 'ligul_f']
        plt.plot([f_tt(t) for t in anot['timestamp']], anot['observation'], 'g^', markersize=7, label='observation (ground-truth)')

        plt.legend(prop={'size': 20}, loc='lower right', markerscale=2)


    # ======================================================================================================

res_emergence = pd.concat(res_emergence)


# ===== from emergence timing to leaf stage + anot =========================================================

res_stage = []

for plantid in res_emergence['plantid'].unique():
    for var in res_emergence['type'].unique():

        df_phenoid = df_pheno[df_pheno['Pot'] == plantid]

        selec = res_emergence[(res_emergence['plantid'] == plantid) & (res_emergence['type'] == var)]

        if selec.empty:
            print('no data {} _ {}'.format(plantid, var))
        if not selec.empty:

            if var == 'visi':
                anot = df_phenoid[df_phenoid['observationcode'] == 'visi_f']
                f = interp1d(selec['t'], selec['n'])
            elif var == 'ligu':
                anot = df_phenoid[df_phenoid['observationcode'] == 'ligul_f']
                f = interp1d(selec['t'], selec['n'], kind='previous')

            f_stage = lambda t: f(t) if min(selec['t']) < t < max(selec['t']) else None

            for _, row in anot.iterrows():
                obs = row['observation']
                pred = f_stage(row['timestamp'])
                #t_nearest = min(leaf_count.keys(), key=lambda x: abs(x - row['timestamp']))
                res_stage.append([plantid, obs, pred, var])

res_stage = pd.DataFrame(res_stage, columns=['plantid', 'obs', 'pred', 'type'])

# ===== plot visible leaf stage ===========================================================

s = res_stage[(res_stage['type'] == 'visi') & (~res_stage['pred'].isna())]
x, y = np.array(s['obs']), np.array(s['pred'])
#x = np.ceil(x - 1)
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
ax.tick_params(axis='both', which='major', labelsize=20) # axis number size
plt.gca().set_aspect('equal', adjustable='box') # same x y scale
#ax.tick_params(axis='both', which='major', labelsize=20)
ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # int axis
plt.title('Visible leaf stage', fontsize=35)
plt.xlabel('observation', fontsize=30)
plt.ylabel('prediction', fontsize=30)
plt.plot([-5, 25], [-5, 25], '-', color='black')
plt.xlim((0, 18))
plt.ylim((0, 18))
plt.plot(x, y, 'k.', markersize=15, alpha=0.05)

# a0, a, b = np.polyfit(x, y, 2)
# xu = np.array(sorted(np.unique(x)))
# plt.plot(xu, a0 * np.array(xu**2) + a * np.array(xu) + b, 'r-')
x = x[[i for i in range(len(y)) if y[i] is not None]]
y = y[[i for i in range(len(y)) if y[i] is not None]]
y = np.array([float(yi) for yi in y])
a, b = np.polyfit(x, y, 1)
plt.plot([-10, 30], a*np.array([-10, 30]) + b, '--', color='black', label='linear regression \n (a = {}, b = {})'.format(round(a, 2), round(b, 3)))
# rmax = 10
# x2, y2 = x[np.where(x <= rmax)], y[np.where(x <= rmax)]
# a2, b2 = np.polyfit(x2, y2, 1)
# plt.plot(x2, a2*x2 + b2, 'g-', label='linear regression (observation < {}) \n (a = {}, b = {})'.format(rmax, round(a2, 3), round(b2, 2)))
leg = plt.legend(prop={'size': 17}, loc='upper left')
leg.get_frame().set_linewidth(0.0)

# set the linewidth of each legend object
# for legobj in leg.legendHandles:
#     legobj.set_linewidth(3)

rmse = np.sqrt(np.sum((x - y) ** 2) / len(x))
r2 = r2_score(x, y)
biais = np.mean(y - x)
ax.text(0.60, 0.22, 'n = {} \nBias = {}\nRMSE = {} \nR² = {}'.format(len(x), round(biais, 2), round(rmse, 2), round(r2, 3)), transform=ax.transAxes, fontsize=25,
        verticalalignment='top')

fig.savefig('paper/results/nvisi_v2', bbox_inches='tight')

# ===== plot ligulated leaf stage ============================================================

s = res_stage[(res_stage['type'] == 'ligu') & (~res_stage['pred'].isna())]
x, y = np.array(s['obs']), np.array(s['pred'])
y = np.array([float(k) for k in y])
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
ax.tick_params(axis='both', which='major', labelsize=20) # axis number size
plt.gca().set_aspect('equal', adjustable='box') # same x y scale
#ax.tick_params(axis='both', which='major', labelsize=20)
ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # int axis
plt.title('Ligulated leaf stage', fontsize=35)
plt.xlabel('observation', fontsize=30)
plt.ylabel('prediction', fontsize=30)
plt.plot([-5, 25], [-5, 25], '-', color='black')
plt.xlim((0, 18))
plt.ylim((0, 18))
plt.plot(x, y, 'k.', markersize=15, alpha=0.05)

a, b = np.polyfit(x, y, 1)
plt.plot([-10, 30], a*np.array([-10, 30]) + b, 'k--', label='linear regression \n (a = {}, b = {})'.format(round(a, 2), round(b, 2)))

leg = plt.legend(prop={'size': 17}, loc='upper left', markerscale=2)
leg.get_frame().set_linewidth(0.0)
# set the linewidth of each legend object
# for legobj in leg.legendHandles:
#     legobj.set_linewidth(3)

rmse = np.sqrt(np.sum((x - y) ** 2) / len(x))
r2 = r2_score(x, y)
biais = np.mean(y - x)
ax.text(0.60, 0.22, 'n = {} \nBias = {}\nRMSE = {} \nR² = {}'.format(len(x), round(biais, 3), round(rmse, 3), round(r2, 3)), transform=ax.transAxes, fontsize=25,
        verticalalignment='top')

fig.savefig('paper/results/nligu_v2', bbox_inches='tight')


