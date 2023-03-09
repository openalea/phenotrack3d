import pandas as pd
import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt
from openalea.maizetrack.utils import get_rgb
from openalea.maizetrack.local_cache import get_metainfos_ZA17
from openalea.maizetrack.stem_correction import smoothing_function, stem_height_smoothing
from scipy.interpolate import interp1d
import time
import datetime
from matplotlib.ticker import MaxNLocator


def date_to_timestamp(date):
    return time.mktime(datetime.datetime.strptime(date[:10], "%Y-%m-%d").timetuple())

colors = {'WW': 'dodgerblue', 'WD': 'orangered', 'error': 'gainsboro'}

# thermal time : available in modulor
df_tt = pd.read_csv('data/TT_ZA17.csv')

# collar detection data
# available in modulor
folder_dl = 'local_cache/cache_ZA17/collars_voxel4_tol1_notop_vis4_minpix100'
all_files_dl = [folder_dl + '/' + rep + '/' + f for rep in os.listdir(folder_dl) for f in os.listdir(folder_dl + '/' + rep)]

# ZA17 GxE data
# available in modulor
pheno_ZA17 = pd.read_csv('data/pheno_ZA17.csv')
pheno_ZA17['timestamp'] = pheno_ZA17['resultdate'].map(date_to_timestamp)
gxe = pheno_ZA17[['Pot', 'Variety_ID', 'Scenario', 'Rep', 'Sampling']].drop_duplicates()
gxe = gxe[gxe['Sampling'] == 'Rep']
gxe = gxe[gxe['Variety_ID'] != 'SolNu']


# ===== stem =========================================================================================================

res_stem = []
for genotype in gxe['Variety_ID'].unique():

    selec = gxe[gxe['Variety_ID'] == genotype]

    for _, row in selec.iterrows():

        plantid_files = [f for f in all_files_dl if int(f.split('/')[-1][:4]) == row['Pot']]
        if len(plantid_files) > 45:

            dfid = []
            for f in plantid_files:
                task = int(f.split('_')[-1].split('.csv')[0])
                #timestamp = next(m for m in metainfos if m.task == task).timestamp
                daydate = f.split('__')[-2]
                tt = df_tt[df_tt['daydate'] == daydate].iloc[0]['ThermalTime']
                dft = pd.read_csv(f)
                dft['tt'] = tt
                dfid.append(dft)
            dfid = pd.concat(dfid)

            # remove > 06-08
            dfid = dfid[dfid['tt'] < 90.5]

            #  remove big time gaps
            times = list(dfid.sort_values('tt')['tt'].unique())
            for i in range(1, len(times)):
                if (times[i] - times[i - 1]) > 5 * np.mean(np.diff(times)):
                    dfid = dfid[dfid['tt'] < times[i]]

            # data preprocessing, stem extraction
            dfid = dfid[dfid['score'] > 0.95]
            df_stem = dfid.sort_values('z_phenomenal', ascending=False).drop_duplicates(['tt']).sort_values('tt')
            f_smoothing = stem_height_smoothing(np.array(df_stem['tt']), np.array(df_stem['z_phenomenal']))
            df_stem['z_smooth'] = [f_smoothing(t) for t in df_stem['tt']]

            for _, row_stem in df_stem.iterrows():
                res_stem.append([row['Variety_ID'], row['Pot'], row['Scenario'], row_stem['z_smooth'], row_stem['tt']])


res_stem = pd.DataFrame(res_stem, columns=['genotype', 'plantid', 'scenario', 'z', 't'])

res_gxe = res_stem.copy()
res_gxe['z'] /= 10
#res_gxe = res_gxe[res_gxe['t'] < np.min(res_gxe.groupby(['scenario', 'genotype']).max('t')['t'])] # same tmax for each scenario
res_gxe = res_gxe.groupby(['genotype', 'scenario', 'plantid', 't']).agg(['median', 'count']).reset_index()
#res_gxe = res_gxe[res_gxe['z']['count'] > 2] # at least 3 rep
#res_gxe = res_gxe[res_gxe['t'] <= min(res_gxe.groupby('scenario').max('t')['t'])]  # same end for each scenario

h_cabin = 300
# res_gxe = res_gxe[res_gxe['z']['median'] < h_cabin] # cabin limit

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
fig.canvas.set_window_title(genotype)
ax.tick_params(axis='both', which='major', labelsize=20)  # axis number size
# for scenario in ['WW', 'WD']:
#     for genotype in res_gxe['genotype'].unique():
for plantid in res_gxe['plantid'].unique():
    selec = res_gxe[(res_gxe['plantid'] == plantid)]
    scenario = selec.iloc[0]['scenario'][0]
    plt.plot(selec['t'], selec['z']['median'], '-', color=colors[scenario], linewidth=0.2)
        #plt.plot(selec['t'], selec['z']['median'], '.', color=colors[scenario], label=scenario, markersize=12)
plt.plot([-100, 1000], [h_cabin, h_cabin], 'k--', label='cabin limit')
plt.xlabel('Thermal time $(day_{20°C})$', fontsize=25)
plt.ylabel('Stem height (cm)', fontsize=25)
plt.title('Stem growth', fontsize=35)
plt.xlim([8, 93])
plt.ylim([-2, 202])
# lgd = plt.legend(prop={'size': 20}, loc='upper left', markerscale=2, handletextpad=-0.5)
# lgd.get_frame().set_linewidth(0.0)

fig.savefig('paper/stem_360'.format(genotype), bbox_inches='tight')
# plt.close('all')

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
fig.canvas.set_window_title(genotype)
ax.tick_params(axis='both', which='major', labelsize=20)  # axis number size
dz = 4.5
selec = res_gxe[res_gxe['t'] == res_gxe['t'].unique()[24]]
print(selec.iloc[0]['t'])
for scenario in ['WW', 'WD']:
    s = selec[selec['scenario'] == scenario]
    data = s['z']['median']
    bins = int(np.ceil((data.max() - data.min()) / dz))
    plt.hist(s['z']['median'], bins=bins, color=colors[scenario], alpha=0.8)
plt.xlabel('Stem height (cm)', fontsize=30)


fig.savefig('paper/stem_hist_360'.format(genotype), bbox_inches='tight')

# ===== profile ======================================================================================================

res_profile_l = []
for genotype in gxe['Variety_ID'].unique():
    print(genotype)
    selec = gxe[gxe['Variety_ID'] == genotype]
    for _, row in selec.iterrows():

        # available in modulor local_benoit
        file = 'data/tracking/results_tracking_360/{}.csv'.format(row['Pot'])
        if os.path.isfile(file):

            dfid = pd.read_csv(file)

            dfid_mature = dfid[dfid['mature']]
            ranks = sorted([r for r in dfid_mature['rank_tracking'].unique() if r != 0])
            profile = []
            for r in ranks:
                dfr = dfid_mature[dfid_mature['rank_tracking'] == r]
                dfr = dfr.sort_values('timestamp')[:15]

                res_profile_l.append([row['Variety_ID'], row['Pot'], row['Scenario'], np.median(dfr['l_extended']), r])


res_profile_l = pd.DataFrame(res_profile_l, columns=['genotype', 'plantid', 'scenario', 'l', 'r'])
res_profile_l['l'] /= 10

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
fig.canvas.set_window_title('profile')
ax.tick_params(axis='both', which='major', labelsize=20)  # axis number size
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

for scenario in ['WW', 'WD']:
    for r in res_profile_l['r'].unique():
        s = res_profile_l[(res_profile_l['scenario'] == scenario) & (res_profile_l['r'] == r)]
        if len(s) > 10:
            dr = 0.07
            if scenario == 'WW':
                dr *= -1
            plt.errorbar(r + dr, np.median(s['l']), yerr=np.std(s['l']), marker='.', color=colors[scenario], capsize=3)

plt.xlabel('Rank', fontsize=25)
plt.ylabel('Ligulated leaf length (cm)', fontsize=25)
plt.title('Leaf length profile', fontsize=35)

fig.savefig('paper/profile_360'.format(genotype), bbox_inches='tight')


# hist

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
ax.tick_params(axis='both', which='major', labelsize=20)  # axis number size
#dz = 5
selec = res_profile_l[res_profile_l['r'] == 10]
for scenario in ['WW', 'WD']:
    s = selec[selec['scenario'] == scenario]
    data = s['l']
    #bins = int(np.ceil((data.max() - data.min()) / dz))
    bins = 30
    plt.hist(data, bins=bins, range=[0, 150], color=colors[scenario], alpha=0.8)
plt.xlabel('Leaf length (cm)', fontsize=30)

fig.savefig('paper/profile_hist_360'.format(genotype), bbox_inches='tight')





