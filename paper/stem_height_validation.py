import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openalea.maizetrack.local_cache import get_metainfos_ZA17
from openalea.maizetrack.stem_correction import stem_height_smoothing

from openalea.maizetrack.local_cache import metainfos_to_paths, check_existence, load_plant
from openalea.maizetrack.stem_correction import get_median_polyline
from openalea.maizetrack.utils import phm3d_to_px2d

import json

from sklearn.metrics import r2_score
from scipy.interpolate import interp1d

# ================================================================

# modulor
df_tt = pd.read_csv('TT_ZA17.csv')

# modulor
folder = 'local_cache/ZA17/collars_voxel4_tol1_notop_vis4_minpix100'
all_files = [folder + '/' + rep + '/' + f for rep in os.listdir(folder) for f in os.listdir(folder + '/' + rep)]

# available in NASHShare2
folder2 = 'data/stem_annotation/'
plantids_anot = [int(f.split('_')[1].split('.')[0]) for f in os.listdir(folder2) if os.path.isfile(folder2 + f)]


# ===== plot just one example ================================================================================

plantid = 783

dfs = []
plantid_files = [f for f in all_files if int(f.split('/')[-1][:4]) == plantid]
for f in plantid_files:
    task = int(f.split('_')[-1].split('.csv')[0])
    dft = pd.read_csv(f)
    daydate = f.split('/')[3]
    tt = df_tt[df_tt['daydate'] == daydate].iloc[0]['ThermalTime']
    dft['t'] = tt
    dfs.append(dft)

df = pd.concat(dfs)

df = df[df['score'] > 0.95]
df['z_phenomenal'] /= 10
#df['y'] = 2448 - df['y']
df_stem = df.sort_values('z_phenomenal', ascending=False).drop_duplicates(['t']).sort_values('t')
f_smoothing = stem_height_smoothing(np.array(df_stem['t']), np.array(df_stem['z_phenomenal']))

plt.figure(plantid)
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
ax.tick_params(axis='both', which='major', labelsize=20)  # axis number size
plt.title(plantid)
plt.xlabel('Thermal time $(day_{20°C})$', fontsize=30)
plt.ylabel('Stem height (cm)', fontsize=30)

plt.plot(df_stem['t'], df_stem['z_phenomenal'], 'k.', markersize=15)
plt.plot(df_stem['t'], [f_smoothing(t) for t in df_stem['t']], 'r-', linewidth=2, label='smoothing')

plt.legend(prop={'size': 20}, loc='lower right', markerscale=2)


# =======================================================================================================

res = []
plt.figure()
for plantid in [p for p in plantids_anot if p != 452]: # TODO : plantid 452 not in cache !

    print(plantid)
    metainfos = get_metainfos_ZA17(plantid)

    # ========== basic phenomenal ============================================

    # available in modulor
    folder_vmsi = 'local_cache/ZA17/segmentation_voxel4_tol1_notop_vis4_minpix100_no_stem_smooth_no_tracking'
    paths = metainfos_to_paths(metainfos, phm_parameters=(4, 1, 'notop', 4, 100), folder=folder_vmsi)

    metainfos2, paths = check_existence(metainfos, paths)

    vmsi_list = load_plant(metainfos2, paths)

    z_base = {sf: np.median([v.get_stem().get_highest_polyline().polyline[0][2] for v in vmsi_list
                             if v.metainfo.shooting_frame == sf]) for sf in ['elcom_2_c1_wide', 'elcom_2_c2_wide']}

    stem_phm = {}
    for v in vmsi_list:
        heights = [l.info['pm_z_base'] for l in v.get_mature_leafs()]
        if len(heights) != 0:
            stem_phm[v.metainfo.timestamp] = max(heights) - z_base[v.metainfo.shooting_frame]

    # ========= median stem on new phenomenal ====================================

    # available in modulor
    folder_vmsi = 'local_cache/ZA17/segmentation_voxel4_tol1_notop_vis4_minpix100_stem_smooth_tracking'
    paths = metainfos_to_paths(metainfos, stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), old=False,
                               folder=folder_vmsi)

    metainfos2, paths = check_existence(metainfos, paths)
    vmsi_list = load_plant(metainfos2, paths)

    stem_polylines = [np.array(vmsi.get_stem().get_highest_polyline().polyline) for vmsi in vmsi_list]
    median_stem = get_median_polyline(polylines=stem_polylines)

    z_base = {sf: np.median([v.get_stem().get_highest_polyline().polyline[0][2] for v in vmsi_list
                             if v.metainfo.shooting_frame == sf]) for sf in ['elcom_2_c1_wide', 'elcom_2_c2_wide']}

    # ====== stem height annotation =====================================

    t_anot = []
    y_anot = []
    z_anot = []

    # available in NASHShare2
    with open('data/stem_annotation/stem_{}.json'.format(plantid)) as f:
        d = json.load(f)

        for name in d.keys():

            task = int(name.split('t')[1].split('_')[0])
            timestamp = next(m for m in metainfos if m.task == task).timestamp
            angle = int(name.split('a')[1].split('.')[0])
            sf = next(m for m in metainfos if m.task == task).shooting_frame

            for shape in d[name]['regions']:

                    y = shape['shape_attributes']['cy']
                    sf = next(m for m in metainfos if m.task == task).shooting_frame
                    median_stem_2d = phm3d_to_px2d(median_stem, sf, angle=angle)

                    if y > min(median_stem_2d[:, 1]):

                        f_2dto3d = interp1d(median_stem_2d[:, 1], np.array(median_stem)[:, 2], fill_value="extrapolate")
                        z = float(f_2dto3d(y))
                        z -= z_base[sf]

                        t_anot.append(timestamp)
                        y_anot.append(2448 - y)
                        z_anot.append(z)

                    else:
                        print('=============')

    #======== getting full prediction dataset ================================

    dfs = []
    plantid_files = [f for f in all_files if int(f.split('/')[-1][:4]) == plantid]
    for f in plantid_files:
        task = int(f.split('_')[-1].split('.csv')[0])

        timestamp = next(m for m in metainfos if m.task == task).timestamp
        dft = pd.read_csv(f)
        dft['t'] = timestamp

        dfs.append(dft)
    df = pd.concat(dfs)

    # ====== 2D ==============================================================

    # data preprocessing, stem extraction
    df = df[df['score'] > 0.95]
    df['y'] = 2448 - df['y']
    df_stem = df.sort_values('y', ascending=False).drop_duplicates(['t']).sort_values('t')
    f_smoothing = stem_height_smoothing(np.array(df_stem['t']), np.array(df_stem['y']))

    # ====== 3D ===========================================================

    # METHOD JUST FOR PLOT
    df_stem['y'] = 2448 - df_stem['y'] # retour a la normale
    z_row = []
    for _, row in df_stem.iterrows():
        sf = next(m for m in metainfos if m.timestamp == row['t']).shooting_frame
        median_stem_2d = phm3d_to_px2d(median_stem, sf, angle=row['angle'])

        f_2dto3d = interp1d(median_stem_2d[:, 1], np.array(median_stem)[:, 2], fill_value="extrapolate")
        z = float(f_2dto3d(row['y']))
        z -= z_base[sf]
        z_row.append(z)

    df_stem['z'] = z_row
    df_stem['y'] = 2448 - df_stem['y']  # retour a la normale


    # data preprocessing, stem extraction (NORMAL PIPELINE METHOD)
    #df = df[df['score'] > 0.95]
    #df_stem_3d = df.sort_values('z_phenomenal', ascending=False).drop_duplicates(['t']).sort_values('t')
    f_smoothing = stem_height_smoothing(np.array(df_stem['t']), np.array(df_stem['z']))
    df_stem['z_smooth'] = [f_smoothing(t) for t in df_stem['t']]

    # plt.figure(str(plantid) + '3D')
    # plt.plot(df['t'], df['z_phenomenal'], 'k*', markersize=2)
    # plt.plot(df_stem['t'], df_stem['z_phenomenal'], '-g')
    # plt.plot(df_stem['t'], [f_smoothing(t) for t in df_stem['t']], '-', color='orange')

    # ============ fast plot =================================================

    plt.figure('plantid {} 2D'.format(plantid))
    plt.plot(t_anot, y_anot, 'g-', label=str(plantid))
    plt.plot(df_stem['t'], df_stem['y'], 'k-', markersize=2)

    plt.figure('plantid {} 3D'.format(plantid))
    plt.plot(t_anot, z_anot, 'g-', label=str(plantid))
    plt.plot(df_stem['t'], df_stem['z'], 'k-', markersize=2)
    plt.plot(df_stem['t'], df_stem['z_smooth'], 'b-', markersize=2)


    # ======= group couples of observation + prediction =======================

    for t, y, z in zip(t_anot, y_anot, z_anot):
        if t in list(df_stem['t']) and y < 2440:
            if t in list(stem_phm.keys()):
                y_pred = float(df_stem[df_stem['t'] == t]['y'])
                z_pred_3d = float(df_stem[df_stem['t'] == t]['z'])
                z_pred_3d_smooth = float(df_stem[df_stem['t'] == t]['z_smooth'])

                res.append([y, y_pred, z, z_pred_3d, z_pred_3d_smooth, stem_phm[t]])

            else:
                print('no data')

res = pd.DataFrame(res, columns=['obs', 'pred', 'obs3d', 'pred3d', 'pred3d_smooth', 'pred_phm'])



res = pd.read_csv('data/paper/stem_height_validation.csv')

# pixel plot

x, y = np.array(res['obs']), np.array(res['pred'])
rmse = np.sqrt(np.sum((x - y) ** 2) / len(x))
mape = 100 * np.mean(np.abs((x - y) / x))
r2 = r2_score(x, y)

fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
plt.xlim((480, 2500))
plt.ylim((480, 2500))
plt.plot([-100, 3000], [-100, 3000], '-', color='grey')
plt.plot(x, y, 'k.', markersize=4)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('observation (px)', fontsize=14)
plt.ylabel('prediction (px)', fontsize=14)
plt.title('pixel')
ax.text(1950, 670, 'R² = {}'.format(round(r2, 3)), fontdict={'size': 12})
ax.text(1950, 600, 'RMSE = {} px'.format(round(rmse, 2)), fontdict={'size': 12})
ax.text(1950, 530, 'n = {}'.format(len(x)), fontdict={'size': 12})

# 3D plot

x, y = np.array(res['obs3d']), np.array(res['pred3d'])
rmse = np.sqrt(np.sum((x - y) ** 2) / len(x))
mape = 100 * np.mean(np.abs((x - y) / x))

for xi, yi in zip(x,y):
    print( xi, yi, np.abs((xi - yi) / xi) )

r2 = r2_score(x, y)

fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
plt.xlim((-60, 2050))
plt.ylim((-60, 2050))
plt.plot([-100, 3000], [-100, 3000], '-', color='grey')
plt.plot(x, y, 'k.', markersize=4)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('observation (mm)', fontsize=14)
plt.ylabel('prediction (mm)', fontsize=14)
plt.title('3D')
ax.text(1950, 670, 'R² = {}'.format(round(r2, 3)), fontdict={'size': 12})
ax.text(1950, 600, 'RMSE = {} mm'.format(round(rmse, 2)), fontdict={'size': 12})
ax.text(1950, 530, 'n = {}'.format(len(x)), fontdict={'size': 12})

# Final graph, 3D z smooth deep learning vs 3D z phenomenal:

x, y = np.array(res['obs3d']) / 10, np.array(res['pred3d_smooth']) / 10
rmse = np.sqrt(np.sum((x - y) ** 2) / len(x))
mape = 100 * np.mean(np.abs((x - y) / x))
r2 = r2_score(x, y)
bias = np.mean(x - y)

x2, y2 = np.array(res['obs3d']) / 10, np.array(res['pred_phm']) / 10
rmse2 = np.sqrt(np.sum((x2 - y2) ** 2) / len(x2))
mape2 = 100 * np.mean(np.abs((x2 - y2) / x2))
r2_2 = r2_score(x2, y2)
bias2 = np.mean(x2 - y2)

fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xlim((-6, 205))
plt.ylim((-6, 205))
plt.plot([-10, 300], [-10, 300], '-', color='grey')
plt.plot(x2, y2, '^', color='grey', markersize=6, label='Phenomenal') # \n RMSE = {} cm, R² = {}'.format(round(rmse2, 1), round(r2_2, 3)))
plt.plot(x, y, 'o', color='black', markersize=6, label='Phenomenal with deep-learning \nstem detection') # \nstem detection \n RMSE = {} cm, R² = {}'.format(round(rmse, 2), round(r2, 3)))
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('observation (cm)', fontsize=30)
plt.ylabel('prediction (cm)', fontsize=30)
plt.title('Stem height', fontsize=35)
#ax.text(1950, 670, 'R² = {}'.format(round(r2, 3)), fontdict={'size': 12})
#ax.text(1950, 600, 'RMSE = {} cm'.format(round(rmse, 2)), fontdict={'size': 12})
#ax.text(1950, 530, 'n = {}'.format(len(x)), fontdict={'size': 12})
leg = plt.legend(prop={'size': 16}, loc='upper left', markerscale=2)
leg.get_frame().set_linewidth(0.0)

ax.text(0.67, 0.17, 'n = {} \nBias = {} cm \nRMSE = {} cm \nR² = {}'.format(len(x), round(bias2, 2), round(rmse2, 1), round(r2_2, 3)), transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='grey')
ax.text(0.3, 0.17, 'n = {} \nBias = {} cm \nRMSE = {} cm \nR² = {}'.format(len(x), round(bias, 2), round(rmse, 2), round(r2, 3)), transform=ax.transAxes, fontsize=20,
        verticalalignment='top')


fig.savefig('paper/results/stem_validation', bbox_inches='tight')