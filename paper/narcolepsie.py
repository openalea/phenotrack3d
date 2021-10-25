from openalea.maizetrack.utils import phm3d_to_px2d
from openalea.maizetrack.stem_correction import xyz_last_mature

from openalea.maizetrack.local_cache import metainfos_to_paths, get_metainfos_ZA17, check_existence, load_plant
from openalea.maizetrack.utils import phm3d_to_px2d
from openalea.maizetrack.trackedPlant import TrackedPlant
from openalea.maizetrack.stem_correction import stem_height_smoothing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.interpolate import interp1d
from skimage import io
import cv2

df_tt = pd.read_csv('TT_ZA17.csv')
f_tt = interp1d(df_tt['timestamp'], df_tt['ThermalTime'])

# ============================================================================

# for k in range(300, 600):
#     img = io.imread('data/rgb_insertion_annotation/train2/img{}.png'.format(k))
#     anot = np.loadtxt('data/rgb_insertion_annotation/train2/img{}.txt'.format(k))
#     if anot.shape == (5,):
#         anot = np.array([anot])
#     for _, x, y, _, _ in anot:
#
#         img = cv2.circle(img, (int(x * 416), int(y * 416)), 6, (255, 0, 0), -1)
#     io.imsave('Narcolepsie/dl/dl{}.png'.format(k), img)
#

# ============================================================================

#df_tt = pd.read_csv('TT_ZA17.csv')

plantid = 1429

res = []

# ===== phenomenal brut ======

folder = 'local_cache/ZA17/1429_nosmooth/'

metainfos = get_metainfos_ZA17(plantid)
paths = metainfos_to_paths(metainfos, folder=folder)
metainfos2, paths = check_existence(metainfos, paths)
vmsi_list = load_plant(metainfos2, paths)

for vmsi in vmsi_list:
    if vmsi.get_mature_leafs():
        i = np.argmax([l.info['pm_z_base'] for l in vmsi.get_mature_leafs()])
        xyz = vmsi.get_mature_leafs()[i].info['pm_position_base']
        xyz[2] = vmsi.get_mature_leafs()[i].info['pm_z_base_voxel']
        y = phm3d_to_px2d(xyz, vmsi.metainfo.shooting_frame, angle=60)[0][1]
        t = vmsi.metainfo.timestamp
        res.append([t, 2448 - y, 'phm'])

# ===== deep learning =====

folder = 'local_cache/ZA17/collars_voxel4_tol1_notop_vis4_minpix100'
all_files = [folder + '/' + rep + '/' + f for rep in os.listdir(folder) for f in os.listdir(folder + '/' + rep)]

dfs = []

plantid_files = [f for f in all_files if int(f.split('/')[-1][:4]) == plantid]
for f in plantid_files:
    daydate = f.split('/')[3]
    dft = pd.read_csv(f)
    #dft['t'] = df_tt[df_tt['daydate'] == daydate].iloc[0]['ThermalTime']
    dft['t'] = next(m for m in metainfos2 if m.daydate == daydate).timestamp
    dfs.append(dft)
df = pd.concat(dfs)

df = df[df['score'] > 0.95]
df['y'] = 2448 - df['y']

fig, ax = plt.subplots()
ax.tick_params(axis='both', which='major', labelsize=20)
plt.plot(df['t'], df['y'], 'k.', markersize=2)
plt.xlabel('Thermal time $(day_{20°C})$', fontsize=30)
plt.ylabel('collar height (pixels)', fontsize=30)

df_stem = df.sort_values('y', ascending=False).drop_duplicates(['t']).sort_values('t')
plt.plot(df_stem['t'], df_stem['y'], 'r-', label='stem height')
f_smoothing = stem_height_smoothing(np.array(df_stem['t']), np.array(df_stem['y']))
plt.plot(df_stem['t'], [f_smoothing(t) for t in df_stem['t']], 'r--', label='stem height smoothing')
plt.legend()

for _, row in df_stem.iterrows():
    res.append([row['t'], row['y'], 'dl'])

# plt.figure(plantid)
# plt.plot(df['t'], df['y'], 'k*', markersize=2)
# plt.plot(df_stem['t'], df_stem['y'], '-b')
# plt.plot(df_stem['t'], [f_smoothing(t) for t in df_stem['t']], '-', color='orange')

# ====== stem height annotation =====================================

with open('data/stem_annotation/stem_{}.json'.format(plantid)) as f:
    d = json.load(f)

    for name in d.keys():

        task = int(name.split('t')[1].split('_')[0])
        timestamp = next(m for m in metainfos if m.task == task).timestamp
        angle = int(name.split('a')[1].split('.')[0])
        sf = next(m for m in metainfos if m.task == task).shooting_frame

        for shape in d[name]['regions']:

                y = shape['shape_attributes']['cy']
                #tt = df_tt.iloc[np.argmin(np.abs(df_tt['timestamp'] - timestamp))]['ThermalTime']
                res.append([timestamp, 2448 - y, 'anot'])


# ===== annotation =====
res = pd.DataFrame(res, columns=['t', 'y', 'var'])
res['tt'] = [f_tt(t) for t in res['t']]

selec = res[res['var'] == 'dl'].sort_values('tt')
plt.plot(selec['tt'], selec['y'], 'g-')
selec = res[res['var'] == 'anot'].sort_values('tt')
plt.plot(selec['tt'], selec['y'], 'r-')
selec = res[res['var'] == 'phm'].sort_values('tt')
plt.plot(selec['tt'], selec['y'], 'k-')



selec = res[res['var'] == 'phm'].sort_values('t')
plt.plot(selec['tt'], (selec['y'] - np.min(selec['y'])) / 10, 'k-')
plt.xlabel('Thermal time $(day_{20°C})$', fontsize=20)
plt.ylabel('Stem height (cm)', fontsize=20)





df_stem['tt'] = [f_tt(t) for t in df_stem['t']]
plt.plot(df_stem['tt'], np.array([f_smoothing(t) for t in df_stem['t']]) - 530, 'r-', label='stem height smoothing')
plt.xlabel('Thermal time $(day_{20°C})$', fontsize=20)
plt.ylabel('Stem height (cm)', fontsize=20)







