import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from openalea.phenomenal import object as phm_obj
from openalea.maizetrack.phenomenal_display import *

from alinea.phenoarch.cache import snapshot_index, load_collar_detection
from alinea.phenoarch.platform_resources import get_ressources
from alinea.phenoarch.meta_data import plant_data

exp = 'ZA22'

# csv generated with maizetrack/scripts/pheno_multi_exp.py
pheno = pd.read_csv('data/pheno.csv')

cache_client, image_client, binary_image_client, calibration_client = get_ressources(exp, cache='X:', studies='Z:', nasshare2='Y:')
index = snapshot_index(exp, image_client=image_client, cache_client=cache_client, binary_image_client=binary_image_client)

""" ZA22 = 2506 pots """
df_plant = plant_data(exp)

""" ZA22 index = 2399 pots (pots >= 2400 not included, probably test or empty) """
meta_exp = index.snapshot_index.copy()

meta_exp['t'] = (meta_exp['timestamp'] - np.min(meta_exp['timestamp'])) / 3600 / 24

meta_exp['genotype'] = ['/'.join(p.split('/')[1:3]) for p in meta_exp['plant']]
meta_exp['scenario'] = [p.split('/')[5] for p in meta_exp['plant']]

# ===================================================================================================================

gb = meta_exp.groupby('genotype').nunique('plant').reset_index()
print([(k, list(gb['plant']).count(k)) for k in gb['plant'].unique()])
""" 309G * 6reps + 58G * 4reps (+ 7G * 8reps + others) """
print(len(meta_exp[meta_exp['genotype'].isin(gb[gb['plant'] > 8]['genotype'])]['pot'].unique()))
genotypes_special = meta_exp[meta_exp['genotype'].isin(gb[gb['plant'] > 8]['genotype'])]['genotype'].unique()
""" others = 257 reps among 7G """

plt.figure()
for plantid in sorted(meta_exp['pot'].unique())[::10]:
    s = meta_exp[meta_exp['pot'] == plantid]
    plt.plot(s['timestamp'], s['pot'], 'k.-')

"""
===> G 4reps: un peu moins d'images / moins longtemps.
===> G 6reps: globalement même frequence/duree entre GxE. ~30% 50j, le reste 65j, semble random selon GxE
"""
for k1, g in enumerate([g for g in meta_exp['genotype'].unique() if g not in genotypes_special][::20]):
    s = meta_exp[meta_exp['genotype'] == g]
    for k2, plantid in enumerate(s['pot'].unique()):
        s2 = s[s['pot'] == plantid]
        col = 'blue' if s2['scenario'].iloc[0] == 'WW' else 'red'
        plt.plot(s2['t'], [k2 + 10 * k1] * len(s2), '.-', color=col)

for g in meta_exp['genotype'].unique():
    selec = meta_exp[meta_exp['genotype'] == g]
    plant = selec.groupby('plant')['timestamp'].max().sort_values().index[-1]
    query = index.filter(plant=plant)
    meta_snapshots = index.get_snapshots(query, meta=True)

    m = meta_snapshots[-5]
    px_max, angle_max = 0, None
    for angle, view, path in zip(m['camera_angle'], m['view_type'], m['binary_path']):
        if view == 'side':
            bin = cv2.cvtColor(cv2.imread('Y:/lepseBinaries/' + path), cv2.COLOR_BGR2RGB)
            px_nb = len(np.where(bin == 255)[0])
            # print(px_nb)
            if px_nb > px_max:
                px_max, angle_max = px_nb, angle

    if angle_max is None:
        path = next(path for a, v, path in zip(m.camera_angle, m.view_type, m.path) if a == 60 and v == 'side')
        name = 'data/ZA22_visualisation/{}_{}_60default.png'.format(g.replace('/', '_'), m.daydate)
    else:
        path = next(path for a, v, path in zip(m.camera_angle, m.view_type, m.path) if a == angle_max and v == 'side')
        name = 'data/ZA22_visualisation/{}_{}_{}.png'.format(g.replace('/', '_'), m.daydate, angle_max)

    rgb = cv2.cvtColor(cv2.imread('Z:/' + path), cv2.COLOR_BGR2RGB)
    plt.imsave(name, rgb)

# ===== binary = f(t) ===============================================================================================

for g in meta_exp['genotype'].unique():

    selec = meta_exp[meta_exp['genotype'] == g]

    for plant in selec['plant'].unique():
        query = index.filter(plant=plant)
        meta_snapshots = index.get_snapshots(query, meta=True)

        angle = 60

        heights = []
        for m in meta_snapshots:
            path = next(path for a, v, path in zip(m.camera_angle, m.view_type, m.binary_path) if a == 60 and v == 'side')
            bin = cv2.imread('Y:/lepseBinaries/' + path)
            px_nb = len(np.where(bin == 255)[0])
            heights.append(2448 - np.min(np.where(bin == 255)[0]))

        col = 'blue' if m.scenario == 'WW' else 'red'
        plt.plot([m.timestamp for m in meta_snapshots], heights, '.-', color=col)








