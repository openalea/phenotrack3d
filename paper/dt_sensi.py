import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from openalea.maizetrack.local_cache import get_metainfos_ZA17, metainfos_to_paths, check_existence, load_plant
from openalea.maizetrack.trackedPlant import TrackedPlant

folder = 'data/tracking/'
plantids = [int(f.split('.')[0]) for f in os.listdir(folder) if os.path.isfile(folder + f)]

df_list = []
for plantid in plantids:
    dfid = pd.read_csv(folder + '{}.csv'.format(plantid))
    df_list.append(dfid)
df = pd.concat(df_list)
df = df[~df['rank_annotation'].isin([-1, 0])]
# df = df[df['timestamp'] < 1496509351]
# df = df[df['timestamp'] < 1495922400] # 05-28

# ====== mature tracking, but with different end dates ============================================================*

# selec = df
n, accuracy = [], []
dt = {}

for dt_div in [1, 2, 4]:
    dt[dt_div] = []

    for plantid in df['plantid'].unique():

        print(plantid)

        metainfos = get_metainfos_ZA17(plantid)

        metainfos = sorted(metainfos, key=lambda k: k.timestamp)

        paths = metainfos_to_paths(metainfos, folder='local_cache/cache_ZA17/segmentation_voxel4_tol1_notop_vis4_minpix100_stem_smooth_tracking')
        metainfos, paths = check_existence(metainfos, paths)

        dt[dt_div].append(np.diff([m.timestamp / 3600 for m in metainfos]))

        metainfos, paths = metainfos[::dt_div], paths[::dt_div]

        vmsi_list = load_plant(metainfos, paths)

        plant_ref = TrackedPlant.load_and_check(vmsi_list)
        plant_ref.load_rank_annotation()

        dmax = '2017-06-03'

        plant = copy.deepcopy(plant_ref)

        plant.snapshots = [s for s in plant.snapshots if s.metainfo.daydate < dmax]

        plant.align_mature(direction=1, gap=12.365, w_h=0.03, w_l=0.002, gap_extremity_factor=0.2, n_previous=500,
                           rank_attribution=True)
        plant.align_growing()

        df_res = plant.get_dataframe(load_anot=False)
        print(dt_div, round(len(df_res[df_res['rank_tracking'] == df_res['rank_annotation']]) / len(df_res), 2))
        df_res.to_csv('data/tracking/test_dt/{}_{}.csv'.format(plantid, dt_div))

plt.figure()
plt.hist(np.concatenate(dt[2]), 80)
plt.xlabel('Δt between two consecutive images (h)')

for dt_div in [1, 2, 4]:
    acc1, acc2, n = [], [], []
    for plantid in plantids:

        df = pd.read_csv('data/tracking/test_dt/{}_{}.csv'.format(plantid, dt_div))

        selecid = df[~df['rank_annotation'].isin([-1, 0])]
        selec = selecid[(selecid['mature'] == True)]
        a1 = len(selec[selec['rank_annotation'] == selec['rank_tracking']]) / len(selec)
        selec = selecid[(selecid['mature'] == False)]
        a2 = len(selec[selec['rank_annotation'] == selec['rank_tracking']]) / len(selec)
        acc1.append(a1)
        acc2.append(a2)
        # print(plantid, round(a1, 3), round(a2, 3), len(selecid))
        n.append(len(selec))
    print('===== {} =========================='.format(dt_div))
    print(min(acc1), np.mean(acc1))
    print(min(acc2), np.mean(acc2))
    print(min(n), np.mean(n))





