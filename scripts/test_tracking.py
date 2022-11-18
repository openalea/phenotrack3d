import os
import numpy as np
import matplotlib.pyplot as plt

from openalea.phenomenal import object as phm_obj
import openalea.maizetrack.phenomenal_display as phm_display

from openalea.maizetrack.trackedPlant import TrackedPlant

from alinea.phenoarch.cache import Cache
from alinea.phenoarch.platform_resources import get_ressources
from alinea.phenoarch.meta_data import plant_data

import pandas as pd

exp = 'ZA22'

cache_client, image_client, binary_image_client, calibration_client = get_ressources(exp, cache='X:',
                                                                                     studies='Z:',
                                                                                     nasshare2='Y:')
parameters = {'reconstruction': {'voxel_size': 4, 'frame': 'pot'},
              'collar_detection': {'model_name': '3exp_xyside_99000'},
              'segmentation': {'force_stem': True}}
cache = Cache(cache_client, image_client, binary_image_client=binary_image_client,
              calibration_client=calibration_client, parameters=parameters)
index = cache.snapshot_index()
df_plant = plant_data(exp)
df_my_plants = pd.read_csv('data/plants_set_tracking.csv')
my_plants = list(df_my_plants[df_my_plants['exp'] == exp]['plant'])

if exp == 'ZA22':
    plants = [p for p in my_plants if int(p.split('/')[0]) not in [1580, 2045]]
elif exp == 'ZA20':
    plants = list(index.plant_index[index.plant_index['rep'].str.contains('EPPN')]['plant'])

# ===== visualise stem ==============================================================================================

from openalea.maizetrack.stem_correction import stem_height_smoothing

plantids = [1657, 1351, 1150, 1118, 1378]
plants_pb = [next(p for p in plants if int(p.split('/')[0]) == plantid) for plantid in plantids]
for plant in plants_pb:
    meta_snapshots = index.get_snapshots(index.filter(plant=plant, nview=13), meta=True)

    try:
        collars = cache.load_collar_temporal(meta_snapshots[0])
        plt.figure(meta_snapshots[0].pot)
        plt.plot(collars['timestamp'], collars['z_3d'], 'k.')
        plt.plot(collars['timestamp'], collars['z_stem'], 'r-')

        for task in collars['task'].unique():
            z_top = sorted(collars[collars['task'] == task]['z_3d'])[-20:]
            z_top_range = max(z_top) - min(z_top)
            print(z_top_range / np.std(z_top))




        gb = collars.groupby('task')[['timestamp', 'z_3d']].max().reset_index().sort_values('timestamp')
        f_smoothing = stem_height_smoothing(gb['timestamp'], gb['z_3d'])
        plt.plot(gb['timestamp'], [f_smoothing(t) for t in gb['timestamp']], 'g-')
    except:
        pass


# ===== explore tracking results ====================================================================================

res = []
plants_gap = []
for k, plant in enumerate(sorted(plants)):
    meta_snapshots = index.get_snapshots(index.filter(plant=plant, nview=13), meta=True)
    try:
        tracking = cache.load_tracking(meta_snapshots[0])
        row = [meta_snapshots[0].pot, len(tracking.groupby('task'))]

        for check in ['task_time_continuity', 'task_valid_stem', 'task_valid_features']:
            checks = tracking.groupby('task')[check].mean()
            print('{}/{} no {}'.format(len(checks) - np.sum(checks), len(checks), check))
            row.append(len(checks) - np.sum(checks))
        res.append(row)
    except:
        pass
res = pd.DataFrame(res, columns=['plantid', 'n', 'ano_t', 'ano_stem', 'ano_features'])

# ===================================================================================================================

plantid = 36  # 93
plant = next(p for p in plants if int(p.split('/')[0]) == plantid)

plant = plants[7]

print(plant)

meta_snapshots = index.get_snapshots(index.filter(plant=plant, nview=13), meta=True)

# ===================================================================================================================

segs = []
for m in meta_snapshots:
    try:
        seg = cache.load_segmentation(m)
        seg.metainfo = pd.Series({'task': m.task, 'timestamp': m.timestamp})
        segs.append(seg)
    except:
        print('{}: cannot load seg'.format(m.daydate))

# plant.features_extraction(w_h=0.03, w_l=0.004)

trackedplant = TrackedPlant.load_and_check(segs)

trackedplant.align_mature(start=0, gap=12.365, w_h=0.03, w_l=0.004,
                   gap_extremity_factor=0.2, n_previous=5000)
# plant.align_growing()
tracking = trackedplant.get_dataframe()

for check in ['task_time_continuity', 'task_valid_stem', 'task_valid_features']:
    checks = tracking.groupby('task')[check].mean()
    print('{}/{} no {}'.format(len(checks) - np.sum(checks), len(checks), check))

# segs = {m.task: cache.load_segmentation(m) for m in meta_snapshots if m.task in tracking['task'].unique()}

tr = tracking[tracking['mature']]

profile = {}
for r in [1, 2, 3]:
    s_r = tr[tr['rank_tracking'] == r]
    s_r = s_r[(s_r['timestamp'] - np.min(s_r['timestamp'])) / 3600 / 24 < 20]
    profile[r] = round(np.mean(s_r['l_extended']), 1)
print(profile)

leaves, ranks = [], []
for _, row in tr.iterrows():
    # if 1584500000 < row.timestamp < 1584600000:
    seg = next(seg for seg in segs if seg.metainfo.task == row.task)
    leaves.append(seg.get_leaf_order(row.rank_phenomenal))
    ranks.append(row.rank_tracking - 1)
phm_display.plot_leaves(leaves, ranks)



























