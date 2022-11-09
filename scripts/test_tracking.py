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

# ===================================================================================================================

plantid = 636  # 93
plant = next(p for p in plants if int(p.split('/')[0]) == plantid)
meta_snapshots = index.get_snapshots(index.filter(plant=plant, nview=13), meta=True)

segs = []
for m in meta_snapshots:
    try:
        seg = cache.load_segmentation(m)
        seg.metainfo = pd.Series({'task': m.task, 'timestamp': m.timestamp})
        segs.append(seg)
    except:
        print('{}: cannot load seg'.format(m.daydate))

# ===================================================================================================================

plant = TrackedPlant.load_and_check(segs)

# plant.features_extraction(w_h=0.03, w_l=0.004)

plant.align_mature(direction=1, gap=12.365, w_h=0.03, w_l=0.004, gap_extremity_factor=0.2, n_previous=5000)
plant.align_growing()

tracking = plant.get_dataframe()

for check in ['time_continuity', 'valid_stem', 'valid_features']:
    checks = tracking.groupby('task')[check].mean()
    print('{}/{} no {}'.format(len(checks) - np.sum(checks), len(checks), check))

segs = {m.task: cache.load_segmentation(m) for m in meta_snapshots if m.task in tracking['task'].unique()}

tr = tracking[tracking['mature']]
leaves, ranks = [], []
for _, row in tr.iterrows():
    # if 1584500000 < row.timestamp < 1584600000:
    seg = next(seg for seg in segs if seg.metainfo.task == row.task)
    leaves.append(seg.get_leaf_order(row.rank_phenomenal))
    ranks.append(row.rank_tracking - 1)
phm_display.plot_leaves(leaves, ranks)



























