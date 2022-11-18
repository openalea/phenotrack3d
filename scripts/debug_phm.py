import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

import openalea.maizetrack.phenomenal_display as phm_display

from alinea.phenoarch.cache import Cache
from alinea.phenoarch.platform_resources import get_ressources

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

df_my_plants = pd.read_csv('data/plants_set_tracking.csv')
my_plants = list(df_my_plants[df_my_plants['exp'] == exp]['plant'])
plants = [p for p in my_plants if int(p.split('/')[0]) not in [1580, 2045]]

# ===== identify where the is a problem ============================================================================

tracking_all = []
for plant in plants:
    print(plant)
    meta_snapshots = index.get_snapshots(index.filter(plant=plant, nview=13), meta=True)
    try:
        tracking = cache.load_tracking(meta_snapshots[0])
        tracking['plantid'] = meta_snapshots[0].pot
        tracking_all.append(tracking)
    except:
        pass
tracking_all = pd.concat(tracking_all)
gb = tracking_all.groupby(['plantid', 'task'])[['timestamp', 'task_valid_features']].mean().reset_index()
gb2 = gb[~gb['task_valid_features']].sort_values('timestamp')

# ===== explore tracking results ====================================================================================

gb3 = []
for k in range(100):
    plantid, task = gb2.iloc[k][['plantid', 'task']]
    plant = next(p for p in plants if int(p.split('/')[0]) == plantid)
    meta_snapshots = index.get_snapshots(index.filter(plant=plant, nview=13), meta=True)
    meta_snapshot = next(m for m in meta_snapshots if m.task == task)
    vmsi = cache.load_segmentation(meta_snapshot)
    if min([len(l.info) for l in vmsi.get_leafs()]) < 23:
        for l in vmsi.get_leafs():
            if len(l.info) < 23:
                print(plantid, l.info['pm_label'], l.info['pm_leaf_number'], len(l.info))
        gb3.append([plantid, task])



# visualise if artifact
angle = 60
path = meta_snapshot.path[meta_snapshot['camera_angle'].index(angle)]
rgb = cv2.cvtColor(cache.image_client.imread(path), cv2.COLOR_BGR2RGB)
projection = cache.load_calibration(meta_snapshot.shooting_frame).get_projection(id_camera='side', rotation=angle,
                                                                     world_frame=cache.parameters['reconstruction'][
                                                                         'frame'])

pl = projection(np.array(vmsi.get_stem().get_highest_polyline().polyline))
plt.imshow(rgb)
plt.plot(pl[:, 0], pl[:, 1], 'r-')
for leaf in vmsi.get_leafs():
    col = ('b' if leaf.info['pm_label'] == 'mature_leaf' else 'orange')
    if len(leaf.info) < 23:
        col = 'g'
    pl = projection(np.array(leaf.real_longest_polyline()))
    plt.plot(pl[:, 0], pl[:, 1], '-', color=col)


voxelgrid = cache.load_voxelgrid(meta_snapshot)
voxelskeleton = cache.load_voxelskeleton(meta_snapshot)
calibration = cache.load_calibration(meta_snapshot['shooting_frame'])

from openalea.maizetrack.utils import missing_data
from openalea.phenomenal.segmentation import graph_from_voxel_grid, maize_segmentation, maize_analysis

collars = cache.load_collar_temporal(meta_snapshot)
z_stem = collars[collars['task'] == meta_snapshot.task].iloc[0]['z_stem']

graph = graph_from_voxel_grid(voxelgrid)
vms = maize_segmentation(voxelskeleton, graph, z_stem=z_stem)
vmsi = maize_analysis(vms)

phm_display.plot_vmsi([vmsi])

# missing data
stem_needed_info = ['pm_z_base', 'pm_z_tip']
if not all([k in vmsi.get_stem().info for k in stem_needed_info]):
    print('missing data stem')
leaf_needed_info = ['pm_position_base', 'pm_z_tip', 'pm_label', 'pm_azimuth_angle', 'pm_length']
for leaf in vmsi.get_leafs():
    if not all([k in leaf.info for k in leaf_needed_info]):
        print('missing data leaf', leaf.info['pm_leaf_number'], len(leaf.info))








