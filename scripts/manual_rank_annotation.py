""" script to annotate leaf ranks manually. REFACTORING 10/11/2022 """

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

import openalea.maizetrack.phenomenal_display as phm_display

from openalea.maizetrack.trackedPlant import TrackedPlant
from openalea.maizetrack.rank_annotation import annotate, rgb_and_polylines

from alinea.phenoarch.cache import Cache
from alinea.phenoarch.platform_resources import get_ressources


exp = 'ZA17'
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


if exp == 'ZA17':
    plants = my_plants
if exp == 'ZA22':
    plants = [p for p in my_plants if int(p.split('/')[0]) not in [1580, 2045]]
    plants = [p for p in plants if int(p.split('/')[0]) not in [1657, 1351, 1150, 1118, 1378]]  # TODO stem problem
elif exp == 'ZA20':
    plants = list(index.plant_index[index.plant_index['rep'].str.contains('EPPN')]['plant'])

# ===== load segmentations + polylines ===============================================================================

datas = {}

for k_plant in range(88, 93):

    # plant = '401/ZM4971/CZL19058/cimmyt/EXPOSE/WW/Rep_4/07_41/ARCH2022-01-10'
    plant = plants[k_plant]
    print(plant)
    # plant = '1953/ZM4996/CZL_160/cimmyt/EXPOSE/WW/Rep_6/33_33/ARCH2022-01-10'
    # plant = '632/ZM2606/NC354/lepse/EXPOSE/WW/Rep_4/11_32/ARCH2022-01-10'  # good to visualize senescing criteria

    print('loading segmentations...')
    meta_snapshots = index.get_snapshots(index.filter(plant=plant, nview=13), meta=True)

    if exp == 'ZA22':
        # remove last snapshot if too late (more than 14 days gap)
        last_dt = np.diff([meta_snapshots[k].timestamp for k in [-2, -1]])[0] / 3600 / 24
        if last_dt > 14:
            meta_snapshots = meta_snapshots[:-1]
            print('removed last snapshot')

    segs = {}
    for m in meta_snapshots:
        try:
            seg = cache.load_segmentation(m)
            seg.metainfo = pd.Series({'task': m.task, 'timestamp': m.timestamp})
            segs[m.task] = seg
        except:
            print('{}: cannot load seg'.format(m.daydate))
    trackedplant = TrackedPlant.load_and_check(list(segs.values()))

    print('computing polylines simplification...')
    trackedplant.simplify_polylines()

    datas[plant] = {'trackedplant': trackedplant, 'meta_snapshots': meta_snapshots}

"""
=======================================================================================================================
=======================================================================================================================
=======================================================================================================================
"""

plant = list(datas.keys())[5]
d = datas[plant]
trackedplant, meta_snapshots = d['trackedplant'], d['meta_snapshots']

print('loading images for annotation...')
angles = [60, 150]
images = {a: {} for a in angles}
for snapshot in trackedplant.snapshots:
    m = next(m for m in meta_snapshots if m.task == snapshot.metainfo.task)
    for angle in angles:
        image_path = next(p for p, v, a in zip(m.path, m.view_type, m.camera_angle) if v == 'side' and a == angle)
        images[angle][m.task] = cv2.cvtColor(cache.image_client.imread(image_path), cv2.COLOR_BGR2RGB)
        # image_path = next(p for p, v, a in zip(m.binary_path, m.view_type, m.camera_angle) if v == 'side' and a == angle)
        # images[angle][m.task] = cv2.cvtColor(cache.binary_image_client.imread(image_path), cv2.COLOR_BGR2RGB)


# ===== mature leaves tracking ====================================================================================

# good for plantid 1050 (col resserés, cas intéressant)
trackedplant.align_mature(start=20, gap=12, w_h=0.03, w_l=0.004, gap_extremity_factor=0.2, n_previous=5000)

# start = -1 better than 0 on one 1 plant
trackedplant.align_mature(start=-1, gap=2.5, w_h=0.02, w_l=0.004, gap_extremity_factor=0.8, n_previous=5000)

trackedplant.align_mature(start=20, gap=1.8, w_h=0.03, w_l=0.004, gap_extremity_factor=0.8, n_previous=5000)

# ===== visualise tracking ========================================================================================

tracking = trackedplant.get_dataframe()
tr = tracking[tracking['mature']]

# valid_tasks = [s.metainfo.task for s in trackedplant.valid_snapshots()]
# valid_tasks = list(set(valid_tasks[::3] + [valid_tasks[-1]]))
# tr = tr[tr['task'].isin(valid_tasks)]

leaves, ranks = [], []
for _, row in tr.iterrows():
    # using trackedleaf objects and not the one from phenomenal, because they have pre-computed simplified polyline
    # which make 3D plotting faster
    snapshot = next(s for s in trackedplant.snapshots if s.metainfo.task == row['task'])
    leaves.append(snapshot.leaves[row['rank_phenomenal'] - 1])
    # ranks.append(1 if snapshot.leaves[row['rank_phenomenal'] - 1].info['pm_insertion_angle'] < 120 else 2)
    ranks.append(row.rank_tracking)
phm_display.plot_leaves(leaves, ranks, simplify=True)

profile = {}
for r in [1, 2, 3, 4]:
    s_r = tr[tr['rank_tracking'] == r]
    s_r = s_r[(s_r['timestamp'] - np.min(s_r['timestamp'])) / 3600 / 24 < 20]
    profile[r] = round(np.mean(s_r['l_extended']), 1)
print(profile)
print(['L{}/L{} = {}'.format(r2, r1, round(profile[r2] / profile[r1], 2)) for (r1, r2) in [[1, 2], [2, 3], [3, 4]]])

# ===== manual annotation ==========================================================================================

print('loading images and leaf polylines...')
angles = list(images.keys())
frame = cache.parameters['reconstruction']['frame']
projections = {a: {sf: cache.load_calibration(sf).get_projection(id_camera='side', rotation=a, world_frame=frame)
               for sf in set([m.shooting_frame for m in meta_snapshots])} for a in angles}

annot = {}
previous_ranks = None
for snapshot in trackedplant.snapshots:
    task = int(snapshot.metainfo.task)
    m = next(m for m in meta_snapshots if m.task == task)
    annot[task] = {'metainfo': m, 'leaves_info': [], 'leaves_pl': [], 'images': {}}
    for angle in angles:
        annot[task]['images'][angle] = images[angle][task]
    ranks = snapshot.get_ranks()
    for leaf, r_tracking in zip(snapshot.leaves, ranks):
        mature = leaf.info['pm_label'] == 'mature_leaf'
        if mature:
            annot[task]['leaves_pl'].append({a: projections[a][m.shooting_frame](leaf.real_pl) for a in angles})
            tip = leaf.real_pl[-1]
            annot[task]['leaves_info'].append({'mature': mature, 'selected': False,
                                               'rank_phm': leaf.info['pm_leaf_number'], 'rank': r_tracking,
                                               'tip': tip})

# # change all ranks of all tasks (to correct shift)
if False:
    for task in annot.keys():
        for l in annot[task]['leaves_info']:
            # if l['rank'] == 4:
            #     l['rank'] == 0
            # if l['rank'] > 4:
            #     l['rank'] -= 1
            if l['rank'] != 0:
                l['rank'] += 1

annotate(annot)

# ===== save annotation ==============================================================================================

df = []
for task, annot_task in annot.items():
    ranks = [r for r in [v['rank'] for v in annot_task['leaves_info']] if r > 0]
    if len(ranks) != len(set(ranks)):
        raise Exception('task', task, ': several leaves have the same rank ! Cannot save')
    snapshot = next(s for s in trackedplant.snapshots if s.metainfo.task == task)
    for leaf_info in annot_task['leaves_info']:
        print(leaf_info['rank_phm'])
        snapshot.get_leaf_order(leaf_info['rank_phm'])
        x_tip, y_tip, z_tip = [round(k) for k in snapshot.leaves[leaf_info['rank_phm'] - 1].real_pl[-1]]  # not super safe
        df.append([int(snapshot.metainfo.task), leaf_info['rank'], x_tip, y_tip, z_tip])
df = pd.DataFrame(df, columns=['task', 'rank', 'x_tip', 'y_tip', 'z_tip'])

df.to_csv('rank_annotation/{}_{}.csv'.format(exp, meta_snapshots[0].pot), index=False)

# ===== load annotation # TODO =======================================================================================

annot = pd.read_csv('rank_annotation/{}_{}.csv'.format(exp, meta_snapshots[0].pot))

# visual verification
leaves, ranks = [], []
annot_tips = np.array(annot[['x_tip', 'y_tip', 'z_tip']])
for _, row in tr.iterrows():
    snapshot = next(s for s in trackedplant.snapshots if s.metainfo.task == row['task'])
    leaf = snapshot.leaves[row['rank_phenomenal'] - 1]
    leaves.append(leaf)
    tip = leaf.real_pl[-1]
    dists = np.sum(np.abs(annot_tips - tip), axis=1)
    if np.min(dists) < 0.001:
        rank = annot.iloc[np.argmin(dists)]['rank']
    else:
        rank = 0
    ranks.append(rank)
phm_display.plot_leaves(leaves, ranks, simplify=True)


# TODO remove
plant = '0330/DZ_PG_01/ZM4381/WW/Rep_2/06_30/ARCH2017-03-30'
meta_snapshots = index.get_snapshots(index.filter(plant=plant, nview=13), meta=True)
segs = {}
for m in meta_snapshots:
    try:
        seg = cache.load_segmentation(m)
        seg.metainfo = pd.Series({'task': m.task, 'timestamp': m.timestamp})
        segs[m.task] = seg
    except:
        print('{}: cannot load seg'.format(m.daydate))

old_annot = pd.read_csv('rank_annotation/2021_paper_phenotrack/rank_annotation_330.csv')

# seg = list(segs.values())[-1]
for task, seg in segs.items():
    print(task)
    # current segmentation
    tips = np.array([leaf.real_longest_polyline()[-1] for leaf in seg.get_mature_leafs()])

    if len(tips) > 0:

        calibration = cache.load_calibration(next(m for m in meta_snapshots if m.task == task).shooting_frame)
        frame = calibration.get_frame(frame='pot')
        tips_bis = frame.global_point(tips)
        annot = old_annot[old_annot['task'] == task]
        tips_annot = np.array([eval(k) for k in annot['leaf_tip']])

        if len(tips_annot) > 0:
            for tip in tips_bis:
                dists = np.sqrt(np.sum((tips_annot - tip) ** 2, axis=1))
                print([round(k, 1) for k in sorted(dists)[:2]])


def load_rank_annotation(self, folder='rank_annotation/'):
    """
    Load rank annotation for each leaf of each snapshot (-2 = not annotated or annotation not found, -1 = anomaly,
    0 = rank 1, 1 = rank 2, etc.) Each annotation is associated to the x,y,z tip position of its corresponding leaf,
    in a csv.
    """

    # TODO : don't stock annotated ranks in both snapshot and leaf ?
    # annotation copy in Z:\lepseBinaries\TaggedImages\ARCH2017-03-30\rank_annotation
    # TODO update: plantid not used anymore.
    df_path = folder + 'rank_annotation_{}.csv'.format(self.plantid)

    if os.path.isfile(df_path):
        df = pd.read_csv(df_path)

        for snapshot in self.snapshots:
            task = snapshot.metainfo.task

            if task not in df['task'].unique():
                # a task was not annotated
                snapshot.rank_annotation = [-2] * len(snapshot.leaves)

                for leaf in snapshot.leaves:
                    leaf.rank_annotation = -2
            else:
                dftask = df[df['task'] == task]
                snapshot.rank_annotation = []
                for leaf in snapshot.leaves:
                    tip = leaf.real_longest_polyline()[-1]
                    dftip = dftask[dftask['leaf_tip'] == str(tip)]
                    if dftip.empty:
                        # a leaf was not annotated
                        snapshot.rank_annotation.append(-2)
                        leaf.rank_annotation = -2
                    else:
                        snapshot.rank_annotation.append(dftip.iloc[0]['rank'])
                        leaf.rank_annotation = dftip.iloc[0]['rank']




