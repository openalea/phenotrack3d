'''
Main tutorial for maize tracking pipeline
'''

import copy
from openalea.maizetrack.data_loading import get_metainfos_ZA17, metainfos_to_paths, check_existence, load_plant
from openalea.maizetrack.trackedPlant import TrackedPlant, align_growing

from openalea.maizetrack.alignment import multi_alignment, detect_abnormal_ranks

# [16, 672, 709, 876, 911, 948, 995, 1014, 1127]
# [17, 30, 557, 566, 567, 586, 603, 613, 796, 1019, 1057, 1083]

# 314 , 444, 464, 471, 477, 480

TEST_SET = [348, 1301, 832, 1276, 1383, 1424, 940, 1270, 931, 925, 474, 794, 1283, 330, 1421,
            907, 316, 1284, 336, 439, 959, 915, 1316, 1434, 905, 313, 1391, 461, 424, 329, 784, 1398, 823, 1402, 430,
            1416, 1309, 1263, 811, 1413, 466, 478, 822, 918, 937, 1436, 1291, 470, 327]

TEST_SET = [1276, 948, 803, 931, 827, 1424, 1435, 479, 449, 318, 348, 1266, 705, 1662, 1668]

# TODO : 1316

# TODO : 1301 pb F1 a fixer ?

# 17 : thalle

# > 50 dates :
# 348 : tres compliqué en haut, feuilles toutes du meme cote
# 316 !!, 474, 1316 : fil metallique pris pour une feuille tout en bas. Indel serait utile la, a revoir
# 781 : azimuts tres variables (rotations de feuilles) + feuilles lacerées
# 907 : fil metallique pris pour une feuille, mais en milieu de plante
# 925 : rotation de la plante, mais pas du pot, a un moment
# 1383 : joli
# 1424 : joli, facile
# 1434 : mature marche mal, double inversion, compliqué a regler

# 905 : plusieurs pb non-matures sur des choses faciles... tester differents calculs polylines


# df2 = pd.DataFrame()
# df2.to_csv('res0906' + '.csv')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openalea.maizetrack.utils import phm3d_to_px2d
from openalea.maizetrack.data_loading import get_rgb
import json


# 832
#for plantid in [931, 1270, 1276, 1301, 1383, 1424]:

plantids = [313, 316, 329, 330, 336, 348, 424, 439, 461, 474, 794, 832, 905, 907, 915, 925, 931]
df = pd.DataFrame(columns=['plantid', 'timestamp', 'leaf_number', 'pred', 'h', 'h_vx'])

for plantid in plantids:

    # ======================== load vmsi, tracking

    print('plantid', str(plantid))

    metainfos = get_metainfos_ZA17(plantid)

    #metainfo = next(m for m in metainfos if m.task == 6696)
    #for angle in [30 * k for k in range(12)]:
    #    get_rgb(metainfo, angle, plant_folder=True, save=True, side=True)

    paths = metainfos_to_paths(metainfos, stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), old=False)
    metainfos, paths = check_existence(metainfos, paths)

    # load vmsi objects, and associate them to their metainfo
    print('Loading vmsi objects ...')
    vmsi_list = load_plant(metainfos, paths)
    print('vmsi found for {}/{} metainfos'.format(len(vmsi_list), len(metainfos)))

    plant_ref = TrackedPlant.load_and_check(vmsi_list)

    print('creating copy')
    plant = copy.deepcopy(plant_ref)
    print('copy ok')

    plant.align_mature(direction=-1)

    align_growing(plant)

    #plant.display()

    # ========================= dataframe (h, h_vx)

    for snapshot in plant.snapshots:

        print(snapshot.metainfo.daydate)

        sf = snapshot.metainfo.shooting_frame

        pred = snapshot.get_ranks()

        stem_pl = np.array(list(snapshot.get_stem().get_highest_polyline().polyline))
        stem_vx = np.array(list(snapshot.get_stem().voxels_position()))

        for i, leaf in enumerate(snapshot.leaves):

            if leaf.info['pm_label'] == 'mature_leaf':

                # 1 - classic determination of leaf insertion (polyline)

                h = phm3d_to_px2d(leaf.real_longest_polyline()[0], sf, angle=60)[0][1]

                # 2 - new method : determination of leaf insertion from voxels

                h_vx = voxel_insertion(leaf, stem_pl)
                if h_vx is None:
                    h_vx = h
                else:
                    h_vx = phm3d_to_px2d(h_vx, sf, angle=60)[0][1]

                df.loc[df.shape[0]] = [plant.plantid, snapshot.metainfo.timestamp, leaf.info['pm_leaf_number'],
                                       pred[i], h, h_vx]






# ===================== visualize results of h_pl vs h_vx comparison.

df = pd.read_csv('data/rgb_insertion_profile_annotation/pl_vx_comparison.csv')
df_val = pd.DataFrame(columns=['h_pl', 'h_vx', 'obs'])

for plantid in list(df['plantid'].unique()):

    df2 = df[df['plantid'] == plantid]

    #df2 = df2[~df2['h_vx'].isna()]
    #plt.plot(df2['timestamp'], 2448 - df2['h'], 'k*', markersize=2)
    #plt.plot(df2['timestamp'], 2448 - df2['h_vx'], 'r*', markersize=2)

    # ========================== median height per rank

    #plt.figure(plantid)

    medians_h = []
    medians_h_vx = []
    ranks = np.array(sorted([r for r in df2['pred'].unique() if r != -1]))

    for r in ranks:
        dfr = df2[df2['pred'] == r]
        plt.plot([r + 1] * len(dfr.sort_values('timestamp')[3:15]['h']), 2448 - dfr.sort_values('timestamp')[3:15]['h'], 'k*', markersize=2)
        medians_h.append(np.median(dfr.sort_values('timestamp')[3:15]['h']))
        plt.plot([r + 1] * len(dfr.sort_values('timestamp')[3:15]['h_vx']), 2448 - dfr.sort_values('timestamp')[3:15]['h_vx'], 'r*', markersize=2)
        medians_h_vx.append(np.median(dfr.sort_values('timestamp')[3:15]['h_vx']))
    plt.plot(ranks + 1, 2448 - np.array(medians_h), 'k-')
    plt.plot(ranks + 1, 2448 - np.array(medians_h_vx), 'r-')

    # =========================== compare with annotation

    #with open('data/rgb_leaf_annotation/{}.json'.format(plantid), 'r') as file:
    #    anot = json.loads(file.read())
    #anot_heights = []
    #ranks = []
    #for name, obj in anot.items():
    #    task = name.split('_t')[1].split('_a')[0]
    #    rank = int(name[name.find('_r') + len('.r'):name.rfind('.png')])
    #    for shape in obj['regions']:
    #        h = shape['shape_attributes']['all_points_y'][-1]
    #        anot_heights.append(h)
    #        ranks.append(rank)

    path = 'data/rgb_insertion_profile_annotation/annotation/'
    anot_files = [path + p for p in os.listdir(path)]
    df_anot = pd.DataFrame(columns=['plantid', 'y'])
    for anot_file in anot_files:
        with open(anot_file) as file:
            anot = json.load(file)
        for name in anot.keys():
            id = int(name.split('id')[1].split('_')[0])
            regions = anot[name]['regions']
            for region in regions:
                x, y = region['shape_attributes']['cx'], region['shape_attributes']['cy']
                df_anot.loc[df_anot.shape[0]] = id, 2448 - y

    dfid = df_anot[df_anot['plantid'] == plantid]

    n = min([len(medians_h), len(medians_h_vx), len(dfid['y'])])
    for h_pl, h_vx, obs in zip(medians_h[:n], medians_h_vx[:n], list(sorted(dfid['y']))[:n]):
        df_val.loc[df_val.shape[0]] = [h_pl, h_vx, obs]

    plt.plot(1 + np.arange(len(dfid)), sorted(dfid['y']), 'g-')


plt.figure('polyline')
plt.plot([0, 2500], [0, 2500], 'r-')
plt.plot(2448 - df_val['obs'], df_val['h_pl'], 'ko', markersize=2)
plt.figure('voxel')
plt.plot([0, 2500], [0, 2500], 'r-')
plt.plot(2448 - df_val['obs'], df_val['h_vx'], 'ko', markersize=2)


