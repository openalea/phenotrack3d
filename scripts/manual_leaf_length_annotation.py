"""
Script to ecxtract/prepare data  for annotating leaf lengths on images with VGG annotator
"""

from openalea.maizetrack.utils import best_leaf_angles, get_rgb
from openalea.maizetrack.local_cache import metainfos_to_paths, get_metainfos_ZA17, load_plant, check_existence
from openalea.maizetrack.utils import phm3d_to_px2d, shooting_frame_conversion
from openalea.maizetrack.trackedPlant import TrackedPlant
from openalea.maizetrack.stem_correction import stem_height_smoothing
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import os
import json
from skimage import io
import cv2
import matplotlib.pyplot as plt


# ===== download images to annotate ===========================================================================

folder = 'local_cache/cache_ZA17/segmentation_voxel4_tol1_notop_vis4_minpix100_stem_smooth_tracking'
all_files = [folder + '/' + rep + '/' + f for rep in os.listdir(folder) for f in os.listdir(folder + '/' + rep)]
plantids = [330, 348, 832, 931, 940, 1270, 1276, 1301, 1383, 1424]

RANKS = [6, 9]
saving_folder = 'data/rgb_leaf_growing_annotation/rgb/'
# benoit tagged result copied to Z:\lepseBinaries\TaggedImages\ARCH2017-03-30

for plantid in plantids:

    print(plantid)

    # load plant metainfos
    metainfos = get_metainfos_ZA17(plantid)

    # load leaf rank annotations (via trackedplant)
    paths = metainfos_to_paths(metainfos, folder=folder)
    metainfos2, paths = check_existence(metainfos, paths)
    vmsi_list = load_plant(metainfos2, paths)
    plant = TrackedPlant.load_and_check(vmsi_list)
    plant.load_rank_annotation()

    # load stem height

    dfs = []
    folder_stem = 'local_cache/cache_ZA17/collars_voxel4_tol1_notop_vis4_minpix100'
    all_files_stem = [folder_stem + '/' + rep + '/' + f for rep in os.listdir(folder_stem) for f in os.listdir(folder_stem + '/' + rep)]
    plantid_files = [f for f in all_files_stem if int(f.split('/')[-1][:4]) == plantid]
    for f in plantid_files:
        task = int(f.split('_')[-1].split('.csv')[0])
        timestamp = next(m for m in metainfos if m.task == task).timestamp
        #timestamp = int(f.split('_')[-1].split('.csv')[0]) # ZB14
        dft = pd.read_csv(f)
        dft['t'] = timestamp
        dfs.append(dft)
    df = pd.concat(dfs)
    df = df[df['score'] > 0.95]
    df['y'] = 2448 - df['y']
    df_stem = df.sort_values('y', ascending=False).drop_duplicates(['t']).sort_values('t')
    f_stem = stem_height_smoothing(np.array(df_stem['t']), np.array(df_stem['y']))


    # select leaves and keep it in a dataframe:

    df = pd.DataFrame(columns=['s', 'l', 'rank', 'timestamp'])
    for s, snapshot in enumerate(plant.snapshots):
        for l, leaf in enumerate(snapshot.leaves):
            if leaf.info['pm_label'] == 'growing_leaf' and leaf.rank_annotation + 1 in RANKS:
                timestamp = snapshot.metainfo.timestamp
                df.loc[df.shape[0]] = [s, l, snapshot.rank_annotation[l] + 1, timestamp]

    # best camera angle for each leaf:

    df['camera_angle'] = 0
    for index, row in df.iterrows():
        snapshot = plant.snapshots[row['s']]
        leaf = snapshot.leaves[row['l']]
        df.loc[index, 'camera_angle'] = best_leaf_angles(leaf, snapshot.metainfo['shooting_frame'], n=1)[0]

    # load images (1 per leaf !):

    for _, row in df.iterrows():

        snapshot = plant.snapshots[row['s']]
        leaf = snapshot.leaves[row['l']]

        image, _ = get_rgb(metainfo=snapshot.metainfo, angle=row['camera_angle'], save=False,
                              main_folder='rgb')

        # image name
        image_name = 'id{}_t{}_r{}.png'.format(plantid, row['timestamp'], row['rank'])

        # display line to show stem height
        h = int(2448 - f_stem(snapshot.metainfo.timestamp))
        image = cv2.line(np.float32(image), (0, h), (2048 - 1, h), (0, 0, 0), 1, lineType=cv2.LINE_4)

        # display leaf to annotate
        pl = leaf.real_longest_polyline()
        pl = phm3d_to_px2d(pl, snapshot.metainfo['shooting_frame'], row['camera_angle'])
        points = [pl[i] for i in np.linspace(0, len(pl) - 1, 10).astype(int)]
        for x, y in points:
            image = cv2.circle(np.float32(image), (int(x), int(y)), 1, (255, 0, 0), -1)

        io.imsave(saving_folder + image_name, image.astype(np.uint8))


# ===== read annotation ===========================================================================================

# load json anot
with open('data/rgb_leaf_growing_annotation/leaf_growing_annotation.json') as f:
    anot = json.load(f)

# convert it to dataframe
df_anot = []
for _, item in anot.items():
    # extract metainfo and leaf rank
    name = item['filename']
    timestamp = int(name.split('t')[1].split('_')[0])
    rank = int(name.split('r')[1].split('.')[0])
    plantid = int(name.split('id')[1].split('_')[0])

    if item['regions'] != []:

        # extract polyline
        shape = item['regions'][0]['shape_attributes']
        x, y = shape['all_points_x'], shape['all_points_y']
        pl = np.array([x, y]).T
        length = np.sum([np.linalg.norm(np.array(pl[k]) - np.array(pl[k + 1])) for k in range(len(pl) - 1)])

        df_anot.append([plantid, rank, timestamp, length])

df_anot = pd.DataFrame(df_anot, columns=['plantid', 'rank', 'timestamp', 'length_px'])

# load metainfos
metainfos_dict = {}
for plantid in df_anot['plantid'].unique():
    metainfos = get_metainfos_ZA17(plantid)
    metainfos_dict[plantid] = metainfos

# convert pixel lengths to mm lengths
lengths_mm = []
for _, row in df_anot.iterrows():

    metainfos = metainfos_dict[row['plantid']]
    metainfo = next(m for m in metainfos if m.timestamp == row['timestamp'])
    conversion_factor, pot = shooting_frame_conversion(metainfo.shooting_frame)
    lengths_mm.append(row['length_px'] * conversion_factor)

df_anot['length_mm'] = lengths_mm

#df_anot.to_csv('data/rgb_leaf_growing_annotation/leaf_growing_annotation.csv', index=False)


# ===== is image annotation accurate ?  == > plot direct plant anot VS image anot =======================================

# Llorenc annotation
l_anot = pd.read_csv('data/ARCH2017-03-30_LMA.csv', sep=';')

# Image annotation
with open('data/rgb_manual_annotation/vgg_annotation_v2.json') as f:
    d = json.load(f)

# Other info for each leaf : px/mm conversion, ...
with open('data/rgb_manual_annotation/phm_leaf_info.json') as f:
    phm_leaf_info = json.load(f)

plantids = []
anot_lengths = []
for key in d.keys():

    name = d[key]['filename']
    plantid = int(name[name.find('plantid') + len('plantid'):name.rfind('_d2017')])

    if d[key]['regions'] != []:

        shape = d[key]['regions'][0]['shape_attributes']
        x, y = shape['all_points_x'], shape['all_points_y']
        pl = np.array([x, y]).T

        length = np.sum([np.linalg.norm(np.array(pl[k]) - np.array(pl[k + 1])) for k in range(len(pl) - 1)])

        length *= phm_leaf_info[str(plantid)]['px_mm_ratio']

        # angle correction
        correct_angle = False
        if correct_angle:
            info = phm_leaf_info[str(plantid)]
            a1, a2 = info['azimuth'], info['cam_angle']
            da = abs(((a1 / 180 - a2 / 180) + 1) % 2 - 1)
            da = min(da, 1 - da) * 180  # 0 - > 90 deg
            cos_ = np.cos(da / 90 * np.pi / 2)

            length /= cos_
            print(plantid, round(da, 3), round(cos_, 3), round(length, 2))


        plantids.append(plantid)
        anot_lengths.append(length / 10)

real_lengths = []
for plantid, anot_length in zip(plantids, anot_lengths):

    row = l_anot[(l_anot['plantid'] == plantid) & (l_anot['observationcode'] == 'Havested_leaf_length_cm')]
    real_length = float(list(row['observation'])[0].replace(',', '.'))
    real_lengths.append(real_length)

    print(plantid, round(real_length - anot_length, 2))


x, y = np.array(real_lengths), np.array(anot_lengths)

fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.plot(x, y, 'k*')
plt.xlabel('Leaf length measured directly on the plant (cm)', fontsize=25)
plt.ylabel('leaf length measured on an image (cm)', fontsize=25)
a, b = min([min(x), min(y)]), max([max(x), max(y)])
#plt.plot([a, b], [a, b])
plt.plot([90, 130], [90, 130], color='grey')

bias = np.mean(x - y)
rmse = np.sqrt(np.sum((x - y) ** 2) / len(x))
r2 = r2_score(x, y)

ax.text(0.65, 0.19, 'n = {} \nBias = {} cm \nRMSE = {} cm\nR² = {} cm'.format(
    len(x), round(bias, 2), round(rmse, 2), round(r2, 2)), transform=ax.transAxes,
        fontsize=20,
        verticalalignment='top')



