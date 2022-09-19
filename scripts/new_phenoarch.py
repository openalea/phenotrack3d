import os
import numpy as np
import matplotlib.pyplot as plt

from openalea.phenomenal import object as phm_obj
from openalea.maizetrack.phenomenal_display import *

from alinea.phenoarch.cache import snapshot_index, load_collar_detection
from alinea.phenoarch.platform_resources import get_ressources
from alinea.phenoarch.meta_data import plant_data
from PIL import Image

import cv2
import pandas as pd

exp = 'ZA17'

cache_client, image_client, binary_image_client, calibration_client = get_ressources(exp, cache='X:', studies='Z:', nasshare2='Y:')
index = snapshot_index(exp, image_client=image_client, cache_client=cache_client, binary_image_client=binary_image_client)
df_plant = plant_data(exp)

VX_DIR = cache_client.cache_dir + '/cache_{}/phenomenal/image3d_voxel4_tol1_notop_pot/'.format(exp)
SK_DIR = cache_client.cache_dir + '/cache_{}/phenomenal/skeleton_voxel4_tol1_notop_pot_vis4_minpix100/'.format(exp)
COL_DIR = cache_client.cache_dir + '/cache_{}/deepcollar/collars-temporal_voxel4_tol1_notop_pot_vis4_minpix100/'.format(exp)
SEG_DIR = cache_client.cache_dir + '/cache_{}/phenomenal/segmentation_voxel4_tol1_notop_pot_vis4_minpix100_force-stem/'.format(exp)
TRACK_DIR = cache_client.cache_dir + '/cache_{}/phenotrack3d/tracking_voxel4_tol1_notop_pot_vis4_minpix100_force-stem/'.format(exp)

# # TODO
# fd = cache_client.cache_dir + '/cache_{}/deepcollar/collars-temporal_voxel4_tol1_notop_pot_vis4_minpix100/'.format(exp)
# f = pd.read_csv(fd + '401_ZM4971_CZL19058_cimmyt_EXPOSE_WW_Rep_4_07_41_ARCH2022-01-10.csv')
# fd = cache_client.cache_dir + '/cache_{}/phenomenal/segmentation_voxel4_tol1_notop_pot_vis4_minpix100_force-stem/'.format(exp)
# f = phm_obj.VoxelSegmentation.read_from_json_gz(fd + '2022-02-02/401_ZM4971_CZL19058_cimmyt_EXPOSE_WW_Rep_4_07_41_ARCH2022-01-10__2022-02-02__4977.json.gz')

df_my_plants = pd.read_csv('data/plants_set_tracking.csv')
my_plants = list(df_my_plants[df_my_plants['exp'] == exp]['plant'])

sf_col = {'4958': 'r', '4960': 'g', '4961': 'b'}

# for m in meta_snapshots:
#     m['reconstruction'] = {'voxel_size': 4, 'error_tolerance': 1, 'use_top_image': False,
#                                         'world_frame': 'pot'}
#     m['skeletonisation'] = {'required_visible': 4, 'nb_min_pixel': 100}
#     m['collar_detection'] = {'model_name': '3exp_xyside_99000'}

# ===== gif =======================================================================================================

plant = df_plant[df_plant['pot'] == 1424].iloc[0]['plant']
query = index.filter(plant=plant)

meta_snapshots = index.get_snapshots(query, meta=True)
meta_snapshots = meta_snapshots[:-18]

imgs = []
for m in meta_snapshots:
    i_angle = next(i for i, (v, a) in enumerate(zip(m.view_type, m.camera_angle)) if v == 'top')
    rgb = cv2.cvtColor(image_client.imread(m.path[i_angle]), cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (819, 979))
    imgs.append(rgb)

from PIL import Image
imgs_gif = [Image.fromarray(np.uint8(img)) for img in imgs]
fps = 5
imgs_gif[0].save('data/videos/gif_small_ppt/top_{}fps.gif'.format(fps),
              save_all=True,
              append_images=imgs_gif[1:],
              optimize=True,
              duration=1000/fps,
              loop=0)

# ===== check =====================================================================================================

# time
path = cache_client.cache_dir + '/cache_ZA22/check_time/vx4/'
res = []
for f in os.listdir(path):
    df = pd.read_csv(path + f)
    df['plantid'] = int(f.split('.')[0])
    res.append(df)
res = pd.concat(res)

selec = res[res['t'] > 0.1]
for type in selec['type'].unique():
    s = selec[selec['type'] == type]
    plt.plot(s['vx_volumne'], s['t'] / 60, '.', label=type)
plt.legend()

# ===== dataset overview ==========================================================================================

for plant in df_plant['plant'][::10]:
    plantid = int(plant.split('/')[0])
    query = index.filter(plant=plant)
    if not (type(query) == pd.core.frame.DataFrame):
        print('no meta : ', plant)
    else:
        meta_snapshots = index.get_snapshots(query, meta=True)
        meta_snapshots = sorted(meta_snapshots, key=lambda k: k.timestamp)
        plt.plot([m.timestamp for m in meta_snapshots], [plantid] * len(meta_snapshots), 'k.-')
for plant in my_plants:
    plantid = int(plant.split('/')[0])
    query = index.filter(plant=plant)
    meta_snapshots = index.get_snapshots(query, meta=True)
    meta_snapshots = sorted(meta_snapshots, key=lambda k: k.timestamp)
    plt.plot([m.timestamp for m in meta_snapshots], [plantid] * len(meta_snapshots), 'r*')


for dir in [VX_DIR, SK_DIR, COL_DIR, SEG_DIR]:
    files = [f for d in os.listdir(dir) for f in os.listdir(dir + d)]
    plantids = np.unique([int(f.split('_')[0]) for f in files])
    print('{} files, {} plants'.format(len(files), len(plantids)))

for i, plantid in enumerate(plantids):
    plant = df_plant[df_plant['pot'] == plantid].iloc[0]['plant']
    query = index.filter(plant=plant)
    meta_snapshots = index.get_snapshots(query, meta=True)
    meta_snapshots = sorted(meta_snapshots, key=lambda k: k.timestamp)
    plt.plot([m.timestamp for m in meta_snapshots], [i] * len(meta_snapshots), 'k.-')

    missing = []
    for m in meta_snapshots:
        sk_path = SK_DIR + '{0}/{1}__{0}__{2}.json.gz'.format(m.daydate, m.plant.replace('/', '_'), m.task)
        if not os.path.isfile(sk_path):
            print(sk_path)
            missing.append(m.timestamp)
    plt.plot(missing, [i] * len(missing), 'r*')

# =================================================================================================================

"""
plantid 437: pb collars 2D -> 3D  ====> OK ! 
plantid 2305 : F1 sous pot, F2 10px au dessus pot. dur a differencier
plantid 1953 : environ 30px de dif entre 2 cols vers le haut

plantid 731 : croissance très particulière de la tige
"""

# ===== vx4 vs vx8 =============================================================================================

plantids_vx4 = [int(p[:4]) for p in os.listdir(TRACK_DIR)]
plantids_vx8 = [int(p[:4]) for p in os.listdir(TRACK_DIR.replace('voxel4', 'voxel8'))]
plants = [p for p in my_plants if int(p[:4]) in plantids_vx4 and int(p[:4]) in plantids_vx8]

# ===== seg / tracking =========================================================================================

for plant in my_plants:

    # plant = df_plant[df_plant['pot'] == plantid].iloc[0]['plant']
    query = index.filter(plant=plant)
    meta_snapshots = index.get_snapshots(query, meta=True)
    meta_snapshots = sorted(meta_snapshots, key=lambda k: k.timestamp)

    col_path = COL_DIR + '{}.csv'.format(plant.replace('/', '_'))
    if os.path.isfile(col_path):
        collars = pd.read_csv(col_path)
        plt.figure(plant)
        plt.plot(collars['timestamp'], collars['z_3d'], 'k.')

    tracking = pd.read_csv(TRACK_DIR + '{}.csv'.format(plant.replace('/', '_')))

# seg vs collar
    plt.plot(collars['timestamp'], collars['z_3d'], 'k.')
    tr = tracking[tracking['mature']]
    for r in tr['rank_tracking'].unique():
        s = tr[tr['rank_tracking'] == r]
        plt.plot(s['timestamp'], s['h'], 'o', color=PALETTE[r - 1] / 255.)
# for seg in segs:
#     leaves = seg.get_mature_leafs()
#     plt.plot([seg.info['t']] * len(leaves), [l.info['pm_z_base_voxel'] for l in leaves], 'ro')

segs = []
for m in meta_snapshots:
    try:
        seg_path = SEG_DIR + '{0}/{1}__{0}__{2}.json.gz'.format(m.daydate, m.plant.replace('/', '_'), m.task)
        seg = phm_obj.VoxelSegmentation.read_from_json_gz(seg_path)
        seg.info['t'] = m.timestamp
        segs.append(seg)
    except:
        print('error', m.daydate)

seg = segs[-2]
plot_vmsi([seg])




leaves = [l for seg in segs for l in seg.get_mature_leafs()]
ranks = [l.info['pm_leaf_number_tracking'] - 1 for l in leaves]
plot_leaves(leaves, ranks)

for seg in segs:
    leaves = seg.get_mature_leafs()
    # leaves = sorted(leaves, key=lambda k: k.info['pm_leaf_number']) # topological order
    ranks = [l.info['pm_leaf_number_tracking'] - 1 for l in leaves]
    sq = ['x' if k in ranks else '-' for k in range(20)]
    print(' '.join(sq))

# ===== collars =================================================================================================

# plt.figure('3D')
plt.plot(collars['timestamp'], collars['z_3d'], '.')

plt.figure('2D')
plt.plot(collars['timestamp'], 2448 - collars['y_2d'], 'k.')

# ===== debug collar ======================================

from scipy.interpolate import interp1d, interpn
from scipy.stats import linregress
from openalea.phenomenal.segmentation import get_highest_segment
from openalea.maizetrack.utils import simplify

collars.index = np.arange(len(collars))
collars['zbis'] = None

# task = 5041
# m = next(m for m in meta_snapshots if m.task == task)

for m in meta_snapshots:

    sk_path = SK_DIR + '{0}/{1}__{0}__{2}.json.gz'.format(m.daydate, m.plant.replace('/', '_'), m.task)
    sk = phm_obj.VoxelSkeleton.read_from_json_gz(sk_path)
    angles = [k * 30 for k in range(12)]

    if sk.segments:

        pl = get_highest_segment(sk.segments).polyline
        stem_pl_3d = simplify(pl, 30)

        # extend the stem polyline under the lowest point (straight segment "under the pot")
        # prevent some weird behaviors of interp1d
        stem_pl_3d = np.concatenate((np.array([stem_pl_3d[0] - np.array([0, 0, 9999])]), stem_pl_3d))

        calib = cache_client.load_calibration(m.shooting_frame)
        projections = {angle: calib.get_projection(id_camera='side', rotation=angle,
                                                   world_frame='pot') for angle in angles}

        for angle in angles:

            s = collars[(collars['task'] == m.task) & (collars['angle'] == angle)]

            # 2D projection of stem polyline
            f_3dto2d = projections[angle]
            stem_pl_2d = f_3dto2d(stem_pl_3d)

            # convert 2D x,y detection to 3D phenomenal height
            f_2dto3d = interp1d(stem_pl_2d[:, 1], np.array(stem_pl_3d)[:, 2], fill_value="extrapolate")

            # # special extrapolation to correct some bugs when the input value is outside known range
            # # (fill_value="extrapolate" in interp1d have weird behaviors sometimes)
            # a, b, _, _, _ = linregress(stem_pl_2d[:, 1], np.array(stem_pl_3d)[:, 2])
            # f_2dto3d = lambda k: f_interp(k) if ((min(stem_pl_2d[:, 1]) < k) & (k < max(stem_pl_2d[:, 1]))) else (a * k + b)

            y = np.array(s['y_2d'])
            z = f_2dto3d(y)

            # Y = np.linspace(1900, 2100)
            # plt.plot(Y, np.array([f_2dto3d(yi) for yi in Y]), 'k-')
            # plt.plot(stem_pl_2d[:, 1], np.array(stem_pl_3d)[:, 2], 'b*')

            collars.loc[s.index, 'zbis'] = z


s = collars[collars['score'] > 0.95]
plt.figure('2D')
plt.gca().invert_yaxis()
plt.plot(s['task'], s['y_2d'], 'k.')
plt.figure('3D')
plt.plot(s['task'], s['z_3d'], 'k.')
plt.figure('3D_new')
plt.plot(s['task'], s['zbis'], 'k.')


# ===== visu ====================================================================================================

# m = next(m for m in meta_snapshots if m.task == 4974)
m = meta_snapshots[-1]

# vx_path = VX_DIR + '{0}/{1}__{0}__{2}.csv'.format(m.daydate, m.plant.replace('/', '_'), m.task)
# vx = phm_obj.VoxelGrid.read_from_csv(vx_path)
# sk_path = SK_DIR + '{0}/{1}__{0}__{2}.json.gz'.format(m.daydate, m.plant.replace('/', '_'), m.task)
# sk = phm_obj.VoxelSkeleton.read_from_json_gz(sk_path)
col_path = COL_DIR + '{0}/{1}__{0}__{2}.csv'.format(m.daydate, m.plant.replace('/', '_'), m.task)
col = pd.read_csv(col_path)
col = col[col['score'] > 0.95]

seg_path = SEG_DIR + '{0}/{1}__{0}__{2}.json.gz'.format(m.daydate, m.plant.replace('/', '_'), m.task)
seg = phm_obj.VoxelSegmentation.read_from_json_gz(seg_path)

for angle in [k * 30 for k in range(12)]:

    i_angle = next(i for i, (v, a) in enumerate(zip(m.view_type, m.camera_angle)) if a == angle and v == 'side')
    rgb = cv2.cvtColor(cv2.imread(image_client.rootdir + m.path[i_angle]), cv2.COLOR_BGR2RGB)
    f_projection = cache_client.load_calibration(m.shooting_frame).get_projection(id_camera='side', rotation=angle,
                                                                                  world_frame='pot')

    plt.figure(angle)
    plt.imshow(rgb)

    for l in seg.get_mature_leafs():
        vx = np.array(list(l.voxels_position()))
        vx = f_projection(vx)
        plt.plot(vx[:, 0], vx[:, 1], 'b.')
        pl = np.array(l.real_longest_polyline())
        pl = f_projection(pl)
        plt.plot(pl[:, 0], pl[:, 1], 'y.-')

    # plt.ylim((2050, 1875))
    # plt.xlim((925, 1150))
    # vx_2d = f_projection(vx.voxels_position)
    # plt.plot(vx_2d[:, 0], vx_2d[:, 1], 'r.')
    col_angle = col[col['angle'] == angle]
    plt.plot(col_angle['x_2d'], col_angle['y_2d'], 'r.')

# plot rotating vx
vx_sets = {}
for angle in range(360):
    print(angle)
    f_projection = cache_client.load_calibration(m.shooting_frame).get_projection(id_camera='side', rotation=angle)
    vx_2d = f_projection(vx.voxels_position)
    vx_sets[angle] = vx_2d
# all_x = np.concatenate([v[:, 0] for v in list(vx_sets.values())])
# all_y = np.concatenate([v[:, 1] for v in list(vx_sets.values())])
# xmin, xmax, ymin, ymax = min(all_x), max(all_x), min(all_y), max(all_y)
imgs = []
for angle, vx_2d in vx_sets.items():
    print(angle)
    img = rgb * 0 + 255
    for x, y in vx_2d:
        img = cv2.circle(img, (round(x), round(y)), 2, (255, 0, 0), -1)
    imgs.append(img)
#imgs_gif = [img[200:-200, 250:-250, :] for img in imgs_gif]
imgs_gif = [Image.fromarray(np.uint8(img)) for img in imgs]
fps = 30
imgs_gif[0].save('data/videos/id{}_{}fps_vx360.gif'.format(plantid, fps),
                 save_all=True, append_images=imgs_gif[1:], optimize=True, duration=1000/fps, loop=0)

# plt.imshow(img)
# plt.figure()
# plt.xlim((xmin, xmax))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.plot(vx_2d[:, 0], - vx_2d[:, 1], 'r.')

# ===============================================================================================================

meta2, vxs, sks = [], [], []
for m in meta_snapshots:
    print(m.daydate)
    vx_path = VX_DIR + '{0}/{1}__{0}__{2}.csv'.format(m.daydate, m.plant.replace('/', '_'), m.task)
    sk_path = SK_DIR + '{0}/{1}__{0}__{2}.json.gz'.format(m.daydate, m.plant.replace('/', '_'), m.task)
    if os.path.isfile(vx_path) and os.path.isfile(sk_path):
        vxs.append(phm_obj.VoxelGrid.read_from_csv(vx_path))
        sks.append(phm_obj.VoxelSkeleton.read_from_json_gz(sk_path))
        meta2.append(m)

plt.plot([m.timestamp for m in meta2], [len(s.voxels_position()) for s in sks], 'k.')

for vx in vxs:
    if vx.voxels_position:
        zmin = np.min(np.array(vx.voxels_position)[:, 2])
        print(zmin)
    else:
        print('no vx positions')

plt.gca().set_aspect('equal', adjustable='box')
for sk, m in zip(sks, meta2):
    for seg in sk.segments:
        pl2d = f_projection(seg.polyline)
        col = sf_col[m.shooting_frame[-4:]]
        plt.plot(pl2d[:, 0], 2448 - pl2d[:, 1], col + '-')


sks = sorted(sks, key=lambda k: len(k.voxels_position()))
plt.plot([len(sk.voxels_position()) for sk in sks], 'k.-')

