import os
import numpy as np
import matplotlib.pyplot as plt

from openalea.phenomenal import object as phm_obj
from openalea.maizetrack.phenomenal_display import plot_vg, plot_sk

from alinea.phenoarch.cache import snapshot_index
from alinea.phenoarch.platform_resources import get_ressources
from alinea.phenoarch.meta_data import plant_data
from PIL import Image

import cv2
import pandas as pd

exp = 'ZA22'

cache_client, image_client, binary_image_client = get_ressources(exp, cache='X:', studies='Z:', nasshare2='Y:')
index = snapshot_index(exp, image_client=image_client, cache_client=cache_client, binary_image_client=binary_image_client)
df_plant = plant_data(exp)

VX_DIR = cache_client.cache_dir + '/cache_{}/image3d_voxel4_tol1_notop_pot/'.format(exp)
SK_DIR = cache_client.cache_dir + '/cache_{}/skeleton_voxel4_tol1_notop_pot_vis4_minpix100/'.format(exp)
COL_DIR = cache_client.cache_dir + '/cache_{}/collars_voxel4_tol1_notop_pot_vis4_minpix100/'.format(exp)

sf_col = {'4958': 'r', '4960': 'g', '4961': 'b'}

# =================================================================================================================

"""
plantid 437: pb collars 2D -> 3D  ====> OK ! 
plantid 2305 : F1 sous pot, F2 10px au dessus pot. dur a differencier
plantid 1953 : environ 30px de dif entre 2 cols vers le haut
"""

# ===== collars =================================================================================================

plantids = np.unique([int(f.split('_')[0]) for d in os.listdir(COL_DIR) for f in os.listdir(COL_DIR + d)])

plantid = plantids[0]

plant = df_plant[df_plant['pot'] == plantid].iloc[0]['plant']
query = index.filter(plant=plant)
meta_snapshots = index.get_snapshots(query, meta=True)
meta_snapshots = sorted(meta_snapshots, key=lambda k: k.timestamp)

collars = []
for m in meta_snapshots:
    col_path = COL_DIR + '{0}/{1}__{0}__{2}.csv'.format(m.daydate, m.plant.replace('/', '_'), m.task)
    if os.path.isfile(col_path):
        df = pd.read_csv(col_path)
        df['t'] = m.timestamp
        df['sf'] = m.shooting_frame
        df['task'] = m.task
        collars.append(df)
collars = pd.concat(collars)

s = collars[collars['score'] > 0.95]

# plt.figure('3D')
plt.plot(s['t'], s['z_3d'], '.')

plt.figure('2D')
plt.plot(s['task'], 2448 - s['y_2d'], 'k.')

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

m = next(m for m in meta_snapshots if m.task == 4974)

# vx_path = VX_DIR + '{0}/{1}__{0}__{2}.csv'.format(m.daydate, m.plant.replace('/', '_'), m.task)
# vx = phm_obj.VoxelGrid.read_from_csv(vx_path)
# sk_path = SK_DIR + '{0}/{1}__{0}__{2}.json.gz'.format(m.daydate, m.plant.replace('/', '_'), m.task)
# sk = phm_obj.VoxelSkeleton.read_from_json_gz(sk_path)
col_path = COL_DIR + '{0}/{1}__{0}__{2}.csv'.format(m.daydate, m.plant.replace('/', '_'), m.task)
col = pd.read_csv(col_path)
col = col[col['score'] > 0.95]

for angle in [k * 30 for k in range(12)]:

    i_angle = next(i for i, (v, a) in enumerate(zip(m.view_type, m.camera_angle)) if a == angle and v == 'side')
    rgb = cv2.cvtColor(cv2.imread(image_client.rootdir + m.path[i_angle]), cv2.COLOR_BGR2RGB)

    f_projection = cache_client.load_calibration(m.shooting_frame).get_projection(id_camera='side', rotation=angle,
                                                                                  world_frame='pot')

    plt.figure(angle)
    plt.imshow(rgb)
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

