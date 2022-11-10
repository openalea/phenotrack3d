import os
import numpy as np
import matplotlib.pyplot as plt

from openalea.phenomenal import object as phm_obj
import openalea.maizetrack.phenomenal_display as phm_display

from alinea.phenoarch.cache import Cache
from alinea.phenoarch.platform_resources import get_ressources
from alinea.phenoarch.meta_data import plant_data
from PIL import Image

import cv2
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

sf_col = {'4958': 'r', '4960': 'g', '4961': 'b'}

if exp == 'ZA22':
    plants = [p for p in my_plants if int(p.split('/')[0]) not in [1580, 2045]]

""""
LIST OF BUGS

ZA22
 - 2156: trop grand dt ==> critère rang absolu fait sauter un rang

TODO
- calcul width phm
- phm ratio length / width F1 vs F2

"""

# ===== ZA20 ====================================================================================================

# ZA20 : filter plants
plants = list(index.plant_index[index.plant_index['rep'].str.contains('EPPN')]['plant'])
plantid_moche = [434]

for plant in plants[::10]:
    meta_snapshots = index.get_snapshots(index.filter(plant=plant), meta=True)
    # plt.plot([m.timestamp for m in meta_snapshots], [meta_snapshots[0].pot] * len(meta_snapshots), 'k.-')

    # remove late last image
    meta_snapshots = [m for m in meta_snapshots if m.timestamp < 1588000000]

    m, angle = meta_snapshots[-10], 60
    rgb_path = next(p for (v, a, p) in zip(m.view_type, m.camera_angle, m.path) if v == 'side' and a == angle)
    rgb = cv2.cvtColor(cache.image_client.imread(rgb_path), cv2.COLOR_BGR2RGB)
    plt.figure('{}_{}'.format(m.pot, m.genotype))
    plt.imshow(rgb)

# stem : all plants
path = 'X:\phenoarch_cache\cache_ZA22\deepcollar\collar-temporal_voxel4_tol1_notop_pot_vis4_minpix100'
for f in list(np.random.permutation(os.listdir(path))):
    plantid = int(f.split('_')[0])
    plant = next(p for p in plants if int(p.split('/')[0]) == int(plantid))
    meta_snapshots = index.get_snapshots(index.filter(plant=plant), meta=True)

    collars = cache.load_collar_temporal(meta_snapshots[0])
    # plt.figure(meta_snapshots[0].pot)
    # plt.plot(collars['timestamp'], collars['z_3d'], 'k.')
    gb = collars.groupby('task')['timestamp', 'z_stem'].mean().reset_index()
    col = {'WW': 'blue', 'WD1': 'orange', 'WD2': 'red'}[meta_snapshots[0].scenario]
    plt.plot(gb['timestamp'], gb['z_stem'], '-', color=col, alpha=0.5, linewidth=0.7)

# stem : per genotype
s_index = index.plant_index[index.plant_index['plant'].isin(plants)]
genotype = 'EPPN7_L'
plants = s_index[s_index['genotype'] == genotype]['plant']
for plant in plants:
    meta_snapshots = index.get_snapshots(index.filter(plant=plant), meta=True)
    collars = cache.load_collar_temporal(meta_snapshots[0])
    gb = collars.groupby('task')['timestamp', 'z_stem'].mean().reset_index()
    col = {'WW': 'blue', 'WD1': 'orange', 'WD2': 'red'}[meta_snapshots[0].scenario]
    plt.plot(gb['timestamp'], gb['z_stem'], '-', color=col, alpha=0.5, linewidth=0.7)

# pheno data and tiller
pheno = pd.read_csv('data/pheno_ZA20.csv')
pheno = pheno[pheno['plantid'].isin([int(p.split('/')[0]) for p in plants])]
plantids_tiller = np.array(sorted(pheno[pheno['observationcode'] == 'tiller_number']['plantid'].unique()))

# ===== test pot rotation =======================================================================================

# explore
fd = 'X:/phenoarch_cache/cache_ZA20/pot_rotation/'
df = []
for f in os.listdir(fd):
    dfi = pd.read_csv(fd + f)
    rotations = dfi['rotation']
    for k in range(len(rotations) - 1):
        a1, a2 = rotations[[k, k + 1]]
        q_test = [-3, -2, -1, 0, 1, 2, 3]
        q = q_test[np.argmin([np.abs(a1 - (a2 + 360 * q)) for q in q_test])]
        # a2_corrected = a2 if np.abs(a2 - a1) < np.abs((a2 - np.sign(a2) * 360) - a1) % 360 else a2 - np.sign(a2) * 360
        rotations[k + 1] = a2 + 360 * q
    print(f, np.max(np.diff(rotations)))
    plt.plot(rotations, 'k.-')
    df.append([int(f.split('.')[0]), np.max(rotations) - np.min(rotations)])
df = pd.DataFrame(df, columns=['plantid', 'dr'])



from openalea.phenomenal.calibration import CalibrationFrame
from alinea.phenoarch.reconstruction import world_transform

plantid = 1
plant = next(p for p in plants if int(p.split('/')[0]) == plantid)
meta_snapshots = index.get_snapshots(index.filter(plant=plant, nview=13), meta=True)
meta_snapshots = [m for m in meta_snapshots if '2020-02-14' <= m.daydate <= '2020-04-01']


df_rotation = pd.read_csv('data/ZA20/pot_rotation.csv')
s_rotation = df_rotation[(df_rotation['plantid'] == plantid)]

# before rotation correction
segments = []
for m in meta_snapshots[5:-1]:
    print(m.task)
    sk = cache.load_voxelskeleton(m)
    segments += sk.segments
global_sk = phm_obj.VoxelSkeleton(segments=segments, voxels_size=4)
phm_display.plot_sk(global_sk)

# after rotation correction
segments2 = []
xyz_list = []
for m in meta_snapshots[5:-1]:
    if m.task in list(s_rotation['task']):

        pot_angle = s_rotation[s_rotation['task'] == m.task]['rotation'].iloc[0]
        print(m.task, pot_angle)
        # pot_angle=60
        pot_angle_rad = np.radians(pot_angle)
        sk = cache.load_voxelskeleton(m)

        calibration = cache.load_calibration(m['shooting_frame'])
        frame = calibration.get_frame(frame='pot')
        pot_frame = calibration.frames['pot']
        plant_frame = CalibrationFrame.from_tuple((0, 0, pot_frame._pos_z, 0, 0, pot_frame._rot_z - pot_angle_rad))

        new_frame = plant_frame.get_frame()

        for segment in sk.segments:
            points = frame.global_point(np.array(segment.polyline))
            new_points = new_frame.local_point(points)
            segments2.append(phm_obj.VoxelSegment(polyline=new_points, voxels_position=4, closest_nodes=None))

        vx = cache.load_voxelgrid(m)
        xyz_list.append(new_frame.local_point(vx.voxels_position))

global_sk2 = phm_obj.VoxelSkeleton(segments=segments2, voxels_size=4)
phm_display.plot_sk(global_sk2)

# vx2 = world_transform(vx, calibration, new_world_frame_name='plant_frame', new_world_frame=plant_frame.get_frame())

# ===== outputs visualisation ===================================================================================

"""
- Pot rotation (ex: plantid 1, voir ZA20/rotation)
- Binarisation bof (ex: plantid 1, 02-21, manque une bonne partie dans la reconstruction)
- Reajustement manuel F1 vs F2 necessaire, a moins d'avoir des meilleurs binaires ?

plantid 78 : tracking mauvais, a cause d'une insertion
plantid 36 : bizarre, devrait pas rater
plantid 199: decalage qui pourrait être eviter

TODO
-test alignement a partir d'un t au milieu (~4-5 f ligulé?)
-surveiller alignement pour : 199, 438 (deletion ?), 445
"""

"""
50 plants examined manually :
[1, 3, 20, 28, 36, 40, 55, 59, 60, 63, 72, 78, 83, 84,  90,  93,  98, 119, 146, 154, 157, 189, 197, 199, 207, 220, 
228, 236, 239, 257, 280, 282, 283, 284, 288, 295, 300, 311, 319, 364, 373, 387, 403, 434, 438, 440, 445, 455, 456, 460]
"""

plantid = 176
plant = next(p for p in plants if int(p.split('/')[0]) == plantid)
print(plant)
meta_snapshots = index.get_snapshots(index.filter(plant=plant, nview=13), meta=True)
tracking = cache.load_tracking(meta_snapshots[0])

# ===== collars + tracking

collars = cache.load_collar_temporal(meta_snapshots[0])
tr = tracking[tracking['mature']]

plt.figure(meta_snapshots[0].pot)
plt.plot(collars['timestamp'], collars['z_3d'], 'k.')
for r in tr['rank_tracking'].unique():
    s = tr[tr['rank_tracking'] == r]
    if r != 0:
        plt.plot(s['timestamp'], s['h'], 'o', color=np.array(phm_display.PALETTE)[r - 1] / 255.)
    else:
        plt.plot(s['timestamp'], s['h'], 'ko', fillstyle='none', markersize=10)

# ===== image (+ seg)

m, angle = meta_snapshots[8], 60

for k, m in enumerate(meta_snapshots):

    rgb_path = next(p for (v, a, p) in zip(m.view_type, m.camera_angle, m.path) if v == 'side' and a == angle)
    print(os.path.isfile('Z:/' + rgb_path))
    rgb = cv2.cvtColor(cache.image_client.imread(rgb_path), cv2.COLOR_BGR2RGB)
    # rgb = cv2.resize(rgb, tuple((np.array(rgb.shape)[[1, 0]] / 4).astype(int)))
    plt.figure(m.daydate)
    plt.imshow(rgb)
    plt.imsave('data/ZA20/rotation/{}_{}_{}_{}.png'.format(m.pot, k, angle, m.daydate), rgb)

    seg = cache.load_segmentation(m)
    projection = cache.load_calibration(m.shooting_frame).get_projection(id_camera='side', rotation=angle,
                                                       world_frame=cache.parameters['reconstruction']['frame'])
    pl = projection(np.array(seg.get_stem().get_highest_polyline().polyline))
    plt.plot(pl[:, 0], pl[:, 1], 'r-')
    for leaf in seg.get_leafs():
        pl = projection(np.array(leaf.real_longest_polyline()))
        plt.plot(pl[:, 0], pl[:, 1], '-', color=('b' if leaf.info['pm_label'] == 'mature_leaf' else 'orange'))

    plt.savefig('data/ZA20/{}.png'.format(k))

# ===== 3D single t

for angle in [k * 30 for k in range(12)]:
    bin_path = next(p for (v, a, p) in zip(m.view_type, m.camera_angle, m.binary_path) if v == 'side' and a == angle)
    bin = cv2.cvtColor(cache.image_client.imread('Y:/lepseBinaries/' + bin_path), cv2.COLOR_BGR2RGB)
    plt.figure(angle)
    plt.imshow(bin)

phm_display.plot_vg(cache.load_voxelgrid(m))
phm_display.plot_sk(cache.load_voxelskeleton(m))
phm_display.plot_vmsi([cache.load_segmentation(m)])

sk = cache.load_voxelskeleton(m)
pos = sk.position()

# ====== 3D all t

phm_display.plot_vmsi([cache.load_segmentation(m) for m in meta_snapshots if m.task in tracking['task'].unique()])

phm_display.plot_vmsi([cache.load_segmentation(m) for m in meta_snapshots if m.task in tracking['task'].unique()],
                      only_mature=True)

segs = {m.task: cache.load_segmentation(m) for m in meta_snapshots if m.task in tracking['task'].unique()}
tr = tracking[tracking['mature']]
leaves, ranks = [], []
for _, row in tr.iterrows():
    # if 1584500000 < row.timestamp < 1584600000:
    leaves.append(segs[row.task].get_leaf_order(row.rank_phenomenal))
    ranks.append(row.rank_tracking - 1)
phm_display.plot_leaves(leaves, ranks)


# ===== 3D reconstruction over time

xyz_list = [np.array(cache.load_voxelgrid(m).voxels_position) for m in meta_snapshots[5:-1]]
for k, xyz in enumerate(xyz_list[::1]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=40)
    for i, f in enumerate([ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        f((np.min(np.concatenate(xyz_list), axis=0)[i], np.max(np.concatenate(xyz_list), axis=0)[i]))
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], '.', color='grey', markersize=1)
    plt.savefig('data/ZA20/rotation_112/after_{}.png'.format(k))
    plt.close('all')

# ===== stem smooth =============================================================================================

for plant in my_plants:
    meta_snapshots = index.get_snapshots(index.filter(plant=plant), meta=True)

    try:
        plt.figure()
        collars = cache.load_collar_temporal(meta_snapshots[0])
        gb = collars.groupby('task')['timestamp', 'z_stem'].mean().reset_index()
        plt.plot(gb['timestamp'], gb['z_stem'], 'k-', alpha=0.2)
        print(plant)
    except:
        pass

# ===== collars + tracking ======================================================================================

for plant in plants:

    # plant = next(p for p in plants if int(p.split('/')[0]) == plantid)

    meta_snapshots = index.get_snapshots(index.filter(plant=plant), meta=True)

    try:
        collars = cache.load_collar_temporal(meta_snapshots[0])
        plt.figure(meta_snapshots[0].pot)
        plt.plot(collars['timestamp'], collars['z_3d'], 'k.')

        tracking = cache.load_tracking(meta_snapshots[0])
        tr = tracking[tracking['mature']]

        for task in tr['task'].unique():
            s = tr[tr['task'] == task]

        for r in tr['rank_tracking'].unique():
            s = tr[tr['rank_tracking'] == r]
            if r != 0:
                plt.plot(s['timestamp'], s['h'], 'o', color=np.array(phm_display.PALETTE)[r - 1] / 255.)
            else:
                plt.plot(s['timestamp'], s['h'], 'ko', fillstyle='none', markersize=10)
            # plt.plot([list(tr['task'].unique()).index(t) for t in s['task']], s['h'], 'o',
            #          color=phm_display.PALETTE[r - 1] / 255.)

        print(plant)

    except:
        pass

# ===== gif =======================================================================================================

plant = df_plant[df_plant['pot'] == 907].iloc[0]['plant']
query = index.filter(plant=plant)

meta_snapshots = index.get_snapshots(query, meta=True)
meta_snapshots = meta_snapshots[:-18]

imgs = []
m = meta_snapshots[50]
i_angles = [i for i, (v, a) in enumerate(zip(m.view_type, m.camera_angle)) if v != 'top']
for i_angle in i_angles:
    rgb = cv2.cvtColor(image_client.imread(m.path[i_angle]), cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (819, 979))
    imgs.append(rgb)

from PIL import Image
imgs_gif = [Image.fromarray(np.uint8(img)) for img in imgs]
fps = 5
imgs_gif[0].save('data/videos/gif_small_ppt/angle_907_{}fps.gif'.format(fps),
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

selec = res[res['t'] > 1.]
for type in selec['type'].unique():
    s = selec[selec['type'] == type]
    plt.plot(s['vx_volumne'], s['t'] / 60, '.', label=type)
plt.legend()

# ===== my_plants overview ========================================================================================

k = 0
for col, exp in zip(['b', 'g', 'r'], df_my_plants['exp'].unique()):
    cache_client, image_client, binary_image_client, calibration_client = get_ressources(exp, cache='X:',
                                                                                         studies='Z:', nasshare2='Y:')

    cache = Cache(cache_client, image_client, binary_image_client=binary_image_client,
                  calibration_client=calibration_client, parameters=parameters)
    index = cache.snapshot_index()

    for plant in df_my_plants[df_my_plants['exp'] == exp]['plant']:
        meta_snapshots = index.get_snapshots(index.filter(plant=plant), meta=True)
        if meta_snapshots:
            timestamps = np.array([m.timestamp for m in meta_snapshots])
            plt.plot(timestamps - min(timestamps), [k] * len(meta_snapshots), col + '.-')
        else:
            print(plant)
        k += 1
    k += 10








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

