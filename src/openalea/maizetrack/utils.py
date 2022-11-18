import os

import numpy as np

from alinea.phenoarch.shooting_frame import get_shooting_frame

from skimage import io

from openalea.phenomenal.calibration import Calibration


# ===== based on Phenomenal =========================================================================================

def shooting_frame_conversion(s_frame_name):
    # returns 2 parameters needed for this conversion :
    # xy pixel -> mm height

    s_frame = get_shooting_frame(s_frame_name)
    # mm / pixel ratio
    ratio = s_frame.pixel_size(convert=False)
    # z = 0
    corners = map(s_frame.cabin_frame.frame.global_point, s_frame.cabin_frame.corner_points['pot'])
    proj_side = s_frame.get_calibration('side').get_projection(0)
    pot_height = int(round(proj_side(np.vstack(corners))[:, 1].mean()))

    return ratio, pot_height


def phm3d_to_px2d(xyz, sf, angle=60):

    # xyz : a xyz array of phm 3D coordinates
    # sf : the corresponding shooting_frame

    xyz = np.array(xyz)
    if xyz.ndim == 1:
        xyz = np.array([xyz])

    if sf.startswith('ARCH'):
        # TODO : temporary
        # phenomenal function
        calib = Calibration.load('V:/lepseBinaries/Calibration/' + sf + '_calibration.json')
        f = calib.get_projection(id_camera='side', rotation=angle)
    else:
        # phenoarch function
        f = get_shooting_frame(sf).get_calibration('side').get_projection(angle)

    return f(xyz)


# ===== polyline analysis =====================================================================================

def quantile_point(pl, q):
    pl = np.array(pl)
    d = np.diff(pl, axis=0)
    segdists = np.sqrt((d ** 2).sum(axis=1))
    s = np.cumsum(segdists) / np.sum(segdists)
    s = np.concatenate((np.array([0]), s))

    try:
        i_q = next(i for i, val in enumerate(s) if val >= q)
    except StopIteration:
        i_q = len(s) - 1

    a, b = pl[i_q - 1], pl[i_q]
    q_pl = a + (b - a) * ((q - s[i_q - 1]) / (s[i_q] - s[i_q - 1]))

    return q_pl


def polyline_until_z(pl, z):
    # TODO : approximatif
    # return the polyline section starting from height z
    if np.max(np.array(pl)[:, 2]) <= z:
        i = 0
    else:
        i = next((i for i, pos in enumerate(pl) if pos[2] > z))
    return pl[i:]


def simplify(pl, n):

    if len(pl) < n:
        return pl
    else:
        return np.array([quantile_point(pl, q) for q in np.linspace(0, 1, n)])


# ===================================================================================================================

def best_leaf_angles(leaf, shooting_frame, angles=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], n=1):
    pl = leaf.real_longest_polyline()

    surfaces = dict()
    for angle in angles:
        pl2d = phm3d_to_px2d(pl, shooting_frame, angle)
        x, y = pl2d[:, 0], pl2d[:, 1]
        surfaces[angle] = (max(x) - min(x)) * (max(y) - min(y))

    sorted_surfaces = dict(sorted(surfaces.items(), key=lambda item: item[1], reverse=True))

    return list(sorted_surfaces.keys())[:n]


def get_rgb(metainfo, angle, main_folder='rgb', plant_folder=True, save=True, side=True):

    plantid = int(metainfo.plant[:4])

    if plant_folder:
        img_folder = main_folder + '/' + str(plantid)
    else:
        img_folder = main_folder
    if not os.path.isdir(img_folder) and save:
        os.mkdir(img_folder)

    if side:
        if angle == 0:
            # 0 can both correspond to side and top ! Need to select side image
            i_angles = [i for i, a in enumerate(metainfo.camera_angle) if a == angle]
            i_angle = [i for i in i_angles if metainfo.view_type[i] == 'side'][0]
        else:
            i_angle = metainfo.camera_angle.index(angle)
    else:
        i_angles = [i for i, a in enumerate(metainfo.camera_angle) if a == 0]
        i_angle = [i for i in i_angles if metainfo.view_type[i] == 'top'][0]

    img_name = 'id{}_d{}_t{}_a{}.png'.format(plantid, metainfo.daydate, metainfo.task, angle)
    path = img_folder + '/' + img_name

    if img_name in os.listdir(img_folder):
        img = io.imread(path)
    else:
        url = metainfo.path_http[i_angle]
        img = io.imread(url)[:, :, :3]

        if save:
            io.imsave(path, img)

    return img, path


def missing_data(vmsi):

    # Check if there are some missing data in vmsi.

    missing = False

    stem_needed_info = ['pm_z_base', 'pm_z_tip']
    if not all([k in vmsi.get_stem().info for k in stem_needed_info]):
        missing = True

    leaf_needed_info = ['pm_position_base', 'pm_z_tip', 'pm_label', 'pm_azimuth_angle', 'pm_length']
    for leaf in vmsi.get_leafs():
        if not all([k in leaf.info for k in leaf_needed_info]):
            missing = True

    return missing

# ===== used for parameter tuning (sequence alignment) ================================================================


def dataset_mean_distance(w_h=0.03, w_l=0.004, step=1):
    """
    mean distance between consecutive leaves (spatially) in a small dataset.
    file leaf_vectors.npy generated using 30 random plants (available in modulor local_benoit)
    w_h=0.03, w_l=0.004 => d = 4.23
    """
    v = np.load('leaf_vectors.npy', allow_pickle=True)
    dists = []
    for vecs in v:
        #vecs2 = np.array([[np.cos(a/360*2*np.pi), np.sin(a/360*2*np.pi), w_h * h] for h, _, a in vecs])
        vecs2 = np.array([[np.cos(a / 360 * 2 * np.pi), np.sin(a / 360 * 2 * np.pi), w_h * h, w_l * l] for h, l, a in vecs])
        dists += [np.linalg.norm(vecs2[k] - vecs2[k + step]) for k in range(len(vecs2) - step)]
    return np.mean(dists)