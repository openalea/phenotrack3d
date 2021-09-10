import cv2
import numpy as np

from alinea.phenoarch.shooting_frame import get_shooting_frame

################
# NOT USED #####
################
from openalea.phenotracking.maize_track.phenomenal_display import PALETTE


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


def sigmoid(x0, k, x, min_, max_):
    y = 1 / (1 + np.exp(-k * (x - x0)))
    y = (y - np.min(y)) / (np.max(y) - np.min(y)) # [0,1]
    y =  y*(max_ - min_) + min_ # [min_, max_]
    return y


def fit_sigmoid(params, x, y):
    x0, k, min_ = params
    y2 = sigmoid(x0, k, x, min_, np.max(y))
    rmse = np.sqrt(np.mean((y - y2) ** 2))
    return rmse


##############
##### USED ###
##############

##### based on phenomenal #####

def phm3d_to_px2d(xyz, sf, angle=60):

    # xyz : a xyz array of phm 3D coordinates
    # sf : the corresponding shooting_frame

    xyz = np.array(xyz)
    if xyz.ndim == 1:
        xyz = np.array([xyz])

    f = get_shooting_frame(sf).get_calibration('side').get_projection(angle)
    return f(xyz)

def rgb_and_polylines(snapshot, angle, selected=None, ranks=None):
    # snapshot : a vmsi or TrackedSnapshot object. A metainfo attribute need to be attached to this object.

    # rgb image
    img = snapshot.image[angle]

    # adding polylines
    polylines_px = [phm3d_to_px2d(leaf.real_pl, snapshot.metainfo.shooting_frame, angle) for leaf in snapshot.leaves]
    matures = [leaf.info['pm_label'] == 'mature_leaf' for leaf in snapshot.leaves]

    if ranks is None:
        ranks = snapshot.rank_annotation

    # plot leaves
    for i, (pl, c, mature) in enumerate(zip(polylines_px, ranks, matures)):

        col = [int(x) for x in PALETTE[c]]

        # unknown rank vs known rank
        if c == -1:
            leaf_border_col = (0, 0, 0)
        else:
            leaf_border_col = (255, 255, 255)

        # selected vs non selected
        ds = 1
        if selected == i:
            ds = 3

        img = cv2.polylines(np.float32(img), [pl.astype(int).reshape((-1, 1, 2))], False, leaf_border_col, 10 * ds)
        img = cv2.polylines(np.float32(img), [pl.astype(int).reshape((-1, 1, 2))], False, col, 7 * ds)

        # tip if mature
        if mature:
            pos = (int(pl[-1][0]), int(pl[-1][1]))
            img = cv2.circle(np.float32(img), pos, 20, (0, 0, 0), -1)

        # rank number
        pos = (int(pl[-1][0]), int(pl[-1][1]))
        img = cv2.putText(img, str(c + 1), pos, cv2.FONT_HERSHEY_SIMPLEX,
               3, (0, 0, 0), 4, cv2.LINE_AA)

    # write date
    id = str(int(snapshot.metainfo.plant[:4]))
    text = 'plantid {} / task {} ({})'.format(id, snapshot.metainfo.task, snapshot.metainfo.daydate)
    cv2.putText(img, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10, cv2.LINE_AA)

    return img, polylines_px


##### polyline analysis #####

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
    i = next((i for i, pos in enumerate(pl) if pos[2] > z))
    return pl[i:]


def simplify(pl, n):

    if len(pl) < n:
        return pl
    else:
        return np.array([quantile_point(pl, q) for q in np.linspace(0, 1, n)])


