# TODO : supprimer n_stem_min ?

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import directed_hausdorff


def z_to_xy(polyline, z):
    """
    For all the points (x', y', z') in a polyline, this function returns the coordinates (x', y') whose corresponding
    z' height is the closest to z.

    Parameters
    ----------
    polyline : 2D array
        a 3D polyline
    z : int
        z coordinate in 3D space

    Returns
    -------
    float
        (x', y') coordinates
    """

    i = np.argmin(abs(np.array(polyline)[:, 2] - z))
    x, y = polyline[i][:2]
    return x, y


def get_median_polyline(polylines, n_stem_min=0, dz=2):
    """
    Returns a median polyline on the z axis

    Parameters
    ----------
    polylines : list of 2D arrays
        list of 3D polylines
    n_stem_min : int
        This parameters determines the maximum height of the median polyline : median polyline is only computed at
        height z if at least n_stem_min polyline have a max height  > z.
    dz : float
        space between two successive points of the median polyline on z axis.

    Returns
    -------
    2D array
        3D median polyline

    """

    z = np.median([pl[0][2] for pl in polylines])
    median_polyline = []

    while len(polylines) > n_stem_min:

        xy = np.array([z_to_xy(pl, z) for pl in polylines])
        xy_median = list(np.median(xy, axis=0))
        median_polyline.append(xy_median + [z])

        polylines = [pl for pl in polylines if pl[-1][2] > z]
        z += dz

    return np.array(median_polyline)


def abnormal_stem(vmsi_list, dist_threshold=100):
    """
    Test if some vmsi in vmsi_list have a stem whose shape is abnormally different compared to the other stems.

    Parameters
    ----------
    vmsi_list : list of openalea.phenomenal.object.voxelSegmentation.VoxelSegmentation objects
    dist_threshold : float

    Returns
    -------
    list of bool
        True (= abnormal) or False (= normal) for each vmsi in vmsi_list

    """

    stem_polylines = [np.array(vmsi.get_stem().get_highest_polyline().polyline) for vmsi in vmsi_list]

    median_stem = get_median_polyline(polylines=stem_polylines)

    abnormal = []
    for i, polyline in enumerate(stem_polylines):
        d = directed_hausdorff(polyline, median_stem)[0]
        abnormal.append(d > dist_threshold)

    return abnormal

# ============================================================================================================

def xyz_last_mature(vmsi):

    # xyz position of the last mature leaf insertion

    if vmsi.get_mature_leafs() == []:
        xyz = vmsi.get_stem().real_longest_polyline()[0]
    else:
        mature_ins = np.array([l.info['pm_position_base'] for l in vmsi.get_mature_leafs()])
        mature_ins = mature_ins[mature_ins[:, 2].argsort()]
        xyz = mature_ins[-1]

    return xyz


def savgol_smoothing_function(x, y, dw, polyorder, repet):

    w = int(len(x) / dw)
    w = w if w % 2 else w + 1  # odd

    # x2, y2 = savgol_filter((x, y), window_length=w, polyorder=polyorder)
    # x2, y2 = savgol_filter((x2, y2), window_length=w, polyorder=polyorder)
    x2, y2 = x, y
    for k in range(repet):
        x2, y2 = savgol_filter((x2, y2), window_length=w, polyorder=polyorder)

    # TODO : bricolage ! trouver une fonction de lissage monotone
    # monotony
    for i in range(1, len(y2)):
        y2[i] = max([max(y2[:i]), y2[i]])

    # interpolating function
    f = UnivariateSpline(x2, y2)

    return f


def ear_anomaly(y, dy):

    i_abnomaly = []
    for i in range(1, len(y)):
        y_max = np.max(y[:i])
        if y_max - y[i] > dy:
            i_abnomaly.append(i)

    return i_abnomaly


def smoothing_function(x, y, dy=None, dw=4, polyorder=2, repet=2):

    if dy is None:
        i_anomaly = []
    else:
        i_anomaly = ear_anomaly(y, dy=dy)

    i_max = max([i for i in range(len(x)) if i not in i_anomaly])
    x_max = x[i_max]

    y = [val for i, val in enumerate(y) if i not in i_anomaly]
    x = [val for i, val in enumerate(x) if i not in i_anomaly]

    f = savgol_smoothing_function(x, y, dw=dw, polyorder=polyorder, repet=repet)

    return f, x_max


def stem_height_smoothing(t, y, neighbours=3, threshold=0.05):
    """
    (Tested on deepcollars outputs)
    """

    # anomaly detection
    i_anomaly = []
    for i in range(len(y)):
        # check if y[i] is too high compared to the next neighbours
        y_ngb = np.array(y[(i + 1):(i + 1 + neighbours)])
        if len(y_ngb) == neighbours and all([y[i] > val for val in y_ngb]) and np.mean(y[i] - y_ngb) / y[i] > threshold:
            i_anomaly.append(i)
        # check if y[i] is too low compared to the previous neighbours
        y_ngb = np.array(y[(i - neighbours):i])
        if len(y_ngb) == neighbours and all([y[i] < val for val in y_ngb]) and np.mean(y_ngb - y[i]) / np.mean(y_ngb) > threshold:
            i_anomaly.append(i)

    # anomaly removing
    t2 = [val for i, val in enumerate(t) if i not in i_anomaly]
    y2 = [val for i, val in enumerate(y) if i not in i_anomaly]

    # smoothing
    f, _ = smoothing_function(t2, y2, dw=6, repet=3)
    f2 = lambda x: float(min(f(x), max(y2)))
    return f2


