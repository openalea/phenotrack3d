import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline



def z_to_xy(polyline, z):

    # TODO : it's an approximation !
    i = np.argmin(abs(np.array(polyline)[:, 2] - z))
    x, y = polyline[i][:2]
    return x, y


def get_median_stem(stem_list, n_stem_min, dz):
    H = np.median([stem.info['pm_z_base'] for stem in stem_list])
    z = 0
    median_stem = []

    while len(stem_list) > n_stem_min:
        polylines = [np.array(stem.get_highest_polyline().polyline) - np.array([0, 0, H]) for stem in stem_list]
        xy = np.array([z_to_xy(polyline, z) for polyline in polylines])
        xy_median = list(np.median(xy, axis=0))
        median_stem.append(xy_median + [z])

        stem_list = [stem for stem in stem_list if stem.info['pm_z_tip'] - H > z]
        z += dz

    return median_stem


def index_with_abnormal_stem_shape(stem_list, median_stem, dist_threshold, plot_=True):

    H = np.median([stem.info['pm_z_base'] for stem in stem_list])
    polylines = [np.array(stem.get_highest_polyline().polyline) - np.array([0, 0, H]) for stem in stem_list]

    if plot_:
        plt.figure()
        plt.xlabel('Distance to median stem (mm)')
        plt.ylabel('Height (mm)')
        z_tip_min = np.min([stem.info['pm_z_base'] - H for stem in stem_list])
        z_tip_max = np.max([stem.info['pm_z_tip'] - H for stem in stem_list])
        plt.plot([dist_threshold, dist_threshold], [z_tip_min, z_tip_max],
                 color='r')

    index_to_remove = []
    for i, polyline in enumerate(polylines):
        z_list = []
        d_list = []
        col = 'b'
        for x, y, z in polyline:
            x_median, y_median = z_to_xy(median_stem, z)
            d = np.sqrt((x - x_median)**2 + (y - y_median)**2)
            z_list.append(z)
            d_list.append(d)

            if d > dist_threshold and i not in index_to_remove:
                index_to_remove.append(i)
                col = 'r'

        if plot_:
            plt.plot(d_list, z_list, color=col)

    return index_to_remove


def abnormal_stem(vmsi_list):
    stem_list = [v.get_stem() for v in vmsi_list]
    median_stem = get_median_stem(stem_list=stem_list,
                                  n_stem_min=5,
                                  dz=2)

    i_abnormal = index_with_abnormal_stem_shape(stem_list=stem_list,
                                                median_stem=median_stem,
                                                dist_threshold=100,
                                                plot_=False)

    abnormal = [i in i_abnormal for i in range(len(vmsi_list))]

    return abnormal


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


