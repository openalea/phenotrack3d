'''
Examples of usages of stem_correction.py
Most of these functions are deprecated
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openalea.maizetrack.data_loading import get_metainfos_ZA17, metainfos_to_paths, check_existence, load_plant, missing_data
from openalea.maizetrack.stem_correction import abnormal_stem, ear_anomaly, smoothing_function, savgol_smoothing_function
from openalea.maizetrack.utils import phm3d_to_px2d

TEST_SET = [348, 1301, 832, 1276, 1383, 1424, 940, 1270, 931, 925, 474, 794, 1283, 330, 1421,
            907, 316, 1284, 336, 439, 959, 915, 1316, 1434, 905, 313, 1391, 461, 424, 329, 784, 1398, 823, 1402,
            430, 1416, 1309, 1263, 811, 1413, 466, 478, 822, 918, 937, 1436, 1291, 470, 327]
plantid = 348

def comp1(plantid):

    dic = {}
    print(plantid)

    # LOAD METAINFOS

    metainfos = get_metainfos_ZA17(plantid)

    # LOAD VMSI

    # vmsi_list = load_plant(metainfos=metainfos, stem_smoothing=False)
    metainfos = get_metainfos_ZA17(plantid)
    paths = metainfos_to_paths(metainfos, stem_smoothing=False, phm_parameters=(4, 1, 'notop', 4, 100))
    metainfos, paths = check_existence(metainfos, paths)
    vmsi_list = load_plant(metainfos, paths)
    print('vmsi found for {}/{} metainfos'.format(len(vmsi_list), len(metainfos)))

    # sort vmsi by time
    timestamps = [vmsi.metainfo.timestamp for vmsi in vmsi_list]
    order = sorted(range(len(timestamps)), key=lambda k: timestamps[k])
    vmsi_list = [vmsi_list[i] for i in order]

    # REMOVE ABNOMALY

    checks_data = [not missing_data(v) for v in vmsi_list]
    print('{} vmsi to remove because of missing data'.format(len(checks_data) - sum(checks_data)))
    checks_stem = [not b for b in abnormal_stem(vmsi_list)]
    print('{} vmsi to remove because of stem shape abnormality'.format(len(checks_stem) - sum(checks_stem)))
    checks = list((np.array(checks_data) * np.array(checks_stem)).astype(int))
    vmsi_list = [vmsi for check, vmsi in zip(checks, vmsi_list) if check]

    # STEM HEIGHT FIT

    # timestamp_list = [v.metainfo.timestamp for v in vmsi_list]
    # sf_list = [v.metainfo.shooting_frame_y for v in vmsi_list]

    # tempo
    # m = min(timestamp_list)
    # timestamp_list = [(t - m) / 3600 for t in timestamp_list]

    # plt.figure()
    # xyz_stem = np.array([v.get_stem().info['pm_position_tip'] for v in vmsi_list])
    # plt.plot(timestamp_list, (700 + np.array(xyz_stem)[:, 2])/10, 'k*', label='Phenomenal')

    # f, i_max = smoothing_function(timestamp_list, xyz_stem[:, 2])
    # plt.plot(timestamp_list, (700 + f(timestamp_list))/10, '-k', label='Lissage')
    # plt.xlabel('Temps (h)')
    # plt.ylabel('Hauteur de la tige (cm)')
    # plt.title(id)
    # plt.legend()

    # date_max = vmsi_list[i_max].metainfo.daydate
    # print('day max = ', date_max)

    # compute_vmsi(id, date_max=date_max, metainfos=metainfos, f_stem=f, replace=False)


    plt.figure(plantid)
    timestamp_list = [v.metainfo.timestamp for v in vmsi_list]
    z_stem = np.array([v.get_stem().info['pm_position_tip'][2] for v in vmsi_list])
    plt.plot(timestamp_list, z_stem, 'k*')
    highest_insertion = [max([l.info['pm_position_base'][2] for l in vmsi.get_leafs()]) for vmsi in vmsi_list]
    plt.plot(timestamp_list, highest_insertion, 'y*')

    dz = 0.2 * (np.max(z_stem) - np.min(z_stem))
    i_anomaly = ear_anomaly(z_stem, dz)

    z_stem2 = [highest_insertion[i] if i in i_anomaly else z_stem[i] for i in range(len(z_stem))]
    plt.plot(timestamp_list, z_stem2, 'k-')

    f, t_max = smoothing_function(timestamp_list, z_stem2, dz)
    plt.plot(timestamp_list, f(timestamp_list), 'r-')

    dic[plantid] = {'t': timestamp_list, 'z': z_stem, 'hi': highest_insertion}



    for plantid, value in dic.items():

        plt.figure(plantid)
        t, z, hi = value['t'], value['z'], value['hi']
        plt.plot(t, z, 'k*')
        plt.plot(t, hi, 'y*')

        dz = 0.20 * (np.max(z) - np.min(z))
        i_anomaly = ear_anomaly(z, dz)
        z2 = [hi[i] if i in i_anomaly else z[i] for i in range(len(z))]
        i_anomaly = ear_anomaly(z2, dz)
        z2 = [hi[i] if i in i_anomaly else z2[i] for i in range(len(z2))]
        plt.plot(t, z2, 'k-')

        f, t_max = smoothing_function(t, z2, None)
        # plt.plot(t, [min(max(hi), z) for z in f(t)], 'r-')
        plt.plot(t, f(t), 'r-')
        plt.title(plantid)


def comp2(xyz_stem, timestamp_list, xyz_last_ins, vmsi_list):

    highest_insertion = [max([l.info['pm_position_base'][2] for l in vmsi.get_leafs()]) for vmsi in vmsi_list]
    plt.plot(timestamp_list, highest_insertion, 'y*')

    xyz_stem2 = np.array(xyz_stem).copy()
    max_decrease = (np.max(xyz_stem) - np.min(xyz_stem) ) /5
    i_decrease = []
    for i in range(1, len(xyz_stem2)):
        if np.max(xyz_stem2[:i][:, 2]) - xyz_stem2[i][2] > max_decrease:
            i_decrease.append(i)
    xyz_stem2[i_decrease] = np.array(xyz_last_ins)[i_decrease]
    plt.plot(timestamp_list, np.array(xyz_stem2)[:, 2], 'k-')

    x = np.array(timestamp_list)
    y = np.array(xyz_stem2)[:, 2]

    # plt.clf()
    # plt.plot(x, y, '*')

    f = savgol_smoothing_function(x, y)

    plt.clf()
    plt.plot(x, f(x))


    n_max = np.argmax(np.array(xyz_stem)[:, 2])
    x = timestamp_list[:n_max]
    y = np.array(xyz_stem)[:, 2][:n_max]
    # plt.figure()
    # plt.plot(x, y, 'k*')
    f = np.poly1d(np.polyfit(x, y, 2))
    # plt.plot(x, f(x), 'k-')
    # plt.title(str(plantid))

    # t = np.array(timestamp_list)
    # h = np.array(xyz_stem)[:, 2]
    # f = np.poly1d(np.polyfit(t, h, 2))

    # COMPUTE VMSI WITH CORRECTION


def clean_plot(t, h, xyz_mature_ins):

    t2 = (t - np.min(t)) / 3600
    h2 = (h - np.min(h)) / 10

    plt.clf()

    plt.xlabel('Time (h)')
    plt.ylabel('Stem height (cm)')

    plt.plot(t2, h2, '-k*', label='phenomenal : stem top')

    f2 = np.poly1d(np.polyfit(t2, h2, 2))
    plt.plot(t2, f2(t2), 'k-', label='Fit (polynomial deg 2)')

    h_mature = (np.array(xyz_mature_ins)[:, 2] - np.min(h) ) /10
    plt.plot(t2, h_mature, 'b*-', label='phenomenal : last insertion (mature)')

    plt.legend()


def show_anot(timestamp_list, xyz_stem, sf_list, xyz_mature_ins, vmsi_list):

    plt.clf()
    px_stem = np.array([phm3d_to_px2d(xyz, sf)[0] for xyz, sf in zip(xyz_stem, sf_list)])
    plt.plot(timestamp_list, 2448 - px_stem[:, 1], 'k*-', label='Phenomenal : stem top')

    px_mature_ins = np.array([phm3d_to_px2d(xyz, sf)[0] for xyz, sf in zip(xyz_mature_ins, sf_list)])
    plt.plot(timestamp_list, 2448 - px_mature_ins[:, 1], 'b*-', label='Phenomenal : last insertion (mature)')

    xyz_last_ins = []
    for vmsi in vmsi_list:
        pos = np.array([l.info['pm_position_base'] for l in vmsi.get_leafs()])
        xyz_last_ins.append(list(pos[pos[:, 2].argsort()][-1]))
    px_last_ins = np.array([phm3d_to_px2d(xyz, sf)[0] for xyz, sf in zip(xyz_last_ins, sf_list)])
    plt.plot(timestamp_list, 2448 - px_last_ins[:, 1], 'y*-')

    anot = pd.read_csv('data/stem_annotation/{}_stem_anot.csv'.format(id), header=None)
    anot.columns = ['label', 'x', 'y', 'name', 'img_w', 'img_h']
    anot['date'] = [n[18:23] for n in anot['name']]
    anot['timestamp'] = [int(n[1:11]) for n in anot['name']]
    plt.plot(anot['timestamp'], 2448 - np.array(anot['y']), '-', label='Annotation : last insertion', color='g')
    plt.legend()

    plt.xlabel('Time')
    plt.ylabel('Stem height')

    plt.xticks([])
    plt.yticks([])


