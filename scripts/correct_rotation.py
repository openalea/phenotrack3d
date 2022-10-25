# TODO : ZA20  '2020-02-14' <= daydate <= '2020-04-01'
# TODO : must be sorted by time (first snapshot = t0)
# TODO plantid 52 task 1912 : 330 angle missing

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from openalea.phenomenal import object as phm_obj
import openalea.maizetrack.phenomenal_display as phm_display

from alinea.phenoarch.cache import Cache
from alinea.phenoarch.platform_resources import get_ressources
from alinea.phenoarch.meta_data import plant_data
from PIL import Image

import cv2
import pandas as pd

exp = 'ZA20'

cache_client, image_client, binary_image_client, calibration_client = get_ressources(exp, cache='X:',
                                                                                     studies='Z:',
                                                                                     nasshare2='Y:')
parameters = {'reconstruction': {'voxel_size': 4, 'frame': 'pot'},
              'collar_detection': {'model_name': '3exp_xyside_99000'},
              'segmentation': {'force_stem': True}}
cache = Cache(cache_client, image_client, binary_image_client=binary_image_client,
              calibration_client=calibration_client, parameters=parameters)

index = cache.snapshot_index()

plants = list(index.plant_index[index.plant_index['rep'].str.contains('EPPN')]['plant'])

# for plant in np.random.choice(plants, 50, replace=False):
#     meta_snapshots = index.get_snapshots(index.filter(plant=plant), meta=True)
#     m = meta_snapshots[-2]
#     for angle in [k * 30 for k in range(12)]:
#         rgb_path = next(p for (v, a, p) in zip(m.view_type, m.camera_angle, m.path) if v == 'side' and a == angle)
#         rgb = cv2.cvtColor(cache.image_client.imread(rgb_path), cv2.COLOR_BGR2RGB)
#
#         rgb = cv2.rectangle(rgb, (977, 2037), (1083, 2210), (255, 0, 0), 2)
#         rgb = rgb[1900:, 850:1200]
#         plt.imsave('data/ZA20/rotation_correction/{}_{}.png'.format(m.pot, int(angle / 30)), rgb)


px_per_degree = 43/30
xlim = (977, 1083)
ylim = (2037, 2210)
bin_threshold = 140

# plant = '0190/ZM4545/EPPN6_H/WD2/EPPN_Rep_1/04_10/ARCH2020-02-03'

df = []


for plant in plants[52:]:

    meta_snapshots = index.get_snapshots(index.filter(plant=plant, nview=13), meta=True)
    meta_snapshots = [m for m in meta_snapshots if '2020-02-14' <= m.daydate <= '2020-04-01']

    mean_spectrums = []
    for m in meta_snapshots:

        # # TODO remove
        # plt.figure()
        # m = meta_snapshots[9]

        snapshot_mean_spectrum = []
        for k, angle in enumerate([k * 30 for k in range(12)]):

            rgb_path = next(p for (v, a, p) in zip(m.view_type, m.camera_angle, m.path) if v == 'side' and a == angle)
            rgb = cv2.cvtColor(cache.image_client.imread(rgb_path), cv2.COLOR_BGR2RGB)
            pot = rgb[ylim[0]:ylim[1], xlim[0]:xlim[1]]
            pot = cv2.cvtColor(pot, cv2.COLOR_RGB2GRAY)
            # plt.figure()
            # plt.imshow(np.dstack((pot, pot, pot)))
            # dx = 40
            # plt.plot([dx, dx], [0, 172], 'r-')
            # plt.plot([(1120 - 935) - dx, (1120 - 935) - dx], [0, 172], 'r-')

            # plt.figure(angle)
            bin = pot < bin_threshold

            segment_px = list(np.sum(bin, axis=0))

            # px scale --> degree scale
            f = interp1d(np.arange(len(segment_px)), segment_px)
            segment_deg = f(np.linspace(0, len(segment_px) - 1, round(len(segment_px) / px_per_degree)))

            # segment = np.array(segment).reshape(-1, 2).mean(axis=1)
            # axis = [0] * (330 + len(segment))
            # axis[angle:(angle + len(segment))] = segment
            spectrum = np.array([np.nan] * 360)
            a1, a2 = angle - (len(segment_deg) / 2), angle + (len(segment_deg) / 2)
            angle_coordinates = (np.arange(a1, a2) % 360).astype(int)
            spectrum[angle_coordinates] = segment_deg
            snapshot_mean_spectrum.append(spectrum)
            # plt.plot(axis, 'k-')

            # plt.subplot(12, 1, k + 1)
            # plt.xlim((0, 360))
            # plt.plot(spectrum, 'r-')
            # plt.plot(angle_coordinates, segment_deg, 'ro')

            # plt.plot([a1 % 360, a2 % 360], [0, 0], 'r*')
            # plt.plot([0] * angle + list(np.sum(bin, axis=0)) + [0] * (360 - angle), 'k-')
            # plt.imshow(bin)
        mean_spectrums.append(np.nanmean(np.array(snapshot_mean_spectrum), axis=0))

    for i in range(len(mean_spectrums)):
        s1, s2 = mean_spectrums[0], mean_spectrums[i]
        da_min, dist_min = 0, float('inf')
        for da in range(360):
            s2_bis = s2[(np.arange(len(s2)) - da) % 360]
            dist = np.mean(np.abs(s1 - s2_bis))
            if dist < dist_min:
                da_min, dist_min = da, dist

        da = da_min if da_min <= 180 else da_min - 360

        m = meta_snapshots[i]
        df.append([m.pot, m.task, da, m.timestamp])
        print(m.pot, m.task, da)

        # M[i, j] = da_min if da_min <= 180 else da_min - 360

        # plt.figure()
        # plt.plot(s1, 'k-')
        # plt.plot(s2, 'r-')
        # plt.plot(s2[(np.arange(len(s2)) - da_min) % 360], 'g-')
        #
        # daydate = next(m for m in meta_snapshots if m.timestamp == list(mean_spectrums.keys())[i + 1]).daydate
        # print(i, daydate, da_min if da_min <= 180 else da_min - 360)


df = pd.DataFrame(df, columns=['plantid', 'task', 'rotation', 'timestamp'])

df.to_csv('data/ZA20/pot_rotation.csv', index=False)

for plantid in df['plantid'].unique():
    # plt.figure(plantid)
    s = df[df['plantid'] == plantid].sort_values('timestamp')
    rot = np.array(s['rotation'])
    for i in range(1, len(rot)):
        a1, a2 = rot[[i - 1, i]]
        q_test = [-3, -2, -1, 0, 1, 2, 3]
        q = q_test[np.argmin([np.abs(a1 - (a2 + 360 * q)) for q in q_test])]
        # a2_corrected = a2 if np.abs(a2 - a1) < np.abs((a2 - np.sign(a2) * 360) - a1) % 360 else a2 - np.sign(a2) * 360
        a2_corrected = a2 + 360 * q
        rot[i] = a2_corrected
    # plt.plot(s['timestamp'], s['rotation'], 'ko-')
    plt.plot(s['timestamp'], rot, 'k-', linewidth=0.7)








