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


plant = '0112/ZM4535/EPPN11_L/WW/EPPN_Rep_4/02_52/ARCH2020-02-03'

df = []


meta_snapshots = index.get_snapshots(index.filter(plant=plant, nview=13), meta=True)
meta_snapshots = [m for m in meta_snapshots if '2020-02-14' <= m.daydate <= '2020-04-01']


all_images = []
for m in meta_snapshots:
    images = {}
    for angle in [k * 30 for k in range(12)]:
        rgb_path = next(p for (v, a, p) in zip(m.view_type, m.camera_angle, m.path) if v == 'side' and a == angle)
        images[angle] = cv2.cvtColor(cache.image_client.imread(rgb_path), cv2.COLOR_BGR2RGB)
    all_images.append(images)


def binary_pot_360(images, bin_threshold=140, xlim=(1004, 1049), ylim=(2037, 2210), px_per_degree=43/30):

    pot360 = np.zeros((np.diff(ylim)[0], 360))

    for angle in images.keys():

        # crop a section of the pot (centered around x-center position of the pot to minimize angle deformations)
        pot = images[angle][ylim[0]:ylim[1], xlim[0]:xlim[1]]

        # binarize pot image
        bin = cv2.cvtColor(pot, cv2.COLOR_RGB2GRAY) < bin_threshold

        # project bin on a flat cylinder space (i.e. image of width=360)
        px_width = bin.shape[1]
        deg_width = 2 * int((px_width / px_per_degree) / 2) + 1  # round to closest odd integer
        a1, a2 = angle - int(deg_width / 2), angle + int(deg_width / 2) + 1
        angle_coordinates = (np.arange(a1, a2) % 360).astype(int)
        bin_deg = cv2.resize(bin.astype(np.uint8), (deg_width, bin.shape[0]))
        pot360[:, angle_coordinates] += bin_deg

    pot360 = cv2.GaussianBlur((pot360 >= 1).astype(np.uint8), (3, 3), 0)
    return pot360


def pot_rotation(images_time_series):

    # extract a 360 degree flat binarization of each pot in the time-series
    pots360 = [binary_pot_360(images) for images in images_time_series]

    pot_ref = pots360[0]
    rotations = [0]
    for pot in pots360[1:]:

        # find the rotation angle 0 <= da <= 360 which leads to the best alignment with the reference pot
        matching_scores = []
        for da in range(360):
            pot_rotated = pot[:, (np.arange(len(pot[0])) - da) % 360]
            score = np.sum(pot_ref == pot_rotated)
            matching_scores.append(score)
        da = np.argmax(matching_scores)

        # rescale the result to [-180, 180]
        rotations.append(da if da <= 180 else da - 360)
        # plt.imshow(pot[:, (np.arange(len(pot[0])) - da) % 360])

    return rotations

plt.figure()
plt.imshow(np.sum(np.array(mean_spectrums), axis=0), cmap='Greys')
plt.figure()
plt.imshow(np.sum(np.array(output), axis=0), cmap='Greys')




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








