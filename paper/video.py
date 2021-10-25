import os
import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib import colors
import cv2

from openalea.maizetrack.local_cache import get_metainfos_ZA17, metainfos_to_paths, check_existence, load_plant
from openalea.maizetrack.trackedPlant import TrackedPlant
from openalea.maizetrack.phenomenal_display import PALETTE, plot_snapshot
from openalea.maizetrack.utils import phm3d_to_px2d


def rgb_and_polylines2(snapshot, angle, ranks=None, ds=1):

    # rgb image
    img = snapshot.image[angle]

    # adding polylines
    polylines_px = [phm3d_to_px2d(leaf.real_pl, snapshot.metainfo.shooting_frame, angle) for leaf in snapshot.leaves]
    matures = [leaf.info['pm_label'] == 'mature_leaf' for leaf in snapshot.leaves]

    if ranks is None:
        #ranks = snapshot.rank_annotation
        ranks = snapshot.get_ranks()

    # plot leaves
    for i, (pl, c, mature) in enumerate(zip(polylines_px, ranks, matures)):

        if c == -1:
            col = [255, 255, 255]
        else:
            col = [int(x) for x in PALETTE[c]]

        # unknown rank vs known rank
        if c == -1:
            leaf_border_col = (0, 0, 0)
        else:
            leaf_border_col = (255, 255, 255)

        #img = cv2.polylines(np.float32(img), [pl.astype(int).reshape((-1, 1, 2))], False, leaf_border_col, 10 * ds)
        img = cv2.polylines(np.float32(img), [pl.astype(int).reshape((-1, 1, 2))], False, col, 7 * ds)

    # write date
    id = str(int(snapshot.metainfo.plant[:4]))
    text = 'plantid {} / task {} ({})'.format(id, snapshot.metainfo.task, snapshot.metainfo.daydate)
    cv2.rectangle(img, (0, 2250), (2048, 2448), (0, 0, 0), -1)
    cv2.putText(img, text, (50, 2380), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 9, cv2.LINE_AA)

    return img, polylines_px


# ===================================================================================================

# available in modulor
folder = 'local_cache/cache_ZA17/segmentation_voxel4_tol1_notop_vis4_minpix100_stem_smooth_tracking'
all_files = [folder + '/' + rep + '/' + f for rep in os.listdir(folder) for f in os.listdir(folder + '/' + rep)]

plantids = [313, 316, 329, 330, 336, 348, 424, 439, 461, 474, 794, 832, 905, 907, 915, 925, 931, 940, 959, 1270,
            1276, 1283, 1284, 1301, 1316, 1383, 1391, 1421, 1424, 1434]

for plantid in plantids:

    metainfos = get_metainfos_ZA17(plantid)

    metainfos = [m for m in metainfos if m.daydate < '2017-06-08']

    paths = metainfos_to_paths(metainfos, folder=folder)
    metainfos, paths = check_existence(metainfos, paths)
    vmsi_list = load_plant(metainfos, paths)

    plant = TrackedPlant.load_and_check(vmsi_list)

    plant.align_mature(direction=1, gap=12.365, w_h=0.03, w_l=0.004, gap_extremity_factor=0.2, n_previous=500)
    plant.align_growing()

    plant.load_images(60)


    imgs = []
    for s in plant.snapshots:
        img, _ = rgb_and_polylines2(s, 60)
        #plt.imshow(img.astype(np.uint8))
        imgs.append(img.astype(np.uint8))

    # ========================================================

    width = 2048
    height = 2448
    channel = 3
    fps = 3

    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    video = cv2.VideoWriter('data/videos/mp4/{}.mp4'.format(plantid), fourcc, float(fps), (width, height))
    for img in imgs:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img_bgr)

    video.release()

    # =============================================================

    from PIL import Image

    imgs_gif = imgs.copy()
    imgs_gif = [Image.fromarray(np.uint8(img)) for img in imgs_gif]
    fps = 3
    imgs_gif[0].save('data/videos/gif/{}.gif'.format(plantid),
                     save_all=True,
                     append_images=imgs_gif[1:],
                     optimize=True,
                     duration=1000 / fps,
                     loop=0)