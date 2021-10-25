from openalea.maizetrack.utils import phm3d_to_px2d

from openalea.maizetrack.local_cache import metainfos_to_paths, get_metainfos_ZA17, check_existence, load_plant
from openalea.maizetrack.utils import get_rgb
from openalea.maizetrack.trackedPlant import TrackedPlant

import cv2
import os
import numpy as np
from skimage import io
from PIL import Image

#local copy of phenoarch_cache/ZA17
cache_dir = 'local_cache/cache_ZA17/'
#folder =  cache_dir + 'segmentation_voxel4_tol1_notop_vis4_minpix100_stem_smooth_tracking'
folder = cache_dir + 'segmentation_voxel4_tol1_notop_vis4_minpix100_no_stem_smooth_no_tracking'
#folder = cache_dir + '1429_nosmooth'

plantid = 1429
metainfos = get_metainfos_ZA17(plantid)
paths = metainfos_to_paths(metainfos, folder=folder)
metainfos, paths = check_existence(metainfos, paths)
vmsi_list = load_plant(metainfos, paths)

plant = TrackedPlant.load_and_check(vmsi_list)

# ===========================================================

print('downloading images..')

imgs = []
angle = 60
for snapshot in plant.snapshots:

    s = snapshot.metainfo
    print(s.daydate)

    try:
        img, _ = get_rgb(metainfo=s, angle=angle)
    except:
        img = imgs[0] * 0
        print('problem')
    imgs.append(img)


print('drawing polylines..')

STOP_STEM_AT_LAST_COLLAR = True

imgs_phm = []
for snapshot, img in zip(plant.snapshots, imgs):

    print(snapshot.metainfo.daydate)

    sf = snapshot.metainfo.shooting_frame_y

    angle = 60

    # plot stem

    pl_stem = np.array(snapshot.get_stem().get_highest_polyline().polyline)

    if STOP_STEM_AT_LAST_COLLAR:
        collar_heights = [l.info['pm_z_base'] for l in snapshot.get_mature_leafs()]
        if collar_heights:
            z_stem = max(collar_heights)
        else:
            z_stem = np.min(pl_stem[2])
        pl_stem = pl_stem[np.where(pl_stem[:, 2] < z_stem)]

    pl_stem = phm3d_to_px2d(pl_stem, sf)

    img = cv2.polylines(np.float32(img), [pl_stem.astype(int).reshape((-1, 1, 2))], False, (255, 255, 255), 5)
    img = cv2.polylines(np.float32(img), [pl_stem.astype(int).reshape((-1, 1, 2))], False, (0, 0, 0), 3)

    # plot leaves

    for leaf in snapshot.get_leafs():
        mature = leaf.info['pm_label'] == 'mature_leaf'

        if mature:
            pl = phm3d_to_px2d(leaf.real_longest_polyline(), sf, angle)
            col = (0, 0, 255)
        else:
            if not STOP_STEM_AT_LAST_COLLAR:
                pl = phm3d_to_px2d(leaf.real_longest_polyline(), sf, angle)
            else:
                pl = np.array(leaf.get_highest_polyline().polyline)
                pl = pl[np.where(pl[:, 2] >= z_stem)[0][0]:]

            pl = phm3d_to_px2d(pl, sf, angle)
            col = (230, 159, 0)

        img = cv2.polylines(np.float32(img), [pl.astype(int).reshape((-1, 1, 2))], False, (0, 0, 0), 7)
        img = cv2.polylines(np.float32(img), [pl.astype(int).reshape((-1, 1, 2))], False, col, 5)


    # plot stem tip

    xy = pl_stem[-1]
    img = cv2.circle(img, tuple(xy.astype(int)), 13, (255, 255, 255), -1)
    img = cv2.circle(img, tuple(xy.astype(int)), 10, (0, 0, 0), -1)

    # # plot insertion of last mature
    #
    # xyz = xyz_last_mature(snapshot)
    # xy = phm3d_to_px2d(xyz, sf)[0]
    # img = cv2.circle(img, tuple(xy.astype(int)), 13, (0, 0, 0), -1)
    # img = cv2.circle(img, tuple(xy.astype(int)), 10, (0, 0, 255), -1)

    # plot last insertion
    #insertions = np.array([l.info['pm_position_base'] for l in snapshot.get_leafs()])
    #xyz = insertions[np.argmax(insertions[:, 2])]
    #xy = phm3d_to_px2d(xyz, sf)[0]
    #img = cv2.circle(img, tuple(xy.astype(int)), 13, (0, 0, 0), -1)
    #img = cv2.circle(img, tuple(xy.astype(int)), 10, (230, 159, 0), -1)


    #plt.imshow(img.astype(np.uint8))
    #plt.imshow(img) # only show polylines !
    imgs_phm.append(img)


# save_images = True
# if save_images:
#     for k, img in enumerate(imgs_phm):
#         io.imsave('img{}.png'.format(k), img)


imgs_gif = imgs_phm.copy()
#imgs_gif = [img[200:-200, 250:-250, :] for img in imgs_gif]
imgs_gif = [Image.fromarray(np.uint8(img)) for img in imgs_gif]
fps = 1
imgs_gif[0].save('gif/animation_id{}_{}fps.gif'.format(plantid, fps),
              save_all=True,
              append_images=imgs_gif[1:],
              optimize=True,
              duration=1000/fps,
              loop=0)