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
        ranks = snapshot.rank_annotation

    # plot leaves
    for i, (pl, c, mature) in enumerate(zip(polylines_px, ranks, matures)):

        if c == -2:
            col = [0, 0, 0]
        else:
            col = [int(x) for x in PALETTE[c]]

        # unknown rank vs known rank
        if c == -1:
            leaf_border_col = (0, 0, 0)
        else:
            leaf_border_col = (255, 255, 255)

        #img = cv2.polylines(np.float32(img), [pl.astype(int).reshape((-1, 1, 2))], False, leaf_border_col, 10 * ds)
        img = cv2.polylines(np.float32(img), [pl.astype(int).reshape((-1, 1, 2))], False, col, 7 * ds)

    return img, polylines_px


# available in modulor
folder = 'local_cache/cache_ZA17/segmentation_voxel4_tol1_notop_vis4_minpix100_stem_smooth_tracking'
all_files = [folder + '/' + rep + '/' + f for rep in os.listdir(folder) for f in os.listdir(folder + '/' + rep)]

plantids = [313, 316, 329, 330, 336, 348, 424, 439, 461, 474, 794, 832, 905, 907, 915, 925, 931, 940, 959, 1270,
            1276, 1283, 1284, 1301, 1316, 1383, 1391, 1421, 1424, 1434]


plantid = 313

metainfos = get_metainfos_ZA17(plantid)
paths = metainfos_to_paths(metainfos, folder=folder)
metainfos, paths = check_existence(metainfos, paths)
vmsi_list = load_plant(metainfos, paths)

plant_ref = TrackedPlant.load_and_check(vmsi_list)
plant_ref.load_rank_annotation()

plant_aligned = copy.deepcopy(plant_ref)
plant_aligned.align_mature(direction=1, gap=12.365, w_h=0.03, w_l=0.002, gap_extremity_factor=0.2, n_previous=500,
                           rank_attribution=False)
plant_aligned.align_growing()

plant = plant_aligned
only_mature = True

snapshots = plant.snapshots[0::3]


from skimage import io
plant.load_images(60)
plant.simplify_polylines()
s = copy.deepcopy(snapshots[-2])
s.leaves = [l for l in s.leaves if l.info['pm_label'] == 'mature_leaf']
img, _ = rgb_and_polylines2(s, angle=60, ranks=[5, 6, 7, 8, 9, 10, 18, 11, 12, 13], ds=3)
io.imsave('test.png', img)

s = copy.deepcopy(snapshots[-8])
s.leaves = [l for l in s.leaves if l.info['pm_label'] == 'growing_leaf']
img, _ = rgb_and_polylines2(s, angle=60, ranks=[10, 12, 11, 13], ds=3)
io.imsave('test2.png', img)

#plot_snapshot(s, ranks=[5, 6, 7, 8, 9, 10, 18, 11, 12, 13], stem=False)


T = len(snapshots)
R = len(snapshots[0].order)
mat = np.zeros((T, R)) * np.NAN
for t, snapshot in enumerate(snapshots):
    for r, index in enumerate(snapshot.order):
        if index != -1:
            if snapshot.leaves[index].info['pm_label'] == 'mature_leaf' or only_mature == False:
                mat[t, r] = snapshot.leaves[index].rank_annotation

# remove empty columns
mat = mat[:, ~np.isnan(mat).all(axis=0)]

fig, ax = plt.subplots()
fig.canvas.set_window_title(str(plantid))

#ax.set_xticks(np.arange(R) - 0.5, minor=True)
ax.set_yticks(np.arange(T + 1) - 0.5, minor=True)
ax.grid(which='minor', color='white', linewidth=12)

# start axis at 1
plt.xticks(np.arange(R), np.arange(R) + 1)
plt.yticks(np.arange(T), np.arange(T) + 1)

ax.set_xlabel('Leaf rank', fontsize=30)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.set_ylabel('Time', fontsize=30)
plt.locator_params(nbins=4)
ax.tick_params(axis='both', which='major', labelsize=25)  # axis number size


rgb = np.array(PALETTE) / 255.
rgb = np.concatenate((np.array([[0., 0., 0.]]), rgb))
cmap = colors.ListedColormap(rgb, "")
val = [k - 1.5 for k in range(50)]
norm = colors.BoundaryNorm(val, len(val)-1)
plt.imshow(mat, interpolation='nearest', cmap=cmap, norm=norm)







