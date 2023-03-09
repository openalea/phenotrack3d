import numpy as np
import cv2
import matplotlib.pyplot as plt

from openalea.maizetrack.local_cache import get_metainfos_ZA17, metainfos_to_paths
from openalea.maizetrack.utils import phm3d_to_px2d, simplify, get_rgb
from openalea.maizetrack.phenomenal_display import plot_sk

from openalea.deepcollar.predict_insertion_DL import detect_insertion

import openalea.phenomenal.object as phm_obj
import openalea.phenomenal.segmentation as phm_seg

from skimage import io

has_pgl_display = True
try:
    from openalea.plantgl import all as pgl
except ImportError:
    has_pgl_display = False


def plot_skeleton2(vmsi, stem_path):

    size = 15
    col = pgl.Material(pgl.Color3(255, 0, 0))
    col2 = pgl.Material(pgl.Color3(213, 100, 0))

    shapes = []
    h = 700  # - vmsi.get_stem().info['pm_z_base']
    for leaf in vmsi.get_leafs():
        segment = leaf.get_highest_polyline().polyline # for leaf

        segment = segment[:-2]

        for k in range(len(segment) - 1):
            # arguments cylindre : centre1, centre2, rayon, nbre segments d'un cercle.
            pos1 = np.array(segment[k]) + np.array([0, 0, h])
            pos2 = np.array(segment[k + 1]) + np.array([0, 0, h])
            cyl = pgl.Extrusion(pgl.Polyline([pos1, pos2]), pgl.Polyline2D.Circle(int(size * 0.8), 8))
            cyl.solid = True  # rajoute 2 cercles aux extremites du cylindre
            cyl = pgl.Shape(cyl, col)
            shapes.append(cyl)

    segment = stem_path.polyline
    for k in range(len(segment) - 1):
        # arguments cylindre : centre1, centre2, rayon, nbre segments d'un cercle.
        pos1 = np.array(segment[k]) + np.array([0, 0, h])
        pos2 = np.array(segment[k + 1]) + np.array([0, 0, h])
        cyl = pgl.Extrusion(pgl.Polyline([pos1, pos2]), pgl.Polyline2D.Circle(int(size * 1.2), 8))
        cyl.solid = True  # rajoute 2 cercles aux extremites du cylindre
        cyl = pgl.Shape(cyl, col2)
        shapes.append(cyl)

    scene = pgl.Scene(shapes)
    pgl.Viewer.display(scene)


plantid = 1424
metainfos = get_metainfos_ZA17(plantid)

daydate = '2017-05-12'
metainfo = next(m for m in metainfos if m.daydate == daydate)

skeleton_path = metainfos_to_paths([metainfo], phm_parameters=(4, 1, 'notop', 4, 100), object='skeleton')[0]
vmsi_path = metainfos_to_paths([metainfo], phm_parameters=(4, 1, 'notop', 4, 100), object='vmsi')[0]

weights = 'deepcollar/examples/data/model/weights.weights'
config = 'deepcollar/examples/data/model/config.cfg'
net = cv2.dnn.readNetFromDarknet(config, weights)
model = cv2.dnn_DetectionModel(net)

angles = [a for a, v in zip(metainfo.camera_angle, metainfo.view_type) if v == 'side']
images = {angle: get_rgb(metainfo, angle, main_folder='data/rgb_insertion_annotation/rgb',# used only to retrieve a copy of rgb image
                         plant_folder=False, save=False, side=True)[0] for angle in angles}

# =============================================================================================================

skeleton = phm_obj.VoxelSkeleton.read_from_json_gz(skeleton_path)

pl = phm_seg.get_highest_segment(skeleton.segments).polyline
pl3d = simplify(pl, 30)

stem_polylines = {angle: phm3d_to_px2d(pl3d, metainfo.shooting_frame, angle=angle) for angle in angles}

for angle in [60]:

    # predict x,y
    res = detect_insertion(image=images[angle], stem_polyline=stem_polylines[angle], model=model, display=True,
                           display_vignettes=True)

    img = images[angle]
    pl = stem_polylines[angle]
    img = cv2.polylines(np.float32(img), [pl.astype(int).reshape((-1, 1, 2))], False, (213, 120, 0), 12)
    plt.imshow(img.astype(np.uint8))

    io.imsave('stem_path.png', img)

stem_segment = phm_seg.get_highest_segment(skeleton.segments)
vmsi = phm_obj.VoxelSegmentation.read_from_json_gz(vmsi_path)
plot_skeleton2(vmsi, stem_segment)


# import os
# folder = 'paper/method/'
# for path in os.listdir(folder):
#     img = io.imread(folder + path)
#     img = img[:, :, :3]
#     img[(img == np.array([255, 255, 255])).all(2)] = [120, 120, 120]
#     img[(img == np.array([0, 0, 0])).all(2)] = [255, 255, 255]
#     io.imsave(folder + 'v2' + path, img)
