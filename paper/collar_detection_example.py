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


plantid = 1424
metainfos = get_metainfos_ZA17(plantid)

daydate = '2017-05-15'
metainfo = next(m for m in metainfos if m.daydate == daydate)

skeleton_path = metainfos_to_paths([metainfo], phm_parameters=(4, 1, 'notop', 4, 100), object='skeleton')[0]

weights = 'deepcollar/examples/data/model/weights.weights'
config = 'deepcollar/examples/data/model/config.cfg'
net = cv2.dnn.readNetFromDarknet(config, weights)
model = cv2.dnn_DetectionModel(net)

angles = [a for a, v in zip(metainfo.camera_angle, metainfo.view_type) if v == 'side']
images = {angle: get_rgb(metainfo, angle, main_folder='data/rgb_insertion_annotation/rgb',
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
    img = cv2.polylines(np.float32(img), [pl.astype(int).reshape((-1, 1, 2))], False, (255, 255, 0), 6)
    plt.imshow(img.astype(np.uint8))

    io.imsave('stem_path.png', img)

stem_segment = phm_seg.get_highest_segment(skeleton.segments)
plot_sk(skeleton, stem_segment=stem_segment)

