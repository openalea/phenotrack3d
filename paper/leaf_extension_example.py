from skimage import io
import os

from openalea.phenomenal import object as phm_obj
import matplotlib.pyplot as plt

from openalea.maizetrack.local_cache import metainfos_to_paths, get_metainfos_ZA17
from openalea.maizetrack.leaf_extension import leaf_extension
from openalea.maizetrack.phenomenal_display import plot_sk, plot_vmsi, plot_snapshot


if __name__ == '__main__':

    plantid = 1424

    metainfos = get_metainfos_ZA17(plantid)

    daydate = '2017-05-12'

    metainfo = next(m for m in metainfos if m.daydate == daydate)

    vmsi_path = metainfos_to_paths([metainfo], object='vmsi', stem_smoothing=True, old=True)[0]
    vmsi = phm_obj.VoxelSegmentation.read_from_json_gz(vmsi_path)

    shooting_frame = metainfo.shooting_frame

    # binary image
    binaries = dict()
    path = 'binary2/' + daydate
    files = [f for f in os.listdir(path) if f[:4] == str(plantid).rjust(4, '0')]
    files = [f for f in files if str(metainfo.task) in f]
    angles = sorted([a for a, t in zip(metainfo.camera_angle, metainfo.view_type) if t == 'side'])
    for angle in angles:
        file = next(f for f in files if str(angle) in f[-7:])
        binaries[angle] = io.imread(path + '/' + file)  # / 255. later

    vmsi = leaf_extension(vmsi, binaries, shooting_frame, display_parameters=[60, True, False, False])

    skeleton_path = metainfos_to_paths([metainfo], object='skeleton')[0]
    skeleton = phm_obj.VoxelSkeleton.read_from_json_gz(skeleton_path)

    plot_sk(skeleton)

