from skimage import io
import os

from openalea.phenomenal import object as phm_obj

from openalea.maizetrack.data_loading import metainfos_to_paths, get_metainfos_ZA17
from openalea.maizetrack.leaf_extension import leaf_extension

if __name__ == '__main__':

    plantid = 1424
    daydate = '2017-05-15'

    metainfos = get_metainfos_ZA17(plantid)

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

    vmsi = leaf_extension(vmsi, binaries, shooting_frame, display_angle=60)

    for k in range(1, 1 + vmsi.get_number_of_leaf()):

        leaf = vmsi.get_leaf_order(k)
        if leaf.info['pm_label'] == 'mature_leaf':
            l1 = leaf.info['pm_length']
            l2 = leaf.info['pm_length_extended']
        else:
            l1 = leaf.info['pm_length_with_speudo_stem']
            l2 = leaf.info['pm_length_extended']

        print('leaf {} : length = {} (extension = {})'.format(k, round(l2, 2), round(l2 / l1, 2)))


