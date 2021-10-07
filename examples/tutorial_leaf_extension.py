"""
An example script computing leaf extension on the 3D segmentation of a maize plant, resulting from the Phenomenal
pipeline.
"""

from skimage import io
from openalea.phenomenal import object as phm_obj
from openalea.maizetrack.leaf_extension import leaf_extension

# ===== load data ========================================================================

data_folder = 'maizetrack/examples/data/'

# 3D maize segmented object (vmsi) from Phenomenal
vmsi_path = data_folder + 'plant1/vmsi.json.gz'
vmsi = phm_obj.VoxelSegmentation.read_from_json_gz(vmsi_path)

# 12 side-view binary images from Phenomenal
image_folder = data_folder + 'plant1/binary/'
images = {angle: io.imread(image_folder + '{}.png'.format(angle)) for angle in [k * 30 for k in range(12)]}

# corresponding shooting_frame name
shooting_frame = 'elcom_2_c2_wide'

# ===== leaf extension ===================================================================

# add an item 'pm_length_extended' in info dict of each leaf of the vmsi
vmsi = leaf_extension(vmsi, images, shooting_frame, display_angle=60)

# ===== print results ====================================================================

for k in range(1, 1 + vmsi.get_number_of_leaf()):
    leaf = vmsi.get_leaf_order(k)
    if leaf.info['pm_label'] == 'mature_leaf':
        l1 = leaf.info['pm_length']
        l2 = leaf.info['pm_length_extended']
    else:
        l1 = leaf.info['pm_length_with_speudo_stem']
        l2 = leaf.info['pm_length_extended']
    print('leaf {} : length = {} (extension = {})'.format(k, round(l2, 2), round(l2 / l1, 2)))
