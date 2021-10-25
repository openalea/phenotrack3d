"""
An example script to determine leaf ranks in a time-series of 3D Phenomenal segmentation (vmsi) objects, for a maize
plant.
"""

import pandas as pd
from openalea.maizetrack.trackedPlant import TrackedPlant
import openalea.phenomenal.segmentation as phm_seg

# ===== load data ====================================================================================================

data_folder = 'maizetrack/examples/data/plant2/'

# list of metainfos for each time point
metainfos = pd.read_csv(data_folder + 'metainfos.csv')
metainfos = [metainfos.iloc[i] for i in range(len(metainfos))]

# list of corresponding Phenomenal segmentations (vmsi) for each time point
vmsi_folder = data_folder + 'vmsi/'
vmsi_list = [phm_seg.VoxelSegmentation.read_from_json_gz(vmsi_folder + '{}.json.gz'.format(m.timestamp)) for m in metainfos]

# attaching metainfos to vmsi.
# The resulting list of vmsi don't need to be ordered by time, since each vmsi is attached to timestamp data contained
# in its metainfo : time ordering is handled later in TrackedPlant object.
for metainfo, vmsi in zip(metainfos, vmsi_list):
    setattr(vmsi, 'metainfo', metainfo)

# ===== run tracking =================================================================================================

plant = TrackedPlant.load_and_check(vmsi_list)
plant.align_mature(direction=1, gap=12.365, w_h=0.03, w_l=0.004, gap_extremity_factor=0.2, n_previous=500)
plant.align_growing()

# ===== show results ==================================================================================================

# dict {timestamp : vmsi} where each leaf of each vmsi contains 'pm_leaf_number_tracking' in its 'info' attribute. It
# corresponds to the rank of a leaf after tracking (a value of 0 corresponds to a leaf considered as an anomaly which
# was not ranked during the tracking process).
new_vmsi_list = plant.dump()

# compute a dataframe summarizing leaves info (rank, length, etc.)
df = plant.get_dataframe(load_anot=False)
#df.to_csv(data_folder + 'tracking.csv', index=False)

# display leaf tracking result in 3D with plantGL
# a) mature leaves
plant.display(only_mature=True)
# b) all leaves
plant.display(only_mature=False)

# extract leaves of rank = 8
leaves = [leaf for s in plant.snapshots for leaf in s.leaves if leaf.info['pm_leaf_number'] == 8]

# get leaves at date 2017-05-15
snapshot = next(s for s in plant.snapshots if s.metainfo.daydate == '2017-05-15')
leaves = snapshot.leaves


