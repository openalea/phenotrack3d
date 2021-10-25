""" script to annotate leaf ranks manually """

from openalea.maizetrack.rank_annotation import annotate
from openalea.maizetrack.local_cache import get_metainfos_ZA17, metainfos_to_paths, check_existence, load_plant
from openalea.maizetrack.trackedPlant import TrackedPlant

plantid = 1434

print('plantid', str(plantid))

metainfos = get_metainfos_ZA17(plantid)
folder = 'local_cache/cache_ZA17/segmentation_voxel4_tol1_notop_vis4_minpix100_stem_smooth_tracking'
paths = metainfos_to_paths(metainfos, folder=folder)
metainfos, paths = check_existence(metainfos, paths)
vmsi_list = load_plant(metainfos, paths)
plant = TrackedPlant.load_and_check(vmsi_list)

plant.align_mature(direction=1, gap=12.365, w_h=0.03, old_method=False,
                   gap_extremity_factor=0.2, n_previous=50)
plant.align_growing()
# TODO : save method to move here
plant = annotate(plant, init='all')

#plant.save_rank_annotation()









