from openalea.maizetrack.data_loading import get_metainfos_ZA17, metainfos_to_paths, check_existence, load_plant
from openalea.maizetrack.phenomenal_display import plot_leaves

plantid = 794

metainfos = get_metainfos_ZA17(plantid)
folder = 'local_cache/segmentation_voxel4_tol1_notop_vis4_minpix100_stem_smooth_tracking'
paths = metainfos_to_paths(metainfos, stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), old=False, folder=folder)
metainfos, paths = check_existence(metainfos, paths)
vmsi_list = load_plant(metainfos, paths)


leaves, ranks = [], []
for vmsi in vmsi_list:
    for leaf in vmsi.get_mature_leafs():
        if leaf.info['pm_label'] == 'mature_leaf':
            leaves.append(leaf)
            ranks.append(leaf.info['pm_leaf_number'])
plot_leaves(leaves, ranks)




