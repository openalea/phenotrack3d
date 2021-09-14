from openalea.maizetrack.data_loading import get_metainfos_ZA17, metainfos_to_paths, check_existence, load_plant
from openalea.maizetrack.phenomenal_display import plot_leaves

plantid = 313

metainfos = get_metainfos_ZA17(plantid)
paths = metainfos_to_paths(metainfos, stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), old=False)
metainfos, paths = check_existence(metainfos, paths)
vmsi_list = load_plant(metainfos, paths)


leaves, ranks = [], []
for vmsi in vmsi_list:
    for leaf in vmsi.get_mature_leafs():
        if leaf.info['pm_label'] == 'mature_leaf':
            leaves.append(leaf)
            ranks.append(leaf.info['pm_leaf_number'])
plot_leaves(leaves, ranks)







# =============== same z space ? ====

plantids = [313, 316, 329, 330, 336, 348, 424, 439, 461, 474, 794, 832, 905, 907, 915, 925, 931, 940, 959, 1270,
                  1276, 1283, 1284, 1301, 1316, 1383, 1391, 1421, 1424, 1434]

d = {}

for plantid in plantids:

    print(plantid)

    metainfos = get_metainfos_ZA17(plantid)
    paths = metainfos_to_paths(metainfos, stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), old=False)
    metainfos, paths = check_existence(metainfos, paths)
    vmsi_list = load_plant(metainfos, paths)

    d[plantid] = {'elcom_2_c1_wide': [], 'elcom_2_c2_wide': []}
    for m, vmsi in zip(metainfos, vmsi_list):
        d[plantid][m.shooting_frame].append(vmsi.get_stem().get_highest_polyline().polyline[0][2])

import numpy as np
for sf in ['elcom_2_c1_wide', 'elcom_2_c2_wide']:
    for plantid in plantids:
        z = d[plantid][sf]
        print(plantid, len(z), np.median(z), np.min(z) - np.median(z), np.max(z) - np.median(z))

# =========== z space ==== plantid 905

from openalea.maizetrack.phenomenal_display import plot_vmsi
vmsi_list2 = [v for v, m in zip(vmsi_list, metainfos) if m.shooting_frame == 'elcom_2_c1_wide']
[vmsi.get_stem().get_highest_polyline().polyline[0][2] for vmsi in vmsi_list2]

for v,m in zip(vmsi_list, metainfos):
    if m.shooting_frame == 'elcom_2_c1_wide':
        print(m.daydate, v.get_stem().get_highest_polyline().polyline[0][2])








