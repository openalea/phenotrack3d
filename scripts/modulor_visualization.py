from openalea.maizetrack.data_loading import get_metainfos_ZA17, metainfos_to_paths, check_existence, load_plant
from openalea.maizetrack.phenomenal_display import plot_leaves, plot_vmsi
from openalea.maizetrack.trackedPlant import TrackedPlant
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

folder = 'local_cache/ZA17/segmentation_voxel4_tol1_notop_vis4_minpix100_stem_smooth_tracking'
all_files = [folder + '/' + rep + '/' + f for rep in os.listdir(folder) for f in os.listdir(folder + '/' + rep)]
plantids = list(set([int(f.split('/')[-1][:4]) for f in all_files]))

# ===================================== visu

plantid = 794

metainfos = get_metainfos_ZA17(plantid)
paths = metainfos_to_paths(metainfos, stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), old=False, folder=folder)
metainfos, paths = check_existence(metainfos, paths)
vmsi_list = load_plant(metainfos, paths)

# sort vmsi objects by time
timestamps = [vmsi.metainfo.timestamp for vmsi in vmsi_list]
order = sorted(range(len(timestamps)), key=lambda k: timestamps[k])
vmsi_list = [vmsi_list[i] for i in order]

leaves, ranks = [], []
for vmsi in vmsi_list:
    for leaf in vmsi.get_mature_leafs():
        if leaf.info['pm_label'] == 'mature_leaf':
            leaves.append(leaf)
            ranks.append(leaf.info['pm_leaf_number_tracking'] - 1)
plot_leaves(leaves, ranks)


# =============================== ZB14

import os
from openalea.phenomenal import object as phm_obj

folder = 'local_cache/ZB14/segmentation_voxel4_tol1_notop_vis4_minpix100_stem_smooth_tracking'
all_files = [folder + '/' + rep + '/' + f for rep in os.listdir(folder) for f in os.listdir(folder + '/' + rep)]
plantids = list(set([int(f.split('/')[-1][:4]) for f in all_files]))

plantid = 57
plantid_files = [f for f in all_files if int(f.split('/')[-1][:4]) == plantid]

vmsi_list = []
for path in plantid_files:
    vmsi = phm_obj.VoxelSegmentation.read_from_json_gz(path)
    vmsi_list.append(vmsi)

leaves, ranks = [], []
for vmsi in vmsi_list:
    for leaf in vmsi.get_mature_leafs():
        if leaf.info['pm_label'] == 'mature_leaf':
            leaves.append(leaf)
            ranks.append(leaf.info['pm_leaf_number_tracking'] - 1)
plot_leaves(leaves, ranks)

# ================================== alignment parameters optimisation

def dataset_mean_distance(w_h, step=1):
    v = np.load('leaf_vectors.npy', allow_pickle=True)
    dists = []
    for vecs in v:
        vecs2 = np.array([[np.cos(a/360*2*np.pi), np.sin(a/360*2*np.pi), w_h * h] for h, _, a in vecs])
        dists += [np.linalg.norm(vecs2[k] - vecs2[k + step]) for k in range(len(vecs2) - step)]
    return np.mean(dists)


df_opti = pd.DataFrame(columns=['plantid', 'w_h', 'h_gap', 'direction', 'n', 'accuracy', 'n_ref', 'accuracy_ref'])

for plantid in plantids:

    print(plantid)

    metainfos = get_metainfos_ZA17(plantid)
    paths = metainfos_to_paths(metainfos, stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), old=False, folder=folder)
    metainfos, paths = check_existence(metainfos, paths)
    vmsi_list = load_plant(metainfos, paths)

    plant = TrackedPlant.load_and_check(vmsi_list)

    plant.load_rank_annotation()

    for direction in [1, -1]:
        for w_h in [100, 30, 10, 3, 1, 0.3, 0.1, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]:

            d_mean = dataset_mean_distance(w_h, step=1)

            for h_gap in [0.1, 0.3, 0.5, 1, 2, 3, 5, 7, 10, 15]:

                for gap_ext in [0.2, 0.35, 0.5, 0.75, 1.]:

                    gap = d_mean * h_gap

                    plant.align_mature(direction=direction, gap=gap, w_h=w_h, old_method=False,
                                       gap_extremity_factor=gap_ext)

                    df = plant.get_dataframe(load_anot=False)  # only non-abnormal dates
                    df1 = df[df['mature']]
                    df2 = df1[df1['rank_annotation'] != -1]  # only annotated leaves (don't consider > 05-30...)
                    n = len(df2)
                    accuracy = len(df2[df2['rank_annotation'] == df2['rank_tracking']]) / n

                    ref_sk = plant.get_ref_skeleton() # TODO : only works after .align()
                    # only when the selected leaf of ref_sk has been annotated (not always the case for late leaves...)
                    ref_sk_correct = [r_trk == leaf.rank_annotation for r_trk, leaf in ref_sk.items() if leaf.rank_annotation != -2]
                    n_ref = len(ref_sk_correct)
                    accuracy_ref = sum(ref_sk_correct) / n_ref

                    df_opti.loc[df_opti.shape[0]] = [plantid, w_h, h_gap, direction, n, accuracy, n_ref, accuracy_ref]

                    print(plantid, direction, w_h, h_gap, round(accuracy, 3))


df_plot = df_opti[df_opti['h_gap'] < 20]
plt.plot(df_plot['h_gap'], df_plot['accuracy'], 'k*')

# ============ smal test

import numpy as np

a = 180
a = a / 360 * 2 * np.pi


# =============







