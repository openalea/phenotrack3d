import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from openalea.maizetrack.trackedPlant import TrackedPlant
from openalea.maizetrack.phenomenal_coupling import phm_to_phenotrack_input
from openalea.maizetrack.display import plot_polylines

from alinea.phenoarch.cache import Cache
from alinea.phenoarch.platform_resources import get_ressources
from alinea.phenoarch.meta_data import plant_data


exp = 'ZA22'
cache_client, image_client, binary_image_client, calibration_client = get_ressources(exp, cache='V:',
                                                                                     studies='Z:',
                                                                                     nasshare2='Y:')
parameters = {'reconstruction': {'voxel_size': 4, 'frame': 'pot'},
              'collar_detection': {'model_name': '3exp_xyside_99000'},
              'segmentation': {'force_stem': True}}
cache = Cache(cache_client, image_client, binary_image_client=binary_image_client,
              calibration_client=calibration_client, parameters=parameters)
index = cache.snapshot_index()

plants = list(plant_data(exp)['plant'])

# ===== leaf extension ============================================

plant = '1658/ZM1584/CML108/lepse/EXPOSE/WW/Rep_5/28_38/ARCH2022-01-10'
task = 5172
meta_snapshots = index.get_snapshots(index.filter(plant=plant, nview=13), meta=True)
m = next(m for m in meta_snapshots if m.task == task)

for angle in [k * 30 for k in range(12)]:
    bin_path = next(p for p, v, a in zip(m.binary_path, m.view_type, m.camera_angle) if v == 'side' and a == angle)
    bin = cache.binary_image_client.imread(bin_path)
    cv2.imwrite('maizetrack/examples/data/leaf_extension/{}.png'.format(angle), bin)

# for maizetrack
seg = cache.load_segmentation(m)
seg.write_to_json_gz('maizetrack/examples/data/leaf_extension/segmentation.gz')


# ===============================================================

# which plant as the most time points :
dir = f'V:/phenoarch_cache/cache_{exp}/phenomenal/segmentation_voxel4_tol1_notop_pot_vis4_minpix100_force-stem'
files = [f for d in os.listdir(dir) for f in os.listdir(dir + '/' + d)]
plantids = [int(f.split('_')[0]) for f in files]
dic = {plantid: plantids.count(plantid) for plantid in set(plantids)}
dic = dict(sorted(dic.items(), key=lambda item: item[1]))


plantid = 515
plant = next(p for p in plants if int(p.split('/')[0]) == plantid)
meta_snapshots = index.get_snapshots(index.filter(plant=plant, nview=13), meta=True)

print('loading segmentations...')
phm_segs, timestamps = [], []
for m in meta_snapshots[::10]:
    try:

        print(m.shooting_frame)

        seg = cache.load_segmentation(m)
        phm_segs.append(seg)

        timestamp = int(m.timestamp)
        timestamps.append(timestamp)

        os.mkdir('maizetrack/examples/data/images/' + str(timestamp))

        for angle in [60, 150]:
            image_path = next(p for p, v, a in zip(m.path, m.view_type, m.camera_angle) if v == 'side' and a == angle)
            img = cv2.cvtColor(image_client.imread(image_path), cv2.COLOR_BGR2RGB)
            plt.imsave(f'maizetrack/examples/data/images/{timestamp}/{angle}.png', img)

        # save it
        # seg.write_to_json_gz(f'maizetrack/examples/data/3d_time_series/{timestamp}.gz')
    except:
        print('{}: cannot load seg'.format(m.daydate))

phenotrack_segs, checks_stem = phm_to_phenotrack_input(phm_segs, timestamps)

trackedplant = TrackedPlant.load(phenotrack_segs)

trackedplant.mature_leaf_tracking(start=5, gap=12, w_h=0.03, w_l=0.004, gap_extremity_factor=0.2, n_previous=5000)
trackedplant.growing_leaf_tracking()
ranks, checks_continuity = trackedplant.output()

r = 0
for phm_seg, check_stem in zip(phm_segs, checks_stem):
    phm_ranks = np.arange(1, 1 + phm_seg.get_number_of_leaf())
    if check_stem:
        phm_tracking_ranks = ranks[r]
        r += 1
    else:
        phm_tracking_ranks = [0] * len(phm_ranks)

ref_skeleton = trackedplant.get_ref_skeleton()
plot_polylines([v.polyline for v in ref_skeleton.values()], np.array(list(ref_skeleton.keys())) + 1)

polylines, ranks = [], []
for snapshot in trackedplant.snapshots:
    for r, leaf in zip(snapshot.leaf_ranks(), snapshot.leaves):
        if True:  #leaf.features['mature']:
            polylines.append(leaf.polyline)
            ranks.append(r)
plot_polylines(polylines, ranks)























