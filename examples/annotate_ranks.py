import copy
import os
import pandas as pd
from openalea.maizetrack.rank_annotation import annotate
from openalea.maizetrack.data_loading import get_metainfos_ZA17, metainfos_to_paths, check_existence
from openalea.maizetrack.trackedPlant import TrackedPlant, align_growing

plantid = 1424

print('plantid', str(plantid))

metainfos = get_metainfos_ZA17(plantid)
paths = metainfos_to_paths(metainfos, stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), old=False)
metainfos, paths = check_existence(metainfos, paths)
plant = TrackedPlant.load_and_check(metainfos, paths)

plant.load_rank_annotation()
plant = annotate(plant, False)

#for snapshot in plant.snapshots:
#    print(snapshot.rank_annotation)
#if save_result:
#    plant.save_rank_annotation()

# ================= load images

plantids = sorted([int(p.split('_')[-1].split('.')[0]) for p in os.listdir('rank_annotation_old')[:-1]])

for plantid in plantids:

    metainfos = get_metainfos_ZA17(plantid)
    paths = metainfos_to_paths(metainfos, stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), old=False)
    metainfos, paths = check_existence(metainfos, paths)
    plant = TrackedPlant.load_and_check(metainfos, paths)

    for angle in [60, 150]:
        plant.load_images(angle)

# ============ create new annotation saving system
# (example case : plantid 1424 - task 6738)

metainfos = get_metainfos_ZA17(plantid)
paths = metainfos_to_paths(metainfos, stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), old=True)
metainfos, paths = check_existence(metainfos, paths)
plant = TrackedPlant.load_and_check(metainfos, paths)
plant.load_rank_annotation()

df_ranks = pd.DataFrame(columns=['task', 'leaf_tip', 'rank'])
for snapshot in plant.snapshots:
    for leaf, r in zip(snapshot.leaves, snapshot.rank_annotation):
        tip = leaf.real_longest_polyline()[-1]
        df_ranks.loc[df_ranks.shape[0]] = [snapshot.metainfo.task, tip, r]

df_ranks.to_csv('rank_annotation/rank_annotation_{}.csv'.format(plantid))

# ============= testing new annotation system

metainfos = get_metainfos_ZA17(plantid)
paths = metainfos_to_paths(metainfos, stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), old=False)
metainfos, paths = check_existence(metainfos, paths)
plant = TrackedPlant.load_and_check(metainfos, paths)
plant.load_rank_annotation()

print('==================================')
df = pd.read_csv('rank_annotation/rank_annotation_{}.csv'.format(plantid))
for snapshot in plant.snapshots:
    task = snapshot.metainfo.task
    dfi = df[(df['task'] == task)]

    new_tips = [str(leaf.real_longest_polyline()[-1]) for leaf in snapshot.leaves]
    old_tips = list(dfi['leaf_tip'])

    if not old_tips:
        pass
        # new date
    else:
        for new_tip in new_tips:
            if new_tip not in old_tips:
                print(plantid, task, 'a new tip appeared')
                # TODO : need to be re-annotated
        for old_tip in old_tips:
            if old_tip not in new_tips:
                print(plantid, task, 'an old tip disappeared')
print('===================================')

plant = annotate(plant, False)











