from openalea.maizetrack.data_loading import get_metainfos_ZA17, metainfos_to_paths, check_existence
from openalea.phenotracking.maize_track.alignment import tracking
from openalea.phenotracking.maize_track.rank_annotation import annotate

plantid = 1424

save_result = False

print('plantid', str(plantid))
metainfos = get_metainfos_ZA17(plantid)
paths = metainfos_to_paths(metainfos, stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100))
metainfos, paths = check_existence(metainfos, paths)

plant = tracking(metainfos, paths)

print('annotation')

plant = annotate(plant, True)

for snapshot in plant.snapshots:
    print(snapshot.rank_annotation)

if save_result:
    plant.save_rank_annotation()