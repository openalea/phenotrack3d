import os
import cv2

import openalea.phenomenal.object.voxelSegmentation as phm_seg

from openalea.maizetrack.manual_annotation import annotate
from openalea.maizetrack.phenomenal_coupling import phm_to_phenotrack_input
from openalea.maizetrack.trackedPlant import TrackedPlant

from openalea.phenomenal.calibration import Calibration

from datadir import datadir

if __name__ == '__main__':

    timestamps = [int(t) for t in os.listdir(datadir + '/images')]

    phm_segs = []
    for timestamp in timestamps:
        phm_segs.append(phm_seg.VoxelSegmentation.read_from_json_gz(datadir + f'/3d_time_series/{timestamp}.gz'))

    calibration = Calibration.load(datadir + f'/phm_calibration/calibration.json')

    phenotrack_segs, _ = phm_to_phenotrack_input(phm_segs, timestamps)
    trackedplant = TrackedPlant.load(phenotrack_segs)  # useful here for polylines simplification

    trackedplant.mature_leaf_tracking()
    trackedplant.growing_leaf_tracking()

    annot = {}
    angles = [60, 150]
    projections = {a: calibration.get_projection(id_camera='side', rotation=a, world_frame='pot') for a in angles}
    for snapshot, timestamp in zip(trackedplant.snapshots, timestamps):

        annot[timestamp] = {'metainfo': None, 'leaves_info': [], 'leaves_pl': [], 'images': {}}

        for angle in angles:
            img = cv2.cvtColor(cv2.imread(datadir + f'/images/{timestamp}/{angle}.png'), cv2.COLOR_BGR2RGB)
            annot[timestamp]['images'][angle] = img
            ranks = snapshot.leaf_ranks()

        for leaf, r_tracking in zip(snapshot.leaves, ranks):
            mature = leaf.features['mature']

            annot[timestamp]['leaves_pl'].append({a: projections[a](leaf.polyline) for a in angles})
            annot[timestamp]['leaves_info'].append({'mature': mature, 'selected': False, 'rank': r_tracking, 'tip': None})

    annotate(annot)
