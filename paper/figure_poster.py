import os

from openalea.maizetrack.local_cache import get_metainfos_ZA17, metainfos_to_paths, check_existence, load_plant
from openalea.maizetrack.phenomenal_display import PALETTE, plot_vmsi_voxel
from openalea.maizetrack.trackedPlant import TrackedPlant
from openalea.plantgl import all as pgl

folder = 'local_cache/cache_ZA17/segmentation_voxel4_tol1_notop_vis4_minpix100_stem_smooth_tracking'
all_files = [folder + '/' + rep + '/' + f for rep in os.listdir(folder) for f in os.listdir(folder + '/' + rep)]

plantids = [313, 316, 329, 330, 336, 348, 424, 439, 461, 474, 794, 832, 905, 907, 915, 925, 931, 940, 959, 1270,
            1276, 1283, 1284, 1301, 1316, 1383, 1391, 1421, 1424, 1434]

plantid = 959 # 1383

metainfos = get_metainfos_ZA17(plantid)
paths = metainfos_to_paths(metainfos, folder=folder)
metainfos, paths = check_existence(metainfos, paths)
vmsi_list = load_plant(metainfos, paths)

plant = TrackedPlant.load_and_check(vmsi_list)
plant.load_rank_annotation()

# [2, 12, 22, 32, 45], cam = (0, 41, manuel bloqué, 112)

for i in [2, 12, 22, 32, 45]:

    snapshot = plant.snapshots[i]

    size = snapshot.voxels_size
    #shapes = []

    organs = [[(0, 0, 0), snapshot.get_stem()]]
    for leaf, rank in zip(snapshot.leaves, snapshot.rank_annotation):
        pass
        organs.append([tuple(PALETTE[rank]), leaf])

    ranks = None

    for col, organ in organs:

        c1, c2, c3 = col

        m = pgl.Material(pgl.Color3(int(c1), int(c2), int(c3)))
        for x, y, z in list(organ.voxels_position())[::1]:
            m = pgl.Material(pgl.Color3(int(c1), int(c2), int(c3)))
            b = pgl.Box(size, size, size)
            vx = pgl.Translated(x, y, z, b)
            vx = pgl.Shape(vx, m)
            shapes.append(vx)

scene = pgl.Scene(shapes)

pgl.Viewer.display(scene)






















