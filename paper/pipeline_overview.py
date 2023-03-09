from openalea.maizetrack.local_cache import get_metainfos_ZA17, metainfos_to_paths, check_existence, load_plant
from openalea.maizetrack.utils import get_rgb
from openalea.maizetrack.trackedPlant import TrackedPlant
from openalea.phenomenal import object as phm_obj
from openalea.maizetrack.phenomenal_display import plot_vmsi, plot_vmsi_voxel, plot_vg, plot_snapshot
import copy

import openalea.phenomenal.segmentation as phm_seg
from openalea.maizetrack.trackedPlant import TrackedSnapshot

# ===============================

plantid = 474

metainfos = get_metainfos_ZA17(plantid)

metainfos = [m for m in metainfos if m.daydate != '2017-04-20']  # debug

paths = metainfos_to_paths(metainfos, stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), old=False)

metainfos, paths = check_existence(metainfos, paths)

vmsi_list = load_plant(metainfos, paths)
plant = TrackedPlant.load_and_check(vmsi_list)

plant.align_mature(direction=1, gap=12.365, w_h=0.03, gap_extremity_factor=0.2, n_previous=500)

plant_growing = copy.deepcopy(plant)
plant_growing.align_growing()

# for m in [m for m in metainfos if '2017-05-23' <= m.daydate <= '2017-05-23']:
#     get_rgb(m, 150, main_folder='data/paper', plant_folder=True, save=True, side=True)
#
# for m in [m for m in metainfos if m.daydate in ['2017-04-14', '2017-05-02']]:
#     get_rgb(m, 60, main_folder='data/paper', plant_folder=True, save=True, side=True)

print([m.daydate for m in metainfos if m.daydate not in [s.metainfo.daydate for s in plant.snapshots]])
path = [p for p in paths if '06-04' in p][0]
vmsi = phm_obj.VoxelSegmentation.read_from_json_gz(path)
plot_vmsi([vmsi])

plot_vmsi_voxel(vmsi)




# ====== t=1 :

d = '2017-04-14'
s = next(s for s in plant.snapshots if s.metainfo.daydate == d)
sm = next(s for s in plant_growing.snapshots if s.metainfo.daydate == d)
plot_snapshot(s, colored=False)  # mature vs growing
plot_snapshot(s, colored=True)  # tracking mature
plot_snapshot(sm, colored=True)  # tracking growing

d = '2017-05-02'
s = next(s for s in plant.snapshots if s.metainfo.daydate == d)
sm = next(s for s in plant_growing.snapshots if s.metainfo.daydate == d)
plot_snapshot(s, colored=False)
plot_snapshot(s, colored=True)
plot_snapshot(sm, colored=True)

# d = '2017-05-22'
# s = next(s for s in plant.snapshots if s.metainfo.daydate == d)
# sm = next(s for s in plant_growing.snapshots if s.metainfo.daydate == d)
# plot_snapshot(s, colored=False)
# plot_snapshot(s, colored=True)
# plot_snapshot(sm, colored=True, ranks=[2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 10])

# 05-23 : tige correcte, tige trop basse
d = '2017-05-23'
s = next(s for s in plant.snapshots if s.metainfo.daydate == d)
sm = next(s for s in plant_growing.snapshots if s.metainfo.daydate == d)
plot_snapshot(s, colored=False)
plot_snapshot(s, colored=True)
plot_snapshot(s, colored=True, ranks=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# # 05-24 bug tige (artificial)
# d = '2017-05-24'
# m = next(m for m in metainfos if m.daydate == d)
# path = \
# metainfos_to_paths([m], object='voxelgrid', stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), old=False)[0]
# vx = phm_obj.VoxelGrid.read_from_csv(path)
# path = \
# metainfos_to_paths([m], object='skeleton', stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), old=False)[0]
# sk = phm_obj.VoxelSkeleton.read_from_json_gz(path)
# graph = phm_seg.graph_from_voxel_grid(vx, connect_all_point=True)
# vms = phm_seg.maize_segmentation(sk, graph, z_stem=100)
# vmsi = phm_seg.maize_analysis(vms)
# s = TrackedSnapshot(vmsi, m, order=None)
# plot_snapshot(s, colored=False)
#
# d = '2017-05-24'
# s = next(s for s in plant.snapshots if s.metainfo.daydate == d)
# sm = next(s for s in plant_growing.snapshots if s.metainfo.daydate == d)
# plot_snapshot(s, colored=False)
# plot_snapshot(s, colored=True)
# plot_snapshot(s, colored=True, ranks=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

d = '2017-06-04'  # bug tige 2
m = next(m for m in metainfos if m.daydate == d)
path = metainfos_to_paths([m], stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), old=False)[0]
vmsi = phm_obj.VoxelSegmentation.read_from_json_gz(path)
s = TrackedSnapshot(vmsi, m, order=None)
plot_snapshot(s, colored=False)

d = '2017-06-14'
s = next(s for s in plant.snapshots if s.metainfo.daydate == d)
sm = next(s for s in plant_growing.snapshots if s.metainfo.daydate == d)
plot_snapshot(s, colored=False)
plot_snapshot(s, colored=True)
plot_snapshot(sm, colored=True)

# ====== T : 06-04 (deformation), 06-03

for k, col in enumerate(PALETTE[:20]):
    plt.plot(0, k, '*', c=col / 255.)

# ======== voxelgrid ######### -> 06-14
d = '2017-04-14'
m = [m for m in metainfos if m.daydate == d][0]
print(len([m for m in metainfos if m.daydate == d]))
path = \
metainfos_to_paths([m], object='voxelgrid', stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), old=False)[0]
vx = phm_obj.VoxelGrid.read_from_csv(path)
plot_vg(vx)

# bug tige (artificial)
d = '2017-05-02'
m = [m for m in metainfos if m.daydate == d][1]
path = \
metainfos_to_paths([m], object='voxelgrid', stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), old=False)[0]
vx = phm_obj.VoxelGrid.read_from_csv(path)
path = \
metainfos_to_paths([m], object='skeleton', stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), old=False)[0]
sk = phm_obj.VoxelSkeleton.read_from_json_gz(path)
graph = phm_seg.graph_from_voxel_grid(vx, connect_all_point=True)
vms = phm_seg.maize_segmentation(sk, graph, z_stem=50)
vmsi = phm_seg.maize_analysis(vms)
s = TrackedSnapshot(vmsi, m, order=None)
plot_snapshot(s, colored=False)

# ===== stem shape anomaly example (plantid 330) ========================================================

vx_path = 'local_cache/cache_ZA17/image3d_voxel4_tol1_notop/2017-05-19/0330_DZ_PG_01_ZM4381_WW_Rep_2_06_30_ARCH2017-03-30__2017-05-19__6726.csv'
sk_path = 'local_cache/cache_ZA17/skeleton_voxel4_tol1_notop_vis4_minpix100/2017-05-19/0330_DZ_PG_01_ZM4381_WW_Rep_2_06_30_ARCH2017-03-30__2017-05-19__6726.json.gz'
vx = phm_obj.VoxelGrid.read_from_csv(vx_path)
sk = phm_obj.VoxelSkeleton.read_from_json_gz(sk_path)
graph = phm_seg.graph_from_voxel_grid(vx, connect_all_point=True)

vms = phm_seg.maize_segmentation(sk, graph, z_stem=260)
vmsi = phm_seg.maize_analysis(vms)

s = TrackedSnapshot(vmsi, m, order=None)
plot_snapshot(s, colored=False)