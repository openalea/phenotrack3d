from openalea.maizetrack.local_cache import get_metainfos_ZA17
from openalea.maizetrack.utils import get_rgb

from openalea.phenomenal import object as phm_obj
from openalea.maizetrack.utils import phm3d_to_px2d

import os

from PIL import Image


vmsi_folder = 'local_cache/cache_ZA17/segmentation_voxel4_tol1_notop_vis4_minpix100_stem_smooth_tracking/'
all_vmsis = [vmsi_folder + d + '/' + f for d in os.listdir(vmsi_folder) for f in os.listdir(vmsi_folder + d)]
vmsi_files = [v for v in all_vmsis if int(v.split('/')[-1][:4]) == 1429]

vmsis = {}
for f in vmsi_files:
    task = int(f.split('__')[-1].split('.')[0])
    print(task)
    vmsis[task] = phm_obj.VoxelSegmentation.read_from_json_gz(f)

plantid = 1429
metainfos = get_metainfos_ZA17(plantid)

images = {}
for task, sk in vmsis.items():
    print(task)
    m = next(m for m in metainfos if m.task == task)
    img, _ = get_rgb(metainfo=m, angle=60, save=False, plant_folder=False)
    images[task] = img

#all_collars = pd.read_csv('data/visu_3D/stem_tracking/1429.csv')

imgs = []
T, H = [], []
for task in sorted(vmsis.keys()):
    print(task)
    vmsi, img = vmsis[task], images[task]
    m = next(m for m in metainfos if m.task == task)
    #collars = all_collars[all_collars['t'] == m.timestamp]
    xyz = vmsi.get_stem().info['pm_position_tip']
    x, y = phm3d_to_px2d(xyz, sf=m.shooting_frame)[0]
    T.append(m.timestamp)
    H.append(y)
    img = cv2.rectangle(np.float32(img), (int(x - 25), int(y - 25)), (int(x + 25), int(y + 25)), (255, 0, 0), 2)
    img = cv2.circle(np.float32(img), (int(x), int(y)), 7, (255, 0, 0), -1)
    #io.imsave('data/visu_3D/{}.png'.format(task), np.uint8(img))
    imgs.append(img)

imgs_gif = imgs[:-7].copy()
#imgs_gif = [img[200:-200, 250:-250, :] for img in imgs_gif]
imgs_gif = [Image.fromarray(np.uint8(img)) for img in imgs_gif]
fps = 4
imgs_gif[0].save('gif/stem_tracking_1429_{}fps.gif'.format(fps),
              save_all=True,
              append_images=imgs_gif[1:],
              optimize=True,
              duration=1000/fps,
              loop=0)







