import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openalea.phenomenal import object as phm_obj
from openalea.maizetrack.utils import missing_data

from maizetrack.scripts.visu_3d import save_image
from openalea.maizetrack.phenomenal_display import plot_vmsi

PATH = 'data/copy_from_modulor/'

exp = 'ZB14'

seg_path = PATH + 'cache_{}/segmentation_voxel4_tol1_notop_vis4_minpix100_stem_smooth_tracking/'.format(exp)

all_files = [seg_path + d + '/' + f for d in os.listdir(seg_path) for f in os.listdir(seg_path + d)]
plantids = np.unique([int(f.split('/')[-1][:4]) for f in all_files])

# contains daydate, task, timestamp, shooting_frame
index = pd.read_csv(PATH + 'cache_{0}/snapshot_index_{0}.csv'.format(exp))

for plantid in plantids:

    print('=====================')

    files = [f for f in all_files if int(f.split('/')[-1][:4]) == plantid]

    for file in files:
    #for file in np.random.choice(files, 5, replace=False):

        plant = next(p for p in index['plant'] if int(p[:4]) == plantid)
        task = int(file.split('/')[-1].split('.json')[0].split('_')[-1])
        index_row = index[(index['plant'] == plant) & (index['task'] == task)].iloc[0]

        print(plantid, index_row.daydate)
        vmsi = phm_obj.VoxelSegmentation.read_from_json_gz(file)

        scene = plot_vmsi([vmsi], )
        _ = save_image(scene, image_name='data/visu_3D/images_ZB14/{}_{}.png'.format(index_row['daydate'], plantid))










