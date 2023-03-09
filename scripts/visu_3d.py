import os
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt

from openalea.phenomenal import object as phm_obj
from openalea.maizetrack.phenomenal_display import plot_vmsi, plot_snapshot
from openalea.plantgl.all import Viewer
from openalea.maizetrack.trackedPlant import TrackedSnapshot
from openalea.maizetrack.utils import missing_data, phm3d_to_px2d

from openalea.phenomenal.calibration import Calibration


# function from openalea-incubator/adel
def save_image(scene, image_name='%s/img%04d.%s', directory='.', index=0, ext='png'):
    '''
    Save an image of a scene in a specific directory
    Parameters
    ----------
        - scene: a PlantGL scene
        - image_name: a string template
            The format of the string is dir/img5.png
        - directory (optional: ".") the directory where the images are written
        - index: the index of the image
        - ext : the image format
    Example
    -------
        - Movie:
            convert *.png movie.mpeg
            convert *.png movie.gif
            mencoder "mf://*.png" -mf type=png:fps=25 -ovc lavc -o output.avi
            mencoder -mc 0 -noskip -skiplimit 0 -ovc lavc -lavcopts vcodec=msmpeg4v2:vhq "mf://*.png" -mf type=png:fps=18 -of avi  -o output.avi

    '''

    if not image_name:
        image_name = '{directory}/img{index:0>4d}.{ext}'
    filename = image_name.format(directory=directory, index=index, ext=ext)
    Viewer.frameGL.saveImage(filename)
    return scene,


if __name__ == '__main__':


    # ===== skeleton reprojected on rgb =================================================================================

    df_db = pd.read_csv('data/copy_from_database/images_ZA22.csv')
    df_sf = pd.read_csv('data/copy_from_modulor/snapshot_index_ZA22.csv')

    rgb_path = 'X:/ARCH2022-01-10/'

    sk_path = 'data/copy_from_modulor/skeleton_voxel4_tol1_notop_vis4_minpix100/'
    sk_files = [sk_path + d + '/' + f for d in os.listdir(sk_path) for f in os.listdir(sk_path + d)]

    sk_file = np.random.choice(sk_files)
    sk = phm_obj.VoxelSkeleton.read_from_json_gz(sk_file)

    plantid, task = np.array(sk_file.split('/')[-1].split('.json.gz')[0].split('_'))[[0, -1]].astype(int)

    angle = 60
    selec_db = df_db[(df_db['taskid'] == task) & (df_db['plantid'] == plantid) & (df_db['imgangle'] == angle)]
    img = io.imread(rgb_path + str(task) + '/' + selec_db.iloc[0]['imgguid'] + '.png')[:, :, :3]

    plant = next(p for p in df_sf['plant'].unique() if int(p.split('/')[0]) == plantid)
    selec_sf = df_sf[(df_sf['task'] == task) & (df_sf['plant'] == plant)]
    sf = selec_sf.iloc[0]['shooting_frame']

    calib = Calibration.load('V:/lepseBinaries/Calibration/' + sf + '_calibration.json')
    f_calib = calib.get_projection(id_camera='side', rotation=angle)

    px_polylines = [f_calib(seg.polyline) for seg in sk.segments]

    print(sf)
    plt.imshow(img)
    vxs = f_calib(sk.voxels_position())
    plt.plot(vxs[:, 0], vxs[:, 1], 'w.', markersize=0.5)
    for pl in px_polylines:
        plt.plot(pl[:, 0], pl[:, 1], 'r-')

    # ==========================================================================================

    exp = 'ZB14'

    PATH = 'data/visu_3D/'
    vmsi_folder = 'set_{}_vmsi/'.format(exp)
    image_folder = 'images_{}/'.format(exp)

    if not os.path.isdir(PATH + image_folder):
        os.mkdir(PATH + image_folder)

    files = os.listdir(PATH + vmsi_folder)

    for k, file in enumerate(files):
        print(k)

        vmsi = phm_obj.VoxelSegmentation.read_from_json_gz(PATH + vmsi_folder + file)

        scene = plot_vmsi([vmsi], )

        if not missing_data(vmsi):
            rank_to_index = {l.info['pm_leaf_number_tracking']: k for k, l in enumerate(vmsi.get_leafs())}
            mature_ranks = [-1 + vmsi.get_leaf_order(k).info['pm_leaf_number_tracking']
                            if vmsi.get_leaf_order(k).info['pm_label'] == 'mature_leaf' else -1
                            for k in range(1, 1 + vmsi.get_number_of_leaf())]
            snapshot = TrackedSnapshot(vmsi, metainfo=None, order=None)
            scene = plot_snapshot(snapshot, ranks=mature_ranks)

            _ = save_image(scene, image_name=PATH + image_folder + '{}.png'.format(file))

        else:
            print('missing data', file)






























