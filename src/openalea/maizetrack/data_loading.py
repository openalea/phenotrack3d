import os
import numpy as np
from skimage import io

from alinea.phenoarch.cache import snapshot_index
from alinea.phenoarch.cache_client import FileCache

from openalea.phenomenal import object as phm_obj


def get_rgb(metainfo, angle, main_folder='rgb', plant_folder=True, save=True, side=True):

    plantid = int(metainfo.plant[:4])

    if plant_folder:
        img_folder = main_folder + '/' + str(plantid)
    else:
        img_folder = main_folder
    if not os.path.isdir(img_folder):
        os.mkdir(img_folder)

    if side:
        if angle == 0:
            # 0 can both correspond to side and top ! Need to select side image
            i_angles = [i for i, a in enumerate(metainfo.camera_angle) if a == angle]
            i_angle = [i for i in i_angles if metainfo.view_type[i] == 'side'][0]
        else:
            i_angle = metainfo.camera_angle.index(angle)
    else:
        i_angles = [i for i, a in enumerate(metainfo.camera_angle) if a == 0]
        i_angle = [i for i in i_angles if metainfo.view_type[i] == 'top'][0]

    # TODO : 'plantid' or 'id' ??
    img_name = 'id{}_d{}_t{}_a{}.png'.format(plantid, metainfo.daydate, metainfo.task, angle)
    path = img_folder + '/' + img_name

    if img_name in os.listdir(img_folder):
        img = io.imread(path)
    else:
        url = metainfo.path_http[i_angle]
        img = io.imread(url)[:, :, :3]

        if save:
            io.imsave(path, img)

    return img, path


def get_plantname_ZA17(plantid):

    str_id = str(plantid).rjust(4, '0')

    # get plant name from U:/
    # df_za17 = pd.read_csv('U:/M3P/PHENOARCH/MANIPS/ZA17/Data/WholeDatasetZA17.csv')
    # name = [name for name in df_za17.plantcode.unique() if name[:4] == str_id][0]
    # FASTER : GET NAME FROM A COPY OF PLANT NAMES LIST :
    names = np.load('plantcodes.npy', allow_pickle=True)
    name = next(name for name in names if name[:4] == str_id)

    return name


def get_metainfos_ZA17(plantid, dates=None):

    name = get_plantname_ZA17(plantid)

    # copy of modulor cache
    index = snapshot_index('ZA17', cache_client=FileCache(''), image_client=None)

    metainfos = []

    if dates is None:
        selection = index.filter(plant=name)
        metainfos = index.get_snapshots(selection)
    else:
        for date in dates:
            # modulor cache, name, date --> metainfo
            selection = index.filter(plant=name, daydate=date)
            metainfo = index.get_snapshots(selection)
            metainfos.append(metainfo[0])

    return metainfos


def get_folder_ZA17(stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), object='vmsi', old=False):

    p0, p1, p2, p3, p4 = phm_parameters

    # folder name
    if object == 'voxelgrid':
        folder = 'image3d_voxel{}_tol{}_{}'.format(p0, p1, p2)
    elif object == 'skeleton':
        folder = 'skeleton_voxel{}_tol{}_{}_vis{}_minpix{}'.format(p0, p1, p2, p3, p4)
    if object == 'vmsi':
        if old:
            folder = 'vmsi_voxel{}_tol{}_{}_vis{}_minpix{}'.format(p0, p1, p2, p3, p4)
            if stem_smoothing:
                folder += '_modified'
        else:
            folder = 'segmentation_voxel{}_tol{}_{}_vis{}_minpix{}'.format(p0, p1, p2, p3, p4)
            if stem_smoothing:
                folder += '_stem_smooth'
            else:
                folder += '_no_stem_smooth'

    return folder

def metainfos_to_paths(metainfos, stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), object='vmsi', old=False):

    folder = get_folder_ZA17(stem_smoothing=stem_smoothing, phm_parameters=phm_parameters, object=object, old=old)

    # file extension
    if object == 'vmsi' or object == 'skeleton':
        extension = '.json.gz'
    elif object == 'voxelgrid':
        extension = '.csv'
    else:
        print('unknown object')

    paths = []
    for metainfo in metainfos:
        name = metainfo.plant.replace('/', '_') + '__' + metainfo.daydate + '__' + str(metainfo.task) + extension
        path = folder + '/' + metainfo.daydate + '/' + name
        paths.append(path)

    return paths


def check_existence(metainfos, paths):

    metainfos2, paths2 = [], []

    for metainfo, path in zip(metainfos, paths):
        if os.path.isfile(path):
            metainfos2.append(metainfo)
            paths2.append(path)

    return metainfos2, paths2


def load_plant(metainfos, paths):

    # load all existing vmsi that correspond to a metainfo : len(vmsi_list) <= len(metainfos)
    # metainfos and paths need to have the same length
    # each loaded vmsi gets the corresponding metainfo as a new attribute.

    vmsi_list = []
    for metainfo, path in zip(metainfos, paths):

        # load vmsi
        vmsi = phm_obj.VoxelSegmentation.read_from_json_gz(path)

        # add metainfo m to vmsi attributes
        setattr(vmsi, 'metainfo', metainfo)

        vmsi_list.append(vmsi)

    return vmsi_list


def missing_data(vmsi):

    # Check if there are some missing data in vmsi.

    missing = False

    stem_needed_info = ['pm_z_base', 'pm_z_tip']
    if not all([k in vmsi.get_stem().info for k in stem_needed_info]):
        missing = True

    leaf_needed_info = ['pm_position_base', 'pm_z_tip', 'pm_label', 'pm_azimuth_angle',
                        'pm_length', 'pm_insertion_angle', 'pm_z_tip']
    for leaf in vmsi.get_leafs():
        if not all([k in leaf.info for k in leaf_needed_info]):
            missing = True

    return missing



### leaf redundancy
#leaf_to_remove = []
#base_pos = [vmsi.get_leaf_order(k+1).info['pm_position_base'] for k in range(len(vmsi.get_leafs()))]
#z_tip = [vmsi.get_leaf_order(k+1).info['pm_z_tip'] for k in range(len(vmsi.get_leafs()))]
#for i in range(len(base_pos)-1):
#    if base_pos[i] == base_pos[i+1]:
#        if z_tip[i] < z_tip[i+1]:
#            leaf_to_remove.append(i)
#        else:
#            leaf_to_remove.append(i+1)


