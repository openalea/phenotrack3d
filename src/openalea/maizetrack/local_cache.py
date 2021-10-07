"""
Scripts to load data in local cache system (Benoit D), in a similar way to Phenoarch package
"""

import os
import numpy as np

from alinea.phenoarch.cache import snapshot_index
from alinea.phenoarch.cache_client import FileCache

from openalea.phenomenal import object as phm_obj


def get_plantname_ZA17(plantid):

    str_id = str(plantid).rjust(4, '0')

    # get plant name from U:/
    # df_za17 = pd.read_csv('U:/M3P/PHENOARCH/MANIPS/ZA17/Data/WholeDatasetZA17.csv')
    # name = [name for name in df_za17.plantcode.unique() if name[:4] == str_id][0]
    # FASTER : GET NAME FROM A COPY OF PLANT NAMES LIST :
    names = np.load('local_cache/ZA17/plantcodes.npy', allow_pickle=True)
    name = next(name for name in names if name[:4] == str_id)

    return name


def get_metainfos_ZA17(plantid, dates=None):

    name = get_plantname_ZA17(plantid)

    # copy of modulor cache
    index = snapshot_index('ZA17', cache_client=FileCache('local_cache/ZA17'), image_client=None)

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

    return 'local_cache/ZA17/' + folder


def metainfos_to_paths(metainfos, stem_smoothing=True, phm_parameters=(4, 1, 'notop', 4, 100), object='vmsi', old=False,
                       folder=None):

    if folder is None:
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

    """
    load all existing vmsi that correspond to a metainfo : len(vmsi_list) <= len(metainfos)
    metainfos and paths need to have the same length
    each loaded vmsi gets the corresponding metainfo as a new attribute
    """

    print('loading vmsi..')

    vmsi_list = []
    for metainfo, path in zip(metainfos, paths):

        # load vmsi
        vmsi = phm_obj.VoxelSegmentation.read_from_json_gz(path)

        # add metainfo m to vmsi attributes
        setattr(vmsi, 'metainfo', metainfo)

        vmsi_list.append(vmsi)

    return vmsi_list
