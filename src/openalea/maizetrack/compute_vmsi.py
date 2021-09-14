import matplotlib.pyplot as plt

import openalea.phenomenal.segmentation as phm_seg

from openalea.maizetrack.data_loading import *
from openalea.maizetrack.stem_correction import smoothing_function, ear_anomaly, abnormal_stem

#import pickle
#with open('filename.pickle', 'wb') as handle:
#    pickle.dump([sk, graph], handle, protocol=pickle.HIGHEST_PROTOCOL)
from openalea.maizetrack.utils import missing_data


def compute_vmsi(plantid, phm_parameters=(4, 1, 'notop', 4, 100), daydate_max=None, smoothing_function=None):

    # searches voxel and skeleton objects. If they are found : creates and saves the corresponding vmsi object.
    # metainfos, f_stem : useful to arbitrary chose the stem height (height = f_stem(metainfos[i].timestamp))

    metainfos = get_metainfos_ZA17(plantid)

    path_dict = {task: {'voxelgrid':'', 'skeleton':'', 'vmsi':''} for task in sorted([m.task for m in metainfos])}
    for object in ['voxelgrid', 'skeleton', 'vmsi']:
        paths = metainfos_to_paths(metainfos, object=object, stem_smoothing=False, phm_parameters=phm_parameters)
        paths = [p for p in paths if os.path.isfile(p)]
        for path in paths:
            task = int(path.split('.')[0][-4:])
            path_dict[task][object] = path

    # keep only task when vmsi can/need to be computed
    tasks = list(path_dict.keys())
    for task in tasks:
        objs = path_dict[task]
        #if objs['voxelgrid'] == '' or objs['skeleton'] == '' or objs['vmsi'] != '':
        if objs['voxelgrid'] == '' or objs['skeleton'] == '':
            path_dict.pop(task, None)

    # filter date
    if daydate_max is not None:
        tasks = list(path_dict.keys())
        for task in tasks:
            daydate = path_dict[task]['voxelgrid'].split('/')[1]
            if daydate > daydate_max:
            #if daydate < '2017-05-28' or daydate > '2017-06-08':
                path_dict.pop(task, None)

    vmsi_dict = dict()
    print('computing graphs')
    for task in list(path_dict.keys()):

        daydate = next(m.daydate for m in metainfos if m.task == task)

        print(daydate)

        vmsi_folder = get_folder_ZA17(stem_smoothing=False, phm_parameters=phm_parameters, object='vmsi') + '/' + daydate

        # checks if vmsi/date folder exists
        if not os.path.isdir(vmsi_folder):
            os.mkdir(vmsi_folder)

        if smoothing_function is None:
            z = None
        else:
            timestamp = next(m.timestamp for m in metainfos if m.task == task)
            z = smoothing_function(timestamp)

        try:
            vx = phm_obj.VoxelGrid.read_from_csv(path_dict[task]['voxelgrid'])
            sk = phm_obj.VoxelSkeleton.read_from_json_gz(path_dict[task]['skeleton'])
            graph = phm_seg.graph_from_voxel_grid(vx, connect_all_point=True)
            vms = phm_seg.maize_segmentation(sk, graph, z_stem=z)
            vmsi = phm_seg.maize_analysis(vms)

            # saves vmsi
            m = next(m for m in metainfos if m.task == task)
            vmsi_path = metainfos_to_paths([m], object='vmsi', stem_smoothing=(smoothing_function is not None), phm_parameters=phm_parameters)[0]
            vmsi.write_to_json_gz(vmsi_path)

        except:
            print("phenomenal pipeline doesn't work for task {}".format(task))




def stem_smoothing(plantid, phm_parameters=(4, 1, 'notop', 4, 100)):

    # LOAD VMSI

    metainfos = get_metainfos_ZA17(plantid)
    paths = metainfos_to_paths(metainfos, stem_smoothing=False, phm_parameters=phm_parameters)
    metainfos, paths = check_existence(metainfos, paths)
    vmsi_list = load_plant(metainfos, paths)
    print('vmsi found for {}/{} metainfos'.format(len(vmsi_list), len(metainfos)))

    # sort vmsi by time
    timestamps = [vmsi.metainfo.timestamp for vmsi in vmsi_list]
    order = sorted(range(len(timestamps)), key=lambda k: timestamps[k])
    vmsi_list = [vmsi_list[i] for i in order]

    # REMOVE ABNOMALY

    checks_data = [not missing_data(v) for v in vmsi_list]
    print('{} vmsi to remove because of missing data'.format(len(checks_data) - sum(checks_data)))
    checks_stem = [not b for b in abnormal_stem(vmsi_list)]
    print('{} vmsi to remove because of stem shape abnormality'.format(len(checks_stem) - sum(checks_stem)))
    checks = list((np.array(checks_data) * np.array(checks_stem)).astype(int))
    vmsi_list = [vmsi for check, vmsi in zip(checks, vmsi_list) if check]

    timestamp_list = [v.metainfo.timestamp for v in vmsi_list]
    z_stem = np.array([v.get_stem().info['pm_position_tip'][2] for v in vmsi_list])
    highest_insertion = [max([l.info['pm_position_base'][2] for l in vmsi.get_leafs()]) for vmsi in vmsi_list]

    dz = 0.2 * (np.max(z_stem) - np.min(z_stem))
    i_anomaly = ear_anomaly(z_stem, dz)

    z_stem2 = [highest_insertion[i] if i in i_anomaly else z_stem[i] for i in range(len(z_stem))]
    i_anomaly = ear_anomaly(z_stem2, dz)
    z_stem2 = [highest_insertion[i] if i in i_anomaly else z_stem2[i] for i in range(len(z_stem2))]

    f, t_max = smoothing_function(timestamp_list, z_stem2, dz)

    plt.figure(plantid)
    plt.plot(timestamp_list, z_stem, 'k*')
    plt.plot(timestamp_list, z_stem2, 'k-')
    plt.plot(timestamp_list, highest_insertion, 'y*')
    plt.plot(timestamp_list, f(timestamp_list), 'r-')
    plt.figure()

    return f



if __name__ == '__main__':

    #for plantid in [1014, 995, 876, 16, 1127, 672, 709, 911, 948]:
    #for plantid in [314]:
    #for plantid in [314, 781, 782, 790, 791, 792, 793, 796, 797, 798, 799,
    #           800, 801, 831, 833, 834, 835, 901, 902, 910, 911,
    #            912, 913, 914, 916]:

    TEST_SET =  [348,1301,832,1276,1383,1424,940,1270,931,925,474,794,1283,330,1421,
            907,316,1284,336,439,959,915,1316,1434,905,313,1391,461,424,329,784,1398,823,1402,430,
            1416,1309,1263,811,1413,466,478,822,918,937,1436,1291,470,327]

    plantids = [1276, 948, 803, 931, 827, 1424, 1435, 479, 449, 318, 348,
     1266, 705, 1662, 1668, 715, 1031, 1525, 560, 1021, 1553, 1165,
     584, 1170, 209, 115, 231, 101]

    # 715 ->

    plantids = [p for p in plantids if p not in TEST_SET]
    print(len(plantids))

    plantids = [1]

    for plantid in plantids:

        print('plantid', plantid)

        phm_parameters = (4, 1, 'notop', 4, 100)
        compute_vmsi(plantid, phm_parameters=phm_parameters, smoothing_function=None)
        f = stem_smoothing(plantid, phm_parameters=phm_parameters)
        #compute_vmsi(plantid, phm_parameters=phm_parameters, smoothing_function=f)






