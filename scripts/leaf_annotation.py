from openalea.maizetrack.utils import best_leaf_angles, get_rgb
from openalea.maizetrack.data_loading import metainfos_to_paths, get_metainfos_ZA17, load_plant
from openalea.maizetrack.utils import phm3d_to_px2d, shooting_frame_conversion, rgb_and_polylines
#from openalea.phenotracking.test_maize_track.plant_align import simplify, TrackedSnapshot
from openalea.maizetrack.compute_vmsi import compute_vmsi
from openalea.maizetrack.stem_correction import savgol_smoothing_function



import pandas as pd
import numpy as np
import os
import json
from skimage import io
import cv2
import matplotlib.pyplot as plt


def extract_json_stem(dic, metainfos):
    res = dict()
    for key in dic.keys():

        regions = dic[key]['regions']

        if regions != []:
            shape = regions[0]['shape_attributes']
            x, y = shape['cx'], shape['cy']
            name = dic[key]['filename']
            task = int(name[name.find('t') + len('t'):name.rfind('_a')])
            timestamp = next(m.timestamp for m in metainfos if m.task == task)

            res[timestamp] = 2448 - y
    return res

def read_leaf_annotation(plantid):

    # leaf annotation
    with open('data/rgb_leaf_annotation/{}.json'.format(plantid)) as f:
        d = json.load(f)

    metainfos = get_metainfos_ZA17(plantid)

    df = pd.DataFrame(columns=['timestamp', 'rank', 'length', 'height'])

    plantids = []
    anot_lengths = []
    for key in d.keys():

        # extract metainfo and leaf rank
        name = d[key]['filename']
        task = int(name[name.find('_t') + len('_t'):name.rfind('_a')])
        rank = int(name[name.find('_r') + len('.r'):name.rfind('.png')])
        metainfo = next(m for m in metainfos if m.task == task)
        timestamp = metainfo.timestamp
        conversion_factor, pot = shooting_frame_conversion(metainfo.shooting_frame)

        if d[key]['regions'] != []:

            # extract polyline
            shape = d[key]['regions'][0]['shape_attributes']
            x, y = shape['all_points_x'], shape['all_points_y']
            pl = np.array([x, y]).T

            # extract data from polyline
            length = np.sum([np.linalg.norm(np.array(pl[k]) - np.array(pl[k + 1])) for k in range(len(pl) - 1)])
            length *= conversion_factor

            height = (pot - pl[-1][1]) * conversion_factor
            df.loc[df.shape[0]] = [timestamp, rank, length, height]

    return df



def image_leaf_annotation(plant, get_mature=True, get_growing=True, load_stem=False):

    # put all leaves in a dataframe:

    df = pd.DataFrame(columns=['s', 'l', 'rank', 'timestamp', 'mature'])
    for s, snapshot in enumerate(plant.snapshots):
        for l, leaf in enumerate(snapshot.leaves):
            mature = leaf.info['pm_label'] == 'mature_leaf'
            timestamp = snapshot.metainfo.timestamp
            df.loc[df.shape[0]] = [s, l, snapshot.rank_annotation[l], timestamp, mature]

    # keep only the leaves that need to be annotated:

    # ranks = 1 -> n
    df['rank'] += 1

    rows_to_keep = []
    ranks = sorted([r for r in df['rank'].unique() if r != 0]) # anomaly = 0 because rank start at r=1
    for r in ranks:

        dfr = df[df['rank'] == r].sort_values('timestamp')

        # keep all growing leaves
        if get_growing:
            rows = dfr[dfr['mature'] == False].index
            rows_to_keep += list(rows)

        # keep only 1 mature (if exists)
        if get_mature:
            rows = dfr[dfr['mature'] == True]
            if len(rows) != 0:
                rows_to_keep.append(rows[:3].index[-1])

    df = df.loc[rows_to_keep]

    # add stem height:

    if load_stem:
        plantid = plant.plantid
        with open('data/stem_annotation/stem_{}.json'.format(plantid)) as f:
            d = json.load(f)
        metainfos = get_metainfos_ZA17(plantid)
        stem_height = extract_json_stem(d, metainfos)
        #f_ligu_interpolated = savgol_smoothing_function(list(stem_height.keys()), list(stem_height.values()))
        #plt.plot(stem_height.keys(), stem_height.values())
        #plt.plot(stem_height.keys(), f_ligu_interpolated([h for h in stem_height.keys()]))
        df['stem_height'] = df.apply(lambda row: stem_height[row.timestamp], axis=1)
    else:
        df['stem_height'] = 0

    # best camera angle for each leaf:

    df['camera_angle'] = 0
    for index, row in df.iterrows():
        snapshot = plant.snapshots[row['s']]
        leaf = snapshot.leaves[row['l']]
        df.loc[index, 'camera_angle'] = best_leaf_angles(leaf, snapshot.metainfo['shooting_frame'], n=1)[0]

    # load images (1 per leaf !):

    df['image_name'] = ''
    for timestamp in df['timestamp'].unique():
        dft = df[df['timestamp'] == timestamp]
        snapshot = plant.snapshots[dft['s'].iloc[0]]

        # load image(t) at a given angle. can be used several times after.
        for angle in dft['camera_angle'].unique():
            image, name = get_rgb(metainfo=snapshot.metainfo, angle=angle, save=False, main_folder='data/rgb_leaf_annotation')
            dfta = dft[dft['camera_angle'] == angle]

            # save this image 1 time / leaf (after somme modifications)
            for _, row in dfta.iterrows():
                # image name
                name_row = name[:-4] + '_r{}.png'.format(row['rank'])
                df.loc[(df['camera_angle'] == angle) & (df['timestamp'] == timestamp), 'image_name'] = name_row
                # display line to show stem height
                h = 2448 - row['stem_height']
                image_row = cv2.line(np.float32(image), (0, h), (2048-1, h), (0,0,0), 1, lineType=cv2.LINE_4)
                # display leaf to annotate
                leaf = snapshot.leaves[row['l']]
                pl = leaf.real_longest_polyline()
                pl = phm3d_to_px2d(pl, snapshot.metainfo['shooting_frame'], row['camera_angle'])
                points = [pl[i] for i in np.linspace(0, len(pl) - 1, 10).astype(int)]
                for x, y in points:
                    image_row = cv2.circle(np.float32(image_row), (int(x), int(y)), 1, (255, 0, 0), -1)

                io.imsave(name_row, image_row.astype(np.uint8))






if False:

    annotation_dict = dict()
    for snapshot in [plant.snapshots[i] for i in [5, 15, 25, 35]]:

        print(snapshot.metainfo['daydate'])

        # n best angles for each leaf
        n = 1
        angle_dict = dict()
        for leaf, rank in zip(snapshot.leaves, snapshot.rank_annotation):

            if rank != -1:

                a = best_leaf_angles(leaf, snapshot.metainfo['shooting_frame'], n=1)
                angle_dict[k] = a

                print(a[0])
                _, image_path = get_rgb(snapshot.metainfo, a[0], main_folder='data/rgb_leaf_annotation')

                #df.loc[df.shape[0]] = [snapshot.metainfo['task'], rank, a[0]]

                image_name = image_path[image_path.find('plantid'):]
                image_size = os.stat(image_path).st_size
                image_key = image_name + str(image_size)

                if image_key not in annotation_dict.keys():
                    annotation_dict[image_key] = {'filename':image_name,
                                                  'size':image_size,
                                                  'regions':[],
                                                  'file_attributes:':{}}

                pl3d = simplify(leaf.real_pl, max(int(leaf.length / 70), 9))
                pl = phm3d_to_px2d(pl3d, snapshot.metainfo.shooting_frame, a[0])

                x = [int(x) for x in pl[:, 0]]
                y = [int(x) for x in pl[:, 1]]
                region = {'shape_attributes':{'name':'polyline', 'all_points_x':x, 'all_points_y':y},
                          'region_attributes':{}}
                annotation_dict[image_key]['regions'].append(region)





        # keep only 1 best angle for each leaf, and try to use the same angles for different leaves
        #while not all([len(angles) == 1 for _, angles in angle_dict.items()]):
        #    occurences = Counter([a for _, angles in angle_dict.items() for a in angles if len(angles) > 1])
        #    a = max(occurences, key=occurences.get)
        #    for r in angle_dict.keys():
        #        if a in angle_dict[r]:
        #            angle_dict[r] = [a]
        #
        #for r, a in angle_dict.items():
        #    df.loc[df.shape[0]] = [vmsi.metainfo['task'], k, a[0]]


    # read json
    #with open('json_test2.json') as f:
    #    d = json.load(f)

    # save json
    with open('paths1424v2.json', 'w', encoding='utf-8') as f:
       json.dump(vmsi_paths2, f, ensure_ascii=False, indent=4)






























l_anot = pd.read_csv('data/ARCH2017-03-30_LMA.csv', sep=';')

plantids = list(l_anot['plantid'].unique())
np.random.seed(1)
np.random.shuffle(plantids)

cont0 = False
if cont0:

    selected_plantids = []
    rank_dict = dict()
    for plantid in plantids:

        angle = 60

        df = l_anot[l_anot['plantid'] == plantid]
        rank = int(df[df['observationcode'] == 'Havested_leaf']['observation'])
        rank_dict[plantid] = {'rank_anot':rank, 'rank_phm':0}
        length =  float(list(df[df['observationcode'] == 'Havested_leaf_length_cm']['observation'])[0].replace(',', '.'))

        metainfos = get_metainfos_ZA17(plantid, dates=None)
        timestamps = [m.timestamp for m in metainfos]
        order = sorted(range(len(timestamps)), key=lambda k: timestamps[k])
        metainfos = [metainfos[i] for i in order]

        #compute_vmsi(plantid, voxels_size=4, phm_parameters=(1, 'notop', 4, 100),
        #             select_dates=[metainfos[-1].daydate], metainfos=None, f_stem=None, replace=False)

        paths = metainfos_to_paths([metainfos[-1]], stem_smoothing=False, phm_parameters=(4, 1, 'notop', 4, 100))
        vmsi_list = load_plant(metainfos, paths)

        if len(vmsi_list) != 0:

            selected_plantids.append(plantid)
            print('valid', plantid, rank)

            selected_metainfos = [metainfos[i] for i in np.linspace(3, len(metainfos) - 1, 6).astype(int)]
            for m in selected_metainfos:
                img, path = get_rgb(m, angle=angle, main_folder='data/rgb_manual_annotation')

            # bricolage
            vmsi = vmsi_list[0]
            snapshot = TrackedSnapshot(vmsi, metainfos[-1])
            snapshot.image[angle] = img
            img_pl, _ = rgb_and_polylines(snapshot, angle, ranks=[k for k in range(vmsi.get_number_of_leaf())])
            io.imsave(path[:-4] + '_pl.png', img_pl)





cont = False
if cont:

    phm_leaf_info = dict()

    with open('data/rgb_manual_annotation/rank.json') as f:
        rank_dict = json.load(f)

    for plantid, rank in rank_dict.items():

        print(plantid)

        phm_leaf_info[plantid] = dict()

        phm_rank = rank['rank_phm']
        anot_rank = rank['rank_anot']

        metainfos = get_metainfos_ZA17(plantid, dates=None)
        timestamps = [m.timestamp for m in metainfos]
        order = sorted(range(len(timestamps)), key=lambda k: timestamps[k])
        metainfos = [metainfos[i] for i in order]
        paths = metainfos_to_paths([metainfos[-1]], stem_smoothing=False, phm_parameters=(4, 1, 'notop', 4, 100))
        vmsi = load_plant(metainfos, paths)[0]

        leaf =  vmsi.get_leaf_order(phm_rank)
        a = best_leaf_angles(leaf, metainfos[-1]['shooting_frame'], n=1)[0]

        img, path = get_rgb(metainfos[-1], angle=a, main_folder='data/rgb_manual_annotation/final', plant_folder=False)

        pl = leaf.real_longest_polyline()
        pl = phm3d_to_px2d(pl, metainfos[-1].shooting_frame, a)
        points = [pl[i] for i in np.linspace(0, len(pl)-1, 10).astype(int)]
        for x, y in points:
            img = cv2.circle(np.float32(img), (int(x), int(y)), 3, (255, 0, 0), -1)
        io.imsave(path, img)

        phm_leaf_info[plantid]['px_mm_ratio'] = shooting_frame_conversion(metainfos[-1].shooting_frame)[0]
        phm_leaf_info[plantid]['azimuth'] = leaf.info['pm_azimuth_angle']
        phm_leaf_info[plantid]['cam_angle'] = a







cont2 = False
if cont2:

    # Llorenc annotation
    l_anot = pd.read_csv('data/ARCH2017-03-30_LMA.csv', sep=';')

    # Image annotation
    with open('data/rgb_manual_annotation/vgg_annotation_v2.json') as f:
        d = json.load(f)

    # Other info for each leaf : px/mm conversion, ...
    with open('data/rgb_manual_annotation/phm_leaf_info.json') as f:
        phm_leaf_info = json.load(f)

    plantids = []
    anot_lengths = []
    for key in d.keys():

        name = d[key]['filename']
        plantid = int(name[name.find('plantid') + len('plantid'):name.rfind('_d2017')])

        if d[key]['regions'] != []:

            shape = d[key]['regions'][0]['shape_attributes']
            x, y = shape['all_points_x'], shape['all_points_y']
            pl = np.array([x, y]).T

            length = np.sum([np.linalg.norm(np.array(pl[k]) - np.array(pl[k + 1])) for k in range(len(pl) - 1)])

            length *= phm_leaf_info[str(plantid)]['px_mm_ratio']

            # angle correction
            correct_angle = False
            if correct_angle:
                info = phm_leaf_info[str(plantid)]
                a1, a2 = info['azimuth'], info['cam_angle']
                da = abs(((a1 / 180 - a2 / 180) + 1) % 2 - 1)
                da = min(da, 1 - da) * 180  # 0 - > 90 deg
                cos_ = np.cos(da / 90 * np.pi / 2)

                length /= cos_
                print(plantid, round(da, 3), round(cos_, 3), round(length, 2))


            plantids.append(plantid)
            anot_lengths.append(length / 10)

    real_lengths = []
    for plantid, anot_length in zip(plantids, anot_lengths):

        row = l_anot[(l_anot['plantid'] == plantid) & (l_anot['observationcode'] == 'Havested_leaf_length_cm')]
        real_length = float(list(row['observation'])[0].replace(',', '.'))
        real_lengths.append(real_length)

        print(plantid, round(real_length - anot_length, 2))


    x, y = real_lengths, anot_lengths
    plt.figure()
    plt.title('ZA17 : annotation terrain vs annotation image')
    plt.plot(x, y, 'r*')
    plt.xlabel('longueur (annotation terrain)')
    plt.ylabel('longueur (annotation image)')
    a, b = min([min(x), min(y)]), max([max(x), max(y)])
    #plt.plot([a, b], [a, b])
    plt.plot([90, 130], [90, 130], 'k')

    biais = np.mean(np.array(x) - np.array(y))
    mae = np.mean(np.abs(np.array(x) - np.array(y)))
    print('MAE = {}cm, biais = {}cm'.format(round(mae, 2), round(biais, 2)))








#### stem annotation

cont3 = False
if cont3:

    plantid = 940

    metainfos = get_metainfos_ZA17(plantid, dates=None)

    with open('data/stem_annotation/stem_{}.json'.format(plantid)) as f:
        d = json.load(f)

    stem_height = extract_json_stem(d)
    plt.plot(stem_height.keys(), stem_height.values())






