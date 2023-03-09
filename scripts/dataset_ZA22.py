import pandas as pd
import numpy as np
import shutil
import os

# =================================================================================================



# ===============================================================================================

df_db = pd.read_csv('data/copy_from_database/images_ZA22.csv')

img_files = [f for f in os.listdir('data/rgb_insertion_annotation/dataset_2022/images/') if 'ZA22' in f]
sk_files = [f for f in os.listdir('data/rgb_insertion_annotation/dataset_2022/skeletons/') if 'ZA22' in f]

img_files_pb = []
for img_file in img_files:
    plantid, task = img_file.split('.png')[0].split('_')[1:-1]
    n = '{}_{}'.format(plantid, task)
    if sum([n in f for f in sk_files]) == 0:
        img_files_pb.append(img_file)

import shutil
for img_file in img_files_pb:
    plantid, task, angle = np.array(img_file.split('.png')[0].split('_')[1:]).astype(int)
    selec = df_db[(df_db['taskid'] == task) & (df_db['plantid'] == plantid)]
    selec = selec[selec['imgangle'] == angle]
    for _, row in selec.iterrows():
        fd = 'data/rgb_insertion_annotation/dataset_2022/images/'
        f = fd + 'ZA22_{}_{}_{}.png'.format(str(int(row['plantid'])).zfill(4), row['taskid'], row['imgangle'])
        if os.path.isfile(f):
            os.rename(f, fd + '0_ZA22_{}_{}_{}.png'.format(str(int(row['plantid'])).zfill(4), row['taskid'], row['imgangle']))
        # path1 = 'X:/ARCH2022-01-10/{}/{}.png'.format(task, row['imgguid'])
        # path2 = 'data/ZA22_visualisation/{}_{}.png'.format(img_file[:-4], row['imgangle'])
        # shutil.copyfile(path1, path2)

# ===== import some csv extracted with the R code from Llorenc ==================================

df = pd.read_csv('data/copy_from_database/images_ZA22.csv')
df['daydate'] = [d[5:10] for d in df['acquisitiondate']]
df = df[(df['viewtypeid'] == 2) & df['imgangle'].isin([k*30 for k in range(12)])]  # 12 side image
# x = df.groupby(['plantid', 'taskid']).size().reset_index()[0]

df2 = pd.read_csv('data/copy_from_database/plants_ZA22.csv')
df2 = df2[df2['plantcode'].str.contains('ARCH2022-01-10')]  # remove strange plant names
df2['genotype'] = [p.split('/')[1] for p in df2['plantcode']]
df2['scenario'] = [p.split('/')[5] for p in df2['plantcode']]

# ===== deepcollar training dataset ==============================================================

# plant set for deepcollar training

genotypes = np.random.choice(np.unique(df2['genotype']), 250, replace=False)

plants = []
for g in genotypes:
    for s in ['WW', 'WD']:
        s = df2[(df2['genotype'] == g) & (df2['scenario'] == s)]
        plants.append(np.random.choice(s['plantcode']))

# load the corresponding images

for plant in plants:
    plantid = int(plant.split('/')[0])
    if plantid > 425: # re-add old dates
        s = df[df['plantid'] == plantid]
        if np.random.random() < 0.1:
            s = s[s['daydate'] == max(s['daydate'])]
        else:
            s = s[s['daydate'] != max(s['daydate'])]
        row = s.sample().iloc[0]
        name = 'ZA22_{}_{}_{}.png'.format(str(int(row['plantid'])).zfill(4), row['taskid'], row['imgangle'])
        path1 = 'Y:/ARCH2022-01-10/{}/{}.png'.format(row['taskid'], row['imgguid'])
        path2 = 'data/rgb_insertion_annotation/dataset_2022/images/' + name
        #img = io.imread(path1)[:, :, :3]
        shutil.copyfile(path1, path2)

# plant set for tracking validation

genotypes2 = np.random.choice([g for g in np.unique(df2['genotype']) if g not in genotypes], 100, replace=False)

plants2 = []
for g in genotypes2:
    s = df2[df2['genotype'] == g]
    plants2.append(np.random.choice(s['plantcode']))

# ===== visu plant for a given plantid ======================================================================

import pandas as pd
import shutil
df = pd.read_csv('data/copy_from_database/images_ZA22.csv')

plantid = 1846
angle = 60

s = df[(df['plantid'] == plantid) & (df['imgangle'] == angle)]

for _, row in s.iterrows():
    name = 'ZA22_{}_{}_{}.png'.format(str(int(row['plantid'])).zfill(4), row['taskid'], row['imgangle'])
    path1 = 'X:/ARCH2022-01-10/{}/{}.png'.format(row['taskid'], row['imgguid'])
    path2 = 'data/ZA22_visualisation/' + name
    shutil.copyfile(path1, path2)

# ===== other data visu... ==================================================================================

for g in s['genotype'].unique():
    s2 = s[s['genotype'] == g]
    for plantid in s2['plantid']:
        s3 = df[df['plantid'] == plantid]

        # tasks with 12 side views
        gb = s3.groupby('taskid').agg('count').reset_index()
        s3 = s3[s3['taskid'].isin(gb[gb['studyid'] == 12]['taskid'])]
        # angle = 60
        s3 = s3[s3['imgangle'] == 60]

        print(g, len(s3[s3['imgangle'] == 60]), max(s3['daydate']))

        print(np.array(sorted(s3['daydate'])))

for d in sorted(df['daydate'].unique()):
    s = df[df['daydate'] == d]
    print(d, len(s['plantid'].unique()))


import os
p = 'X:/lepseBinaries/ARCH2022-01-10'
tasks = [t for t in os.listdir(p) if '.' not in t]
task_to_subdir = {int(t): next(s for s in subdirs if s in os.listdir(p + '/' + t)) for t in tasks}













