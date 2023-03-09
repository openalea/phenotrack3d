import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openalea.maizetrack.local_cache import get_metainfos_ZA17
from openalea.maizetrack.stem_correction import stem_height_smoothing

from openalea.maizetrack.local_cache import metainfos_to_paths, check_existence, load_plant
from openalea.maizetrack.stem_correction import get_median_polyline
from openalea.maizetrack.utils import phm3d_to_px2d

import json

from sklearn.metrics import r2_score
from scipy.interpolate import interp1d

# ================================================================

# modulor
df_tt = pd.read_csv('TT_ZA17.csv')

# modulor
folder = 'local_cache/cache_ZA17/collars_voxel4_tol1_notop_vis4_minpix100'
all_files = [folder + '/' + rep + '/' + f for rep in os.listdir(folder) for f in os.listdir(folder + '/' + rep)]

#plantids = np.unique([int(f.split('/')[-1][:4]) for f in all_files])
plantids = [1152,  475,  313, 1303, 1292,  803, 1430, 1424,  309,  958]

dfs = []
plantid_files = [f for f in all_files if int(f.split('/')[-1][:4]) == plantid]
for f in plantid_files:
    task = int(f.split('_')[-1].split('.csv')[0])
    dft = pd.read_csv(f)
    daydate = f.split('/')[3]
    tt = df_tt[df_tt['daydate'] == daydate].iloc[0]['ThermalTime']
    dft['t'] = tt
    dfs.append(dft)
df = pd.concat(dfs)

df = df[df['score'] > 0.95]
df['z_phenomenal'] /= 10
#df['y'] = 2448 - df['y']
df_stem = df.sort_values('z_phenomenal', ascending=False).drop_duplicates(['t']).sort_values('t')
f_smoothing = stem_height_smoothing(np.array(df_stem['t']), np.array(df_stem['z_phenomenal']))

fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
ax.tick_params(axis='both', which='major', labelsize=20)  # axis number size
plt.title(plantid)
plt.xlabel('Thermal time $(day_{20°C})$', fontsize=30)
plt.ylabel('Stem height (cm)', fontsize=30)

plt.plot(df_stem['t'], 2448 - df_stem['y'], 'k.', markersize=15)
plt.legend(prop={'size': 20}, loc='lower right', markerscale=2)






