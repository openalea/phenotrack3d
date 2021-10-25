"""
An example script to extract the following phenotypic traits after Phenomenal + maizetrack :
- length profile
- insertion height profile
- azimuth profile
- leaf growth over time
- stem growth over time
- visible leaf stage
- ligulated leaf stage
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ===== load data ====================================================================================================

data_folder = 'maizetrack/examples/data/trait_extraction/'

# dataframe resulting from the leaf tracking on a time-series of Phenomenal objects (see tutorial_tracking.py)
# It countains data about each leaf of each Phenomenal object (vmsi) in the time-series
df = pd.read_csv(data_folder + 'tracking.csv')

# dataframe resulting from the collars detection + stem height smoothing (see tutorials in maizetrack package)
df_stem = pd.read_csv(data_folder + 'stem.csv')

# when the panicle appears for this plant
timestamp_panicle = 1496095200

# ===== length profile  ===============================================================================================

df_mature = df[df['mature']]
ranks = sorted([r for r in df_mature['rank_tracking'].unique() if r != 0])
length_profile = []
for r in ranks:
    dfr = df_mature[df_mature['rank_tracking'] == r]
    dfr = dfr.sort_values('timestamp')[:15]  # remove old time points (risk of senescence)
    length_profile.append([r, np.median(dfr['l_extended'])])  # leaf length, after extension
length_profile = pd.DataFrame(length_profile, columns=['rank', 'length'])

plt.figure()
plt.xlabel('Rank')
plt.ylabel('Leaf length')
plt.title('Length profile')
plt.plot(length_profile['rank'], length_profile['length'], 'k*-')

# ===== length profile  ===============================================================================================

df_mature = df[df['mature']]
ranks = sorted([r for r in df_mature['rank_tracking'].unique() if r != 0])
height_profile = []
for r in ranks:
    dfr = df_mature[df_mature['rank_tracking'] == r]
    dfr = dfr.sort_values('timestamp')[:15]  # remove old time points (risk of senescence)
    height_profile.append([r, np.median(dfr['h'])])  # leaf length, after extension
height_profile = pd.DataFrame(height_profile, columns=['rank', 'height'])

# TODO : fix calibrations issues
height_profile['height'] += 720

plt.figure()
plt.xlabel('Rank')
plt.ylabel('Leaf insertion height')
plt.title('Insertion height profile')
plt.plot(height_profile['rank'], height_profile['height'], 'k*-')

# ===== stem growth ===================================================================================================

plt.figure()
plt.xlabel('Time')
plt.ylabel('Stem height')
plt.title('Stem growth over time (smoothed)')
plt.plot(df_stem['t'], df_stem['z_phenomenal_smooth'], 'k-')

# ===== individual leaves growth ======================================================================================

max_rank = 8

plt.figure()
df_growing = df[~df['mature']]
ranks = sorted([r for r in df_growing['rank_tracking'].unique() if r != 0 and r <= max_rank])
for r in ranks:
    dfr = df_growing[df_growing['rank_tracking'] == r]
    plt.plot(dfr['timestamp'], dfr['l_extended'], '*-', label='leaf {}'.format(r))

plt.legend()
plt.xlabel('Time')
plt.ylabel('Leaf length')
plt.title('Individual leaves growth over time')

# ===== visible leaf stage ===========================================================================================

# stop at panicle emergence (Risk that panicle can be segmented as leaves in Phenomenal)
df_nopanicle = df[df['timestamp'] < timestamp_panicle]

# ===== a) first estimation of visible leaf stage over time
df_visi = []
for t in df_nopanicle['timestamp'].unique():

    dft = df_nopanicle[df_nopanicle['timestamp'] == t]
    mature_ranks = dft[dft['mature']]['rank_tracking']
    # n_mature = max rank among mature leaves (0 if no mature leaves)
    if mature_ranks.empty:
        n_mature = 0
    else:
        n_mature = max(mature_ranks)
    # n_growing = number of growing leaves
    n_growing = len(dft[~dft['mature']])
    df_visi.append([t, n_mature + n_growing])
df_visi = pd.DataFrame(df_visi, columns=['t', 'n'])

# ===== b) smoothing

median_visi = df_visi.groupby('n').median('t').reset_index()

# add missing n values if needed (interpolation)
for n in range(min(median_visi['n']) + 1, max(median_visi['n'])):
    if n not in list(median_visi['n']):
        t1 = median_visi[median_visi['n'] < n].sort_values('n').iloc[-1]['t']
        t2 = median_visi[median_visi['n'] > n].sort_values('n').iloc[0]['t']
        median_visi = median_visi.append({'n': n, 't': (t1 + t2) / 2}, ignore_index=True)

# compute emergence timing for each leaf (except for first and last n)
emerg_visi = []
for n in sorted(list(median_visi['n'].unique()))[1:-1]:
    t = np.mean(median_visi[median_visi['n'].isin([n - 1, n])]['t'])
    emerg_visi.append([t, n])
emerg_visi = pd.DataFrame(emerg_visi, columns=['t', 'n'])

# if needed, force monotony : t = f(n) can only increase
emerg_visi = emerg_visi.sort_values('n')
for i_row, rowe in emerg_visi.iterrows():
    previous = emerg_visi[emerg_visi['n'] < rowe['n']]
    if not previous.empty:
        emerg_visi.iloc[i_row, emerg_visi.columns.get_loc('t')] = max(rowe['t'], max(previous['t']))

plt.figure()
plt.plot(df_visi['t'], df_visi['n'], 'k*', label='brut visible leaf stage')
plt.plot(emerg_visi['t'], emerg_visi['n'], 'r.-', label='smoothed visible leaf stage')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Number of leaves')
plt.title('Visible leaf stage')

# ===== ligulated leaf stage ==========================================================================================

# this section uses the dataframes height_profile and df_stem from the previous sections.

T = np.linspace(min(df_stem['t']), max(df_stem['t']), 3000)
f = interp1d(df_stem['t'], df_stem['z_phenomenal_smooth'])
Z = np.array([f(ti) for ti in T])

# determine the ligulation timing of each leaf
emerg_ligu = []
ranks = sorted(height_profile['rank'])
for r in ranks[1:]:
    t = T[np.argmin(np.abs(Z - height_profile[height_profile['rank'] == r - 1].iloc[0]['height']))]
    emerg_ligu.append([t, r])
emerg_ligu = pd.DataFrame(emerg_ligu, columns=['t', 'n'])

plt.figure()
plt.xlabel('Time')
plt.ylabel('Number of leaves')
plt.title('Ligulated leaf stage')
plt.plot(emerg_ligu['t'], emerg_ligu['n'], 'r-', label='Linear interpolation')
plt.step(emerg_ligu['t'], emerg_ligu['n'], 'b-', where='post', label='Step interpolation')
plt.plot(emerg_ligu['t'], emerg_ligu['n'], 'k*', label='Ligulation timing')
plt.legend()



