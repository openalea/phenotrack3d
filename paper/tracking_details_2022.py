import os
import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib import colors
import cv2
import pandas as pd

from openalea.maizetrack.local_cache import get_metainfos_ZA17, metainfos_to_paths, check_existence, load_plant
from openalea.maizetrack.trackedPlant import TrackedPlant
from openalea.maizetrack.phenomenal_display import PALETTE, plot_snapshot
from openalea.maizetrack.utils import phm3d_to_px2d

from openalea.maizetrack.phenomenal_display import plot_vmsi, plot_leaves


folder = 'data/tracking/'
plantids = [int(f.split('.')[0]) for f in os.listdir(folder) if os.path.isfile(folder + f)]

df_list = []
for plantid in plantids:
    dfid = pd.read_csv(folder + '{}.csv'.format(plantid))
    df_list.append(dfid)
df = pd.concat(df_list)
df = df[~df['rank_annotation'].isin([-1, 0])]
# df = df[df['timestamp'] < 1496509351]
# df = df[df['timestamp'] < 1495922400] # 05-28

# ====== mature tracking, but with different end dates ============================================================*

# selec = df
n, accuracy = [], []
for plantid in df['plantid'].unique():

    print(plantid)

    metainfos = get_metainfos_ZA17(plantid)

    paths = metainfos_to_paths(metainfos, folder='local_cache/cache_ZA17/segmentation_voxel4_tol1_notop_vis4_minpix100_stem_smooth_tracking')
    metainfos, paths = check_existence(metainfos, paths)
    vmsi_list = load_plant(metainfos, paths)

    plant_ref = TrackedPlant.load_and_check(vmsi_list)
    plant_ref.load_rank_annotation()

    for dmax in ['2017-06-10', '2017-06-03', '2017-05-27', '2017-05-20']:

        plant = copy.deepcopy(plant_ref)

        plant.snapshots = [s for s in plant.snapshots if s.metainfo.daydate < dmax]

        plant.align_mature(direction=1, gap=12.365, w_h=0.03, w_l=0.002, gap_extremity_factor=0.2, n_previous=500,
                           rank_attribution=True)
        snapshots = plant.snapshots

        df_res = plant.get_dataframe(load_anot=False)
        df_res = df_res[df_res['mature']]
        print(dmax, round(len(df_res[df_res['rank_tracking'] == df_res['rank_annotation']]) / len(df_res), 2))
        df_res.to_csv('data/tracking/tests_2022/{}_{}.csv'.format(plantid, dmax))

# ===== growing tracking, but with/without ground-truth mature tracking ==============================================

# selec = df
n, accuracy = [], []
for plantid in df['plantid'].unique():

    print(plantid)

    metainfos = get_metainfos_ZA17(plantid)

    paths = metainfos_to_paths(metainfos, folder='local_cache/cache_ZA17/segmentation_voxel4_tol1_notop_vis4_minpix100_stem_smooth_tracking')
    metainfos, paths = check_existence(metainfos, paths)
    vmsi_list = load_plant(metainfos, paths)

    plant_ref = TrackedPlant.load_and_check(vmsi_list)
    plant_ref.load_rank_annotation()
    plant_ref.snapshots = [s for s in plant_ref.snapshots if s.metainfo.daydate < '2017-06-03']

    for method in ['normal', 'ground-truth']:

        plant = copy.deepcopy(plant_ref)

        # a) normal tracking
        if method == 'normal':
            plant.align_mature(direction=1, gap=12.365, w_h=0.03, w_l=0.002, gap_extremity_factor=0.2, n_previous=500,
                               rank_attribution=True)
            plant.align_growing()

        # b) tracking starting from ground-truth
        elif method == 'ground-truth':
            for s in plant.snapshots:
                rmax = 99
                s.order = [-1 if i not in s.rank_annotation else s.rank_annotation.index(i) for i in range(rmax + 1)]
                s.order = [i if (i != -1 and s.leaves[i].info['pm_label'] == 'mature_leaf') else -1 for i in s.order]
            rmax2 = [all([s.order[i] == -1 for s in plant.snapshots]) for i in range(rmax)].index(True)
            for s in plant.snapshots:
                s.order = s.order[:rmax2]
            plant.align_growing()

        df_res = plant.get_dataframe(load_anot=False)
        df_res = df_res[~df_res['mature']]
        print(method, round(len(df_res[df_res['rank_tracking'] == df_res['rank_annotation']]) / len(df_res), 2))
        df_res.to_csv('data/tracking/tests_2022/growing_{}_{}.csv'.format(plantid, method))

# ===============================================================================================================
# ===============================================================================================================
# ===============================================================================================================
# ===============================================================================================================
# ===============================================================================================================
# ===============================================================================================================

df_res = []
for plantid in df['plantid'].unique():
    for dmax in ['2017-06-10', '2017-06-03', '2017-05-27', '2017-05-20']:
        dfi = pd.read_csv('data/tracking/tests_2022/{}_{}.csv'.format(plantid, dmax))
        dfi['dmax'] = dmax
        dfi['method'] = None
        dfi['mature'] = True
        df_res.append(dfi)
    for method in ['normal', 'ground-truth']:
        dfi = pd.read_csv('data/tracking/tests_2022/growing_{}_{}.csv'.format(plantid, method))
        dfi['dmax'] = None
        dfi['method'] = method
        dfi['mature'] = False
        df_res.append(dfi)
df_res = pd.concat(df_res)
df_res = df_res[~df_res['rank_annotation'].isin([-1, 0])]
df_res['i'] = np.arange(len(df_res))

# remove growing leaves that didn't reach mature
for plantid in df_res['plantid'].unique():
    # mature ground-truth (max rank)
    selec_m = df_res[(df_res['dmax'] == '2017-06-03') & (df_res['plantid'] == plantid)]
    # growing ground-truth above
    selec_g = df_res[(df_res['plantid'] == plantid) &
                     (df_res['rank_annotation'] > max(selec_m['rank_annotation']))]
    # remove above
    df_res = df_res[~df_res['i'].isin(list(selec_g['i']))]

ranks = [r for r in sorted(df_res['rank_annotation'].unique())]
res = []

# ===== A) MATURE LEAVES =======

dmax = '2017-06-03'  # pipeline simulation end-date

selec = df_res[(df_res['dmax'] == dmax)]
for rank in ranks:
    selecrank = selec[selec['rank_annotation'] == rank]
    if not selecrank.empty:
        acc_list = []
        n_plants = 0
        for plantid in selecrank['plantid'].unique():
            s = selecrank[selecrank['plantid'] == plantid]
            if not s.empty:
                n_plants += 1
                accuracy = len(s[s['rank_annotation'] == s['rank_tracking']]) / len(s) * 100
                acc_list.append(accuracy)

        # compute bootstrap confidence interval
        n_bootstrap = 5000
        conf = 0.95
        bootstrap = sorted([np.mean(np.random.choice(acc_list, len(acc_list))) for k in range(n_bootstrap)])
        err1, err2 = bootstrap[int(n_bootstrap * ((1 - conf)/2))], bootstrap[int(n_bootstrap * (conf + (1 - conf)/2))]

        res.append([True, 'mature', rank, np.mean(acc_list), err1, err2, len(selecrank), n_plants])


# ===== a) remove leaves too old
to_keep = []
for plantid in selec['plantid'].unique():
    s = selec[selec['plantid'] == plantid]
    for r in s['rank_annotation'].unique():
        s2 = s[s['rank_annotation'] == r]
        to_keep = to_keep + list(s2.sort_values('timestamp').iloc[:10]['i'])
selec = selec[selec['i'].isin(to_keep)]

# ===== b) remove timestamps with ear
s = selec[selec['plantid'] == 461]
s1 = s[s['timestamp'].isin(sorted(s['timestamp'].unique())[-2:])]
s = selec[selec['plantid'] == 1391]
s2 = s[s['timestamp'].isin(sorted(s['timestamp'].unique())[-2:])]
s = selec[selec['plantid'] == 905]
s3 = s[s['timestamp'].isin(sorted(s['timestamp'].unique())[-1:])]
s = selec[selec['plantid'] == 832]
s4 = s[s['timestamp'].isin(sorted(s['timestamp'].unique())[-1:])]
selec = selec[~selec['i'].isin(list(s1['i']) + list(s2['i']) + list(s3['i']) + list(s4['i']))]

# ===== c) remove timestamps with last ligul
# to_keep = []
# for plantid in selec['plantid'].unique():
#     s = selec[selec['plantid'] == plantid]
#     s2 = s[s['rank_annotation'] == max(s['rank_annotation'])]
#     to_keep += list(s[s['timestamp'] < min(s2['timestamp'])]['i'])
# selec = selec[selec['i'].isin(to_keep)]

for rank in ranks:
    selecrank = selec[selec['rank_annotation'] == rank]
    if not selecrank.empty:
        acc_list = []
        n_plants = 0
        for plantid in selecrank['plantid'].unique():
            s = selecrank[selecrank['plantid'] == plantid]
            if not s.empty:
                n_plants += 1
                accuracy = len(s[s['rank_annotation'] == s['rank_tracking']]) / len(s) * 100
                acc_list.append(accuracy)

        # compute bootstrap confidence interval
        n_bootstrap = 5000
        conf = 0.95
        bootstrap = sorted([np.mean(np.random.choice(acc_list, len(acc_list))) for k in range(n_bootstrap)])
        err1, err2 = bootstrap[int(n_bootstrap * ((1 - conf)/2))], bootstrap[int(n_bootstrap * (conf + (1 - conf)/2))]

        res.append([True, 'mature2', rank, np.mean(acc_list), err1, err2, len(selecrank), n_plants])

# ===== B) GROWING LEAVES =======

for (criteria, method) in zip(['normal', 'ground-truth'], ['growing', 'growing2']):
    selec = df_res[(df_res['method'] == criteria)]
    for rank in df_res[(df_res['dmax'] == '2017-06-03')]['rank_annotation'].unique():
        selecrank = selec[selec['rank_annotation'] == rank]
        if not selecrank.empty:
            acc_list = []
            n_plants = 0
            for plantid in selecrank['plantid'].unique():
                s = selecrank[selecrank['plantid'] == plantid]
                if not s.empty:
                    n_plants += 1
                    accuracy = len(s[s['rank_annotation'] == s['rank_tracking']]) / len(s) * 100
                    acc_list.append(accuracy)

            # compute bootstrap confidence interval
            n_bootstrap = 5000
            conf = 0.95
            bootstrap = sorted([np.mean(np.random.choice(acc_list, len(acc_list))) for k in range(n_bootstrap)])
            err1, err2 = bootstrap[int(n_bootstrap * ((1 - conf) / 2))], bootstrap[
                int(n_bootstrap * (conf + (1 - conf) / 2))]

            res.append([False, method, rank, np.mean(acc_list), err1, err2, len(selecrank), n_plants])

res = pd.DataFrame(res, columns=['mature', 'option', 'rank', 'accuracy', 'err1', 'err2', 'n', 'n_plants'])

# ===== C) PLOT =====

from matplotlib.ticker import MaxNLocator
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # int axis
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Leaf rank', fontsize=30)
plt.ylabel('Rank assignment accuracy (%)', fontsize=30)
plt.xlim([0.5, 14.5])
plt.ylim([65, 101])
plt.plot([-100, 100], [100, 100], 'k--')

res['pc'] = [row['n'] / sum(res[res['option'] == row['option']]['n']) for _, row in res.iterrows()]

res_selec = res[res['n_plants'] > 15]
#res_selec = res[res['n'] > 40]
# res_selec = res[res['pc'] > 0.0000012]
# res_selec = res_selec[res_selec['rank'] <= 14]

label_dic = {'mature': 'Ligulated leaves',
             'mature2': None,
             'growing': 'Growing leaves',
             'growing2': None}

for option in label_dic.keys():

    s = res_selec[res_selec['option'] == option]

    color = 'blue' if 'mature' in option else 'orange'
    symbol = '^' if 'mature' in option else 'o'
    #dx = 0.02 if s.iloc[0]['mature'] else -0.02

    linewidth = 3 if option in ['mature', 'growing'] else 1.2
    markersize = 10 if option in ['mature', 'growing'] else 5
    linestyle = '-' if option in ['mature', 'growing'] else '--'
    linestyle = ':' if option == 'mature2' else linestyle

    # color, linestyle = ('green', '-.') if option == 'old removed' else (color, linestyle)
    # color, linestyle = ('purple', ':') if option == 'old removed + no ear' else (color, linestyle)
    # color, linestyle = ('black', ':') if option == 'old removed + no ear + no last ligul' else (color, linestyle)
    # color = 'red' if option == 'ground-truth' else color

    plt.plot(s['rank'], s['accuracy'], symbol, color=color,
             label=label_dic[option],
             markersize=markersize, linewidth=linewidth)
    plt.plot(s['rank'], s['accuracy'], linestyle + symbol, color=color,
             markersize=markersize, linewidth=linewidth)

    if option in ['mature', 'growing']:
        plt.fill_between(s['rank'], s['err1'], s['err2'], color=color, alpha=0.15 if color == 'blue' else 0.25)

leg = plt.legend(prop={'size': 30}, loc='lower left')
leg.get_frame().set_linewidth(0.0)
# remove "white non transparent rectangle" around legend
leg.get_frame().set_alpha(None)
leg.get_frame().set_facecolor((0, 0, 0, 0))


# ===== visu some matrixes to understand =========================================================================

# 1) plot errors

res2 = []

#for plantid in df['plantid'].unique()[:1]:
for method in ['normal', 'ground-truth']:
    dfm = pd.read_csv('data/tracking/tests_2022/growing_{}_{}.csv'.format(plantid, method))
    for rank in dfm['rank_annotation'].unique():
        selecrank = dfm[dfm['rank_annotation'] == rank]
        accuracy = len(selecrank[selecrank['rank_annotation'] == selecrank['rank_tracking']]) / len(selecrank) * 100
        res2.append([False, method, rank, accuracy, len(selecrank)])

# plt.figure()
# res2 = pd.DataFrame(res2, columns=['mature', 'option', 'rank', 'accuracy', 'n']).sort_values('rank')
# s1 = res2[res2['option'] == 'normal']
# plt.plot(s1['rank'], s1['accuracy'], 'r-*')
# s2 = res2[res2['option'] == 'ground-truth']
# plt.plot(s2['rank'] + 0.1, s2['accuracy'], 'g-*')

# 2) plot matrix

for plantid in plantids:

    #df1 = pd.read_csv('data/tracking/tests_2022/growing_{}_{}.csv'.format(plantid, method))
    df2 = pd.read_csv('data/tracking/tests_2022/{}_{}.csv'.format(plantid, '2017-06-03'))
    #dfm = pd.concat((df1, df2))
    dfm = df2

    T = len(dfm['timestamp'].unique())
    R = max(dfm['rank_tracking'].unique())
    mat = np.zeros((T, R)) * np.NAN
    for t_index, t in enumerate(dfm['timestamp'].unique()):
        s = dfm[dfm['timestamp'] == t]
        for _, row in s.iterrows():
            mat[t_index, row['rank_tracking'] - 1] = row['rank_annotation'] - 1

    # remove empty columns
    mat = mat[:, ~np.isnan(mat).all(axis=0)]

    fig, ax = plt.subplots()
    fig.canvas.set_window_title(str(plantid))

    #ax.set_xticks(np.arange(R) - 0.5, minor=True)
    # ax.set_yticks(np.arange(T + 1) - 0.5, minor=True)
    # ax.grid(which='minor', color='white', linewidth=12)

    # # start axis at 1
    # plt.xticks(np.arange(R), np.arange(R) + 1)
    # plt.yticks(np.arange(T), np.arange(T) + 1)
    #
    # ax.set_xlabel('Leaf rank', fontsize=30)
    # ax.xaxis.tick_top()
    # ax.xaxis.set_label_position('top')
    # ax.set_ylabel('Time', fontsize=30)
    # plt.locator_params(nbins=4)
    # ax.tick_params(axis='both', which='major', labelsize=25)  # axis number size

    rgb = np.array(PALETTE) / 255.
    rgb = np.concatenate((np.array([[0., 0., 0.]]), rgb))
    cmap = colors.ListedColormap(rgb, "")
    val = [k - 1.5 for k in range(50)]
    norm = colors.BoundaryNorm(val, len(val)-1)
    plt.imshow(mat, interpolation='nearest', cmap=cmap, norm=norm)

    plt.title(plantid)

    # for t_index, t in enumerate(dfm['timestamp'].unique()):
    #     s = dfm[dfm['timestamp'] == t]
    #     for _, row in s.iterrows():
    #         if row['mature']:
    #             plt.plot(row['rank_tracking'], t_index, 'k.')
    #             #mat[t_index, row['rank_tracking']] = row['rank_annotation'] - 1
    #
    #
