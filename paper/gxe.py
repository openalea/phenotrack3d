import pandas as pd
import numpy as np
import os
from skimage import io
import matplotlib.pyplot as plt
from openalea.maizetrack.utils import get_rgb
from openalea.maizetrack.local_cache import get_metainfos_ZA17
from openalea.maizetrack.stem_correction import smoothing_function, stem_height_smoothing
from scipy.interpolate import interp1d
import time
import datetime
from matplotlib.ticker import MaxNLocator


def date_to_timestamp(date):
    return time.mktime(datetime.datetime.strptime(date[:10], "%Y-%m-%d").timetuple())


def mad(x):
    """ median absolute deviation """
    return np.median(np.abs(x - np.median(x)))


# thermal time : available in modulor
df_tt = pd.read_csv('TT_ZA17.csv')

# collar detection data
# available in modulor
folder_dl = 'local_cache/cache_ZA17/collars_voxel4_tol1_notop_vis4_minpix100'
all_files_dl = [folder_dl + '/' + rep + '/' + f for rep in os.listdir(folder_dl) for f in os.listdir(folder_dl + '/' + rep)]

# ZA17 GxE data
# available in modulor
pheno_ZA17 = pd.read_csv('pheno_ZA17.csv')
pheno_ZA17['timestamp'] = pheno_ZA17['resultdate'].map(date_to_timestamp)
gxe = pheno_ZA17[['Pot', 'Variety_ID', 'Scenario', 'Rep', 'Sampling']].drop_duplicates()
gxe = gxe[gxe['Sampling'] == 'Rep']
gxe = gxe[gxe['Variety_ID'] != 'SolNu']
genotypes = ['DZ_PG_34', 'DZ_PG_41', 'DZ_PG_68', 'DZ_PG_69', 'DZ_PG_72']
gxe = gxe[gxe['Variety_ID'].isin(genotypes)]
gxe = gxe[gxe['Pot'] != 1070]  # not in cache


# ============

df = pheno_ZA17[pheno_ZA17['Variety_ID'] == 'DZ_PG_41']
df = df[df['observationcode'].isin(['ligul_f', 'visi_f'])]
thermaltimes = np.array(df_tt['ThermalTime'])
timestamps = np.array(df_tt['timestamp'])
df['tt'] = [thermaltimes[np.argmin(np.abs(timestamps - t))] for t in df['timestamp']]
plt.figure()
for var, symb in zip(['ligul_f', 'visi_f'], ['o', '^']):
    for scenario, color, decalage in zip(['WW', 'WD'], ['blue', 'red'], [0, 0.3]):
        selec = df[(df['Scenario'] == scenario) & (df['observationcode'] == var)].sort_values('tt')
        plt.plot(selec['tt'] + decalage, selec['observation'], symb, color=color, markersize=4)

        f, _ = smoothing_function(np.array(selec['tt']), np.array(selec['observation']), dy=None, dw=4, polyorder=2, repet=2)
        plt.plot(selec['tt'], [f(t) for t in np.array(selec['tt'])], '-', color=color)



colors = {'WW': 'dodgerblue', 'WD': 'orangered', 'error': 'gainsboro'}

# # get rgb
# for g in gxe['Variety_ID'].unique():
#     #for e in gxe['Scenario'].unique():
#     for e in ['WW']:
#         selec = gxe[(gxe['Variety_ID'] == g) & (gxe['Scenario'] == e)]
#         plantid = selec.iloc[0]['Pot']
#         metainfos = get_metainfos_ZA17(plantid)
#         metainfo = next(m for m in metainfos if m.daydate == '2017-05-20')
#         img, _ = get_rgb(metainfo, 60, main_folder='rgb', plant_folder=True, save=False, side=True)
#         io.imsave('paper/ZA17_visu/{}_{}_2017-05-20_angle60.png'.format(g, e), img)

genotype = 'DZ_PG_41'

genotypes = ['DZ_PG_41', 'DZ_PG_69']
genotypes = ['DZ_PG_34', 'DZ_PG_41', 'DZ_PG_68', 'DZ_PG_69', 'DZ_PG_72']
for genotype in genotypes:

    # ===== stem height = f(t) ==================================================================================

    plt.figure()
    res_stem = []
    for scenario in ['WW', 'WD']:

        selec = gxe[(gxe['Variety_ID'] == genotype) & (gxe['Scenario'] == scenario)]
        for _, row in selec.iterrows():

            dfid = []
            plantid_files = [f for f in all_files_dl if int(f.split('/')[-1][:4]) == row['Pot']]
            for f in plantid_files:
                task = int(f.split('_')[-1].split('.csv')[0])
                #timestamp = next(m for m in metainfos if m.task == task).timestamp
                daydate = f.split('__')[-2]
                tt = df_tt[df_tt['daydate'] == daydate].iloc[0]['ThermalTime']
                dft = pd.read_csv(f)
                dft['tt'] = tt
                dfid.append(dft)
            dfid = pd.concat(dfid)

            # remove > 06-08
            dfid = dfid[dfid['tt'] < 90.5]

            #  remove big time gaps
            times = list(dfid.sort_values('tt')['tt'].unique())
            for i in range(1, len(times)):
                if (times[i] - times[i - 1]) > 5 * np.mean(np.diff(times)):
                    dfid = dfid[dfid['tt'] < times[i]]

            # data preprocessing, stem extraction
            dfid = dfid[dfid['score'] > 0.95]
            df_stem = dfid.sort_values('z_phenomenal', ascending=False).drop_duplicates(['tt']).sort_values('tt')
            f_smoothing = stem_height_smoothing(np.array(df_stem['tt']), np.array(df_stem['z_phenomenal']))
            df_stem['z_smooth'] = [f_smoothing(t) for t in df_stem['tt']]

            for _, row_stem in df_stem.iterrows():
                res_stem.append([row['Pot'], scenario, row_stem['z_smooth'], row_stem['tt']])

            plt.plot(df_stem['tt'], df_stem['z_phenomenal'], '-', color=colors[scenario])
            #plt.plot(df_stem['tt'], df_stem['z_smooth'], '-', color=colors[scenario])

    res_stem = pd.DataFrame(res_stem, columns=['plantid', 'scenario', 'z', 't'])

    res_gxe = res_stem.copy()
    res_gxe['z'] /= 10
    res_gxe = res_gxe[res_gxe['t'] < np.min(res_gxe.groupby(['scenario']).max('t')['t'])] # same tmax for each scenario
    res_gxe = res_gxe.groupby(['scenario', 't']).agg(['median', ('mad', mad), 'count']).reset_index()
    res_gxe = res_gxe[res_gxe['z']['count'] > 2] # at least 3 rep
    res_gxe = res_gxe[res_gxe['t'] <= min(res_gxe.groupby('scenario').max('t')['t'])]  # same end for each scenario

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.canvas.set_window_title(genotype)
    ax.tick_params(axis='both', which='major', labelsize=20)  # axis number size
    for scenario in ['WW', 'WD']:
        selec = res_gxe[res_gxe['scenario'] == scenario]
        plt.fill_between(selec['t'], selec['z']['median'] - selec['z']['mad']/1, selec['z']['median'] + selec['z']['mad']/1,
                         color=colors['error'])
        plt.plot(selec['t'], selec['z']['median'], '.-', color=colors[scenario], markersize=12)
        plt.plot(selec['t'], selec['z']['median'], '.', color=colors[scenario], label=scenario, markersize=12)
    # lgd = plt.legend(prop={'size': 20}, loc='upper left', markerscale=2, handletextpad=-0.5)
    # lgd.get_frame().set_linewidth(0.0)
    plt.xlabel('Thermal time $(day_{20°C})$', fontsize=30)
    plt.ylabel('Stem height (cm)', fontsize=30)
    plt.xlim([8, 93])
    plt.ylim([-2, 202])

    fig.savefig('paper/stem_{}'.format(genotype), bbox_inches='tight')
    plt.close('all')

    # ===== length profile ==================================================================================

    res_profile_l = []
    res_profile_h = []
    for scenario in ['WW', 'WD']:

        selec = gxe[(gxe['Variety_ID'] == genotype) & (gxe['Scenario'] == scenario)]
        for _, row in selec.iterrows():

            # available in modulor local_benoit
            file = 'data/tracking/gxe/{}.csv'.format(row['Pot'])
            if not os.path.isfile(file):
                print('no file', row['Pot'])
            else:
                dfid = pd.read_csv(file)

                dfid_mature = dfid[dfid['mature']]
                ranks = sorted([r for r in dfid_mature['rank_tracking'].unique() if r != 0])
                profile = []
                for r in ranks:
                    dfr = dfid_mature[dfid_mature['rank_tracking'] == r]
                    dfr = dfr.sort_values('timestamp')[:15]

                    res_profile_l.append([row['Pot'], scenario, np.median(dfr['l_extended']), r])
                    # TODO : height correction
                    res_profile_h.append([row['Pot'], scenario, 720 + np.median(dfr['h']), r])

    res_profile_l = pd.DataFrame(res_profile_l, columns=['plantid', 'scenario', 'l', 'r'])
    res_profile_h = pd.DataFrame(res_profile_h, columns=['plantid', 'scenario', 'h', 'r'])

    res_gxe = res_profile_l.copy()
    res_gxe['l'] /= 10

    plt.figure('all reps {}'.format(genotype))
    for plantid in res_gxe['plantid'].unique():
        selec = res_gxe[res_gxe['plantid'] == plantid]
        plt.plot(selec['r'], selec['l'], '-', color=colors[selec.iloc[0]['scenario']])

    res_gxe = res_gxe.groupby(['scenario', 'r']).agg(['median', ('mad', mad), 'count']).reset_index()
    #res_gxe = res_gxe[res_gxe['l']['count'] > 2] # at least 3 rep
    #res_gxe = res_gxe[res_gxe['r'] <= min(res_gxe.groupby('scenario').max('r')['r'])]  # same end for each scenario

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.canvas.set_window_title(genotype + '_profile')
    ax.tick_params(axis='both', which='major', labelsize=20)  # axis number size
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for scenario in ['WW', 'WD']:
        selec = res_gxe[res_gxe['scenario'] == scenario]
        plt.fill_between(selec['r'], selec['l']['median'] - selec['l']['mad']/1, selec['l']['median'] + selec['l']['mad']/1,
                         color=colors['error'])
        plt.plot(selec['r'], selec['l']['median'], '.-', color=colors[scenario], markersize=12)
        plt.plot(selec['r'], selec['l']['median'], '.', color=colors[scenario], label=scenario, markersize=12)
    # lgd = plt.legend(prop={'size': 20}, loc='upper left', markerscale=2, handletextpad=-0.5)
    # lgd.get_frame().set_linewidth(0.0)
    plt.xlabel('Rank', fontsize=30)
    plt.ylabel('Leaf length (cm)', fontsize=30)
    plt.xlim([0.3, 18.5])
    plt.ylim([-3, 120])

    fig.savefig('paper/profile_{}'.format(genotype), bbox_inches='tight')
    plt.close('all')

    # ===== leaf stage =================================================================================

    plt.figure(genotype)

    res_emergence = []
    for scenario in ['WW', 'WD']:

        selec = gxe[(gxe['Variety_ID'] == genotype) & (gxe['Scenario'] == scenario)]
        for _, row in selec.iterrows():

            # ===== emergence visi ====================================

            # available in modulor local_benoit
            file = 'data/tracking/gxe/{}.csv'.format(row['Pot'])
            if not os.path.isfile(file):
                print('no file', row['Pot'])
            else:
                dfid = pd.read_csv(file)

                # only reps continuing after july
                if len(set(dfid[dfid['timestamp'] > 1495806823]['timestamp'])) < 3:
                    short_rep = True
                else:
                    short_rep = False

                print(row['Pot'], scenario)

                # # stop at panicle emergence
                # df_phenoid = pheno_ZA17[pheno_ZA17['Pot'] == row['Pot']]
                # select = df_phenoid[
                #     (df_phenoid['observationcode'] == 'panicule_visi') & (df_phenoid['observation'] == 1)]
                # t_max = max(dfid['timestamp'])
                # if not select.empty:
                #     t_max = min(select['timestamp'])
                # timestamps = [t for t in dfid['timestamp'].unique() if t < t_max]

                timestamps = [t for t in dfid['timestamp'].unique()]

                # # TODO : useful with new gxe data ?
                # # remove last day
                # timestamps = sorted(timestamps)[:-1]

                df_visi = []
                for t in timestamps:

                    dfidt = dfid[dfid['timestamp'] == t]

                    if short_rep:
                        mature_ranks = dfidt[dfidt['mature']]['rank_tracking']
                        if mature_ranks.empty:
                            n_mature = 0
                        else:
                            n_mature = max(mature_ranks)
                        n_growing = len(dfidt[dfidt['mature'] == False])
                        n_leaves = n_mature + n_growing

                    else:
                        n_leaves = max(dfidt['rank_tracking'])

                    df_visi.append([t, n_leaves])
                df_visi = pd.DataFrame(df_visi, columns=['t', 'n'])

                median_visi = df_visi.groupby('n').median('t').reset_index()

                # add missing n values (interpolation)
                for n in range(min(median_visi['n']) + 1, max(median_visi['n'])):
                    if n not in list(median_visi['n']):
                        t1 = median_visi[median_visi['n'] < n].sort_values('n').iloc[-1]['t']
                        t2 = median_visi[median_visi['n'] > n].sort_values('n').iloc[0]['t']
                        median_visi = median_visi.append({'n': n, 't': (t1 + t2) / 2}, ignore_index=True)

                # compute timing of leaf appearance (except for first and last n)
                emerg_visi = []
                for n in sorted(list(median_visi['n'].unique()))[1:-1]:
                    t = np.mean(median_visi[median_visi['n'].isin([n - 1, n])]['t'])
                    emerg_visi.append([row['Pot'], scenario, t, n, 'visi'])
                emerg_visi = pd.DataFrame(emerg_visi, columns=['plantid', 'scenario', 't', 'n', 'type'])

                # force monotony : t = f(n) can only increase
                emerg_visi = emerg_visi.sort_values('n')
                for i_row, rowe in emerg_visi.iterrows():
                    previous = emerg_visi[emerg_visi['n'] < rowe['n']]
                    if not previous.empty:
                        emerg_visi.iloc[i_row, emerg_visi.columns.get_loc('t')] = max(rowe['t'], max(previous['t']))

                thermaltimes = np.array(df_tt['ThermalTime'])
                timestamps = np.array(df_tt['timestamp'])
                emerg_visi['t'] = [thermaltimes[np.argmin(np.abs(timestamps - t))] for t in emerg_visi['t']]

                plt.plot(emerg_visi['t'], emerg_visi['n'], '-*', label=row['Pot'])

                res_emergence.append(emerg_visi)

                # ===== emergence ligu =======================================

                df_stem = res_stem[res_stem['plantid'] == row['Pot']]
                df_profile = res_profile_h[res_profile_h['plantid'] == row['Pot']]

                # debug : some profiles may have no data for a rank (??)
                for r in range(min(df_profile['r']) + 1, max(df_profile['r'])):
                    if r not in list(df_profile['r']):
                        h1 = df_profile.sort_values('r')[df_profile['r'] < r].iloc[-1]['h']
                        h2 = df_profile.sort_values('r')[df_profile['r'] > r].iloc[0]['h']
                        h = (h1 + h2) / 2
                        df_profile = df_profile.append({'plantid':df_profile.iloc[0]['plantid'], 'scenario': scenario, 'h': h, 'r': r},
                                                       ignore_index=True).sort_values('r')

                T = np.linspace(min(df_stem['t']), max(df_stem['t']), 3000)
                f = interp1d(df_stem['t'], df_stem['z'])
                Z = np.array([f(ti) for ti in T])

                emerg_ligu = []
                ranks = sorted(df_profile['r'])
                for r in ranks[1:]:
                    t = T[np.argmin(np.abs(Z - df_profile[df_profile['r'] == r - 1].iloc[0]['h']))]
                    emerg_ligu.append([row['Pot'], scenario, t, r, 'ligu'])
                emerg_ligu = pd.DataFrame(emerg_ligu, columns=['plantid', 'scenario', 't', 'n', 'type'])

                res_emergence.append(emerg_ligu)

    res_emergence = pd.concat(res_emergence)

    # ===== plot =======

    linestyles = {'visi': 'solid', 'ligu': 'dashed'}

    # brut
    plt.figure('{} brut'.format(genotype))
    for var in ['visi', 'ligu']:
        for plantid in res_emergence['plantid'].unique():
            selec = res_emergence[(res_emergence['type'] == var) & (res_emergence['plantid'] == plantid)]
            plt.plot(selec['t'], selec['n'], '.', linestyle=linestyles[var], color=colors[selec.iloc[0]['scenario']])

    # interpolation
    res_stage = []
    T = np.linspace(min(res_emergence['t']), max(res_emergence['t']), 20)
    for var in ['visi', 'ligu']:
        for plantid in res_emergence['plantid'].unique():
            selec = res_emergence[(res_emergence['type'] == var) & (res_emergence['plantid'] == plantid)]

            if var == 'visi':
                f = interp1d(selec['t'], selec['n'])
            elif var == 'ligu':
                f = interp1d(selec['t'], selec['n'])

            t_interpolation = [t for t in T if min(selec['t']) < t < max(selec['t'])]
            for t in t_interpolation:
                res_stage.append([plantid, selec.iloc[0]['scenario'], t, float(f(t)), var])

    res_stage = pd.DataFrame(res_stage, columns=['plantid', 'scenario', 't', 'n', 'type'])

    res_gxe = res_stage.groupby(['t', 'scenario', 'type']).agg(['median', 'count', ('mad', mad)]).reset_index()
    #res_gxe = res_gxe[res_gxe['n']['count'] > 2]  # at least 3 rep

    # tmax_ligu = min(res_gxe[res_gxe['type'] == 'ligu'].groupby('scenario').max('t')['t'])
    # tmax_visi = min(res_gxe[res_gxe['type'] == 'visi'].groupby('scenario').max('t')['t'])
    # # same end
    # res_gxe = res_gxe[(res_gxe['type'] == 'visi') & (res_gxe['t'] <= tmax_visi) |
    #                   (res_gxe['type'] == 'ligu') & (res_gxe['t'] <= tmax_ligu)]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.canvas.set_window_title(genotype + '_stage')
    ax.tick_params(axis='both', which='major', labelsize=20)  # axis number size
    for var in ['visi', 'ligu']:
        for scenario in res_gxe['scenario'].unique():
            selec = res_gxe[(res_gxe['scenario'] == scenario) & (res_gxe['type'] == var)]
            plt.plot(selec['t'], selec['n']['median'], '.', linestyle=linestyles[var], color=colors[scenario])
            plt.fill_between(selec['t'],
                             selec['n']['median'] - selec['n']['mad']/1,
                             selec['n']['median'] + selec['n']['mad']/1,
                             color=colors['error'])

    handles = []
    for label, linestyle in zip(['visible', 'ligulated'], ['solid', 'dashed']):
        h, = plt.plot(-100, -100, linestyle=linestyle, markersize=12, label=label, color='black')
        handles.append(h)
    lgd1 = plt.legend(prop={'size': 20}, handles=handles, loc='lower right', markerscale=2)
    lgd1.get_frame().set_linewidth(0.0)
    plt.gca().add_artist(lgd1)
    handles = []
    for scenario in ['WW', 'WD']:
        h, = plt.plot(-100, -100, '.', markersize=12, label=scenario, color=colors[scenario])
        handles.append(h)
    # lgd2 = plt.legend(prop={'size': 20}, handles=handles, loc='upper left', markerscale=2, handletextpad=-0.5)
    # lgd2.get_frame().set_linewidth(0.0)

    #xlim, ylim = ax.get_xlim(), ax.get_ylim()
    plt.xlim((7, 93))
    plt.ylim((2, 18))
    plt.xlabel('Thermal time $(day_{20°C})$', fontsize=30)
    plt.ylabel('Leaf stage', fontsize=30)

    fig.savefig('paper/stage_{}'.format(genotype), bbox_inches='tight')
    plt.close('all')


    # ===== leaf growth =================================================================================
    for RANK in [6, 9]:

        plt.figure('{}_{}'.format(genotype, RANK))
        res_growth = []
        for scenario in ['WW', 'WD']:

            selec = gxe[(gxe['Variety_ID'] == genotype) & (gxe['Scenario'] == scenario)]
            for _, row in selec.iterrows():

                # h = length_profile(rank = 8)
                selec2 = res_profile_l[(res_profile_l['plantid'] == row['Pot']) & (res_profile_l['r'] == RANK)]
                if selec2.empty:
                    print('plantid {} no ligulated height'.format(row['Pot']))
                else:
                    l_max = selec2.iloc[0]['l']

                    # available in modulor local_benoit
                    file = 'data/tracking/gxe/{}.csv'.format(row['Pot'])
                    if not os.path.isfile(file):
                        print('plantid {} no tracking file'.format(row['Pot']))
                    else:
                        dfid = pd.read_csv(file)
                        dfr = dfid[dfid['rank_tracking'] == RANK].sort_values('timestamp').iloc[:20]

                        thermaltimes = np.array(df_tt['ThermalTime'])
                        timestamps = np.array(df_tt['timestamp'])
                        dfr['tt'] = [thermaltimes[np.argmin(np.abs(timestamps - t))] for t in dfr['timestamp']]

                        # premiere date ou on atteint 70% longueur max
                        t_start = dfr[dfr['l_extended'] > 0.70 * l_max].sort_values('timestamp').iloc[0]['tt']

                        # method 2
                        selec3 = res_emergence[(res_emergence['plantid'] == row['Pot']) &
                                      (res_emergence['type'] == 'visi') &
                                      (res_emergence['n'] == RANK)]
                        if selec3.empty:
                            print('no emergence date')
                        else:
                            t_start = selec3.iloc[0]['t']
                            print(t_start)

                        dfr['tt'] -= t_start

                        plt.plot(dfr['tt'], dfr['l_extended'], '-.', color=colors[scenario])

                        dfr = dfr[~dfr['mature']]
                        for _, row in dfr.iterrows():
                            res_growth.append([row['plantid'], scenario, row['l_extended'], row['tt']])

        res_growth = pd.DataFrame(res_growth, columns=['plantid', 'scenario', 'l', 'tt'])

        # interpolation
        res_gxe = []
        t_range = np.linspace(min(res_growth['tt']), max(res_growth['tt']), 20)
        for plantid in res_growth['plantid'].unique():
            selec = res_growth[res_growth['plantid'] == plantid]
            f = interp1d(selec['tt'], selec['l'])
            for t in [t for t in t_range if min(selec['tt']) <= t <= max(selec['tt'])]:
                res_gxe.append([plantid, selec.iloc[0]['scenario'], float(f(t)), t])
        res_gxe = pd.DataFrame(res_gxe, columns=['plantid', 'scenario', 'l', 'tt'])

        res_gxe['l'] /= 10
        res_gxe = res_gxe.groupby(['scenario', 'tt']).agg(['mean', 'median', 'count', ('mad', mad)]).reset_index()
        res_gxe = res_gxe[res_gxe['l']['count'] > 2]  # at least 3 rep

        #res_gxe = res_gxe[res_gxe['tt'] <= min(res_gxe.groupby('scenario').max('tt')['tt'])]  # same end for each scenario

        fig, ax = plt.subplots(figsize=(5, 6), dpi=150)
        fig.canvas.set_window_title(genotype + '_growth')
        ax.tick_params(axis='both', which='major', labelsize=20)  # axis number size
        for scenario in ['WW', 'WD']:

            selec = res_gxe[res_gxe['scenario'] == scenario]

            plt.plot(selec['tt'], selec['l']['median'], '-', color=colors[scenario], markersize=12)
            plt.plot(selec['tt'], selec['l']['median'], '.', color=colors[scenario], label=scenario, markersize=12)
            plt.fill_between(selec['tt'],
                             selec['l']['median'] - selec['l']['mad'] / 1,
                             selec['l']['median'] + selec['l']['mad'] / 1,
                             color=colors['error'])
        # lgd = plt.legend(prop={'size': 20}, loc='upper left', markerscale=2, handletextpad=-0.5)
        # lgd.get_frame().set_linewidth(0.0)
        #plt.xlabel('Thermal time after reaching 70% of max length (°C)', fontsize=20)
        plt.xlabel('Thermal time after leaf emergence $(day_{20°C})$', fontsize=25)
        plt.ylabel('Leaf length (cm)', fontsize=30) #labelpad=-5)

        ax.text(0.6, 0.13, 'Leaf {}'.format(RANK), transform=ax.transAxes,
                fontsize=30,
                verticalalignment='top')

        plt.xlim((-2, 27))
        plt.ylim((25, 125))

        fig.savefig('paper/growth_{}_{}'.format(RANK, genotype), bbox_inches='tight')
        plt.close('all')



