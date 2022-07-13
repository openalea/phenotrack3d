import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import time
import datetime

def date_to_timestamp(date):
    return time.mktime(datetime.datetime.strptime(date[:10], "%Y-%m-%d").timetuple())

# available in modulor local_benoit
folder = 'data/tracking/'
plantids = [int(f.split('.')[0]) for f in os.listdir(folder) if os.path.isfile(folder + f)]

#df_pheno = pd.read_csv('pheno_ZA17.csv')
#df_pheno[(df_pheno['Pot'] == plantid) & (df_pheno['observationcode'] == 'panicule_visi')].sort_values('resultdate').iloc[0]['resultdate']

# ===== group all plantids in a same dataframe ===============================

df_list = []
for plantid in plantids:

    dfid = pd.read_csv(folder + '{}.csv'.format(plantid))

    df_list.append(dfid)
df = pd.concat(df_list)

df = df[~df['rank_annotation'].isin([-1, 0])]

# available in modulor
tt = pd.read_csv('TT_ZA17.csv')
t = [tt[tt['timestamp'] < t].sort_values('timestamp').iloc[-1]['ThermalTime'] for t in df['timestamp']]
df['tt'] = t

#df = df[df['timestamp'] < 1496872800]  # 06-08
df = df[df['timestamp'] < 1496509351]  # 06-03
#df = df[df['timestamp'] < 1495310577]  # 05-20

# ===== accuracy per plant ===================================================

#selec = df
n, accuracy = [], []
for plantid in df['plantid'].unique():

    selecid = df[df['plantid'] == plantid]

    selec = selecid[(selecid['mature'] == True)]
    a1 = len(selec[selec['rank_annotation'] == selec['rank_tracking']]) / len(selec)
    selec = selecid[(selecid['mature'] == False)]
    a2 = len(selec[selec['rank_annotation'] == selec['rank_tracking']]) / len(selec)
    accuracy.append(a1)
    print(plantid, round(a1, 3), round(a2, 3), len(selecid))
    n.append(len(selecid))
print('=====================================')
print(min(accuracy), np.mean(accuracy))
print(min(n), np.mean(n))

# rmse
df2 = df[~(df['rank_tracking'] == 0)]
x, y = df2[df2['mature']]['rank_annotation'], df2[df2['mature']]['rank_tracking']
rmse = np.sqrt(np.sum((x - y) ** 2) / len(x))
print('mature: RMSE = ', rmse)
x, y = df2[~df2['mature']]['rank_annotation'], df2[~df2['mature']]['rank_tracking']
rmse = np.sqrt(np.sum((x - y) ** 2) / len(x))
print('growing: RMSE = ', rmse)

# Brut : 94.9%. -> 06-03 : 97.7%.

# ======= accuracy per plant but no senescence ===========================

n, accuracy = [], []
for plantid in df['plantid'].unique():
    dfid = df[df['plantid'] == plantid]
    ranks = [r for r in sorted(dfid['rank_annotation'].unique())]
    total, correct = 0, 0
    for rank in ranks:
        selec = dfid[(dfid['rank_annotation'] == rank) & (dfid['mature'])]
        selec = selec.sort_values('timestamp').iloc[:15]
        total += len(selec)
        correct += len(selec[selec['rank_annotation'] == selec['rank_tracking']])
    accuracy.append(correct / total)
    n.append(total)
print(min(accuracy), np.mean(accuracy))
print(min(n), np.mean(n))


# ===== accuracy per rank ===================================================

ranks = [r for r in sorted(df['rank_annotation'].unique())]
res = []
for m in [True, False]:
    selec = df[(df['mature'] == m)]
    n, accuracy = [], []
    for rank in ranks:
        selecrank = selec[selec['rank_annotation'] == rank]
        if not selecrank.empty:
            accuracy = len(selecrank[selecrank['rank_annotation'] == selecrank['rank_tracking']]) / len(selecrank) * 100
            n = len(selecrank)
            res.append([m, rank, accuracy, n])
res = pd.DataFrame(res, columns=['mature', 'rank', 'accuracy', 'n'])

fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # int axis
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Leaf rank', fontsize=30)
plt.ylabel('Tracking accuracy (%)', fontsize=30)
plt.xlim([0.5, 17])
plt.ylim([48, 102])
plt.plot([-100, 100], [100, 100], 'k--')

res = res[res['n'] > 30]
resm = res[res['mature']]
plt.plot(resm['rank'], resm['accuracy'], '^-', color='blue', label='Ligulated leaves', markersize=9)
resg = res[res['mature'] == False]
plt.plot(resg['rank'], resg['accuracy'], 'o-', color='orange', label='Growing leaves', markersize=9)

leg = plt.legend(prop={'size': 20}, loc='lower right')
leg.get_frame().set_linewidth(0.0)

fig.savefig('paper/tracking_accuracy', bbox_inches='tight')


# ===== accuracy per time period ===========================================

#df['period'] = np.digitize(df['timestamp'], bins=np.linspace(min(df['timestamp']), max(df['timestamp']), 10))
df['period'], limits = pd.qcut(df['tt'], 15, retbins=True)

res2 = []
for p in df['period'].unique():
    dfp = df[df['period'] == p]
    for m in [True, False]:
        selec = dfp[(dfp['mature'] == m)]
        n, accuracy = [], []
        accuracy = len(selec[selec['rank_annotation'] == selec['rank_tracking']]) / len(selec) * 100
        n = len(selec)
        res2.append([p, m, accuracy, n])
res2 = pd.DataFrame(res2, columns=['period', 'mature', 'accuracy', 'n'])

fig, ax = plt.subplots()
ax.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Thermal time', fontsize=30)
plt.ylabel('Tracking accuracy (%)', fontsize=30)
plt.xlim([min(df['tt']), max(df['tt'])])
plt.ylim([-4, 104])
plt.plot([-100, 100], [100, 100], 'k--')

res2 = res2[res2['n'] > 30]
resm = res2[res2['mature']]
t = [float(str(x).split(',')[0][1:]) for x in resm['period']]
plt.plot(t, resm['accuracy'], '^-', color='blue', label='Ligulated leaves', markersize=9)
resg = res2[res2['mature'] == False]
t = [float(str(x).split(',')[0][1:]) for x in resg['period']]
plt.plot(t, resg['accuracy'], 'o-', color='orange', label='Growing leaves', markersize=9)
plt.legend(prop={'size': 30}, loc='lower left')

# ======== test heatmap ====================================

dfm = df[df['mature']]

dfm['r_interval'] = pd.qcut(dfm['rank_annotation'], 8)
dfm['t_interval'] = pd.qcut(dfm['timestamp'], 8)

m = np.zeros((len(dfm['r_interval'].unique()), len(dfm['t_interval'].unique())))
for i, rint in enumerate(dfm['r_interval'].unique()):
    for j, tint in enumerate(dfm['t_interval'].unique()):
        selec = dfm[(dfm['r_interval'] == rint) & (dfm['t_interval'] == tint)]
        if len(selec) > 50:
            accuracy = len(selec[selec['rank_annotation'] == selec['rank_tracking']]) / len(selec) * 100
            m[i, j] = accuracy
        else:
            m[i, j] = np.nan

plt.imshow(m, vmin=85, vmax=100, cmap='RdYlGn')
im_ratio = m.shape[0] / m.shape[1]
plt.colorbar(fraction=0.046 * im_ratio, pad=0.04, label='Accuracy (%)')
plt.xlabel('time interval')
plt.ylabel('rank interval')


# ===== growing leaf : jusqu'a quand on peut rembobiner =========================

dfg = df[~df['mature']]

rembobinage = []
for _, row in dfg.iterrows():
    selec = dfg[(dfg['plantid'] == row['plantid']) & (dfg['rank_annotation'] == row['rank_annotation'])]
    selec = selec.sort_values('tt')
    rembobinage.append(np.max(selec['tt']) - row['tt'])
dfg['rembobinage'] = rembobinage

dfg['rb_interval'] = pd.qcut(dfg['rembobinage'], 8)

t = [float(str(x).split(',')[0][1:]) for x in dfg['rb_interval']]

x, y = [], []
for tint in sorted(dfg['rb_interval'].unique()):
    x.append(float(str(tint).split(',')[0][1:]))
    selec = dfg[dfg['rb_interval'] == tint]
    accuracy = len(selec[selec['rank_annotation'] == selec['rank_tracking']]) / len(selec) * 100
    y.append(accuracy)
    print(len(selec))

plt.plot(x, y, 'k-*')
plt.xlabel('thermal time before ligulation')
plt.ylabel('accuracy')







