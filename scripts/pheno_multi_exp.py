import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def date_to_timestamp(date):
    return time.mktime(datetime.datetime.strptime(date[:10], "%Y-%m-%d").timetuple())

# ===== generate a csv with phenology data from different maize experiments ===========================================

df = []

exp = 'ZB14'
df_pheno = pd.read_csv('data/pheno_{}.csv'.format(exp))
df_pheno['Date2'] = ['{}-{}-{}'.format(d.split('/')[2], d.split('/')[1], d.split('/')[0]) for d in df_pheno['Date']]
df_pheno['timestamp'] = df_pheno['Date2'].map(date_to_timestamp)
s = df_pheno[~df_pheno['Pheno'].isna()]
for _, row in s.iterrows():
    df.append([exp, int(row['ID code'][:4]), row['Date2'], row['timestamp'], 'visi_f', row['Pheno']])
s = df_pheno[~df_pheno['Ligulee'].isna()]
for _, row in s.iterrows():
    df.append([exp, int(row['ID code'][:4]), row['Date2'], row['timestamp'], 'ligul_f', row['Ligulee']])

exp = 'ZA17'
df_pheno = pd.read_csv('data/pheno_{}.csv'.format(exp))
df_pheno['timestamp'] = df_pheno['resultdate'].map(date_to_timestamp)
s = df_pheno[df_pheno['observationcode'].isin(['visi_f', 'ligul_f'])]
for _, row in s.iterrows():
    df.append([exp, row['Pot'], row['resultdate'][:10], row['timestamp'], row['observationcode'], row['observation']])

exp = 'ZA22'
df_pheno = pd.read_csv('data/pheno_{}.csv'.format(exp))
df_pheno['timestamp'] = df_pheno['resultdate'].map(date_to_timestamp)
s = df_pheno[df_pheno['observationcode'].isin(['visi_f', 'ligul_f', 'leaf_tot'])]
s = s[~((s['observationcode'] == 'visi_f') & (s['observation'].isin(['n', 'y', '12..5', '16..5'])))]
s = s[~((s['observationcode'] == 'ligul_f') & (s['observation'].isin(['pas demar', '14.5', '10.5', '16.8', '610.5'])))]
s = s[~((s['observationcode'] == 'leaf_tot') & (s['observation'].isin([k for k in s['observation'].unique() if not k.isdigit()])))]
s['observation'] = pd.to_numeric(s['observation'])
for _, row in s.iterrows():
    df.append([exp, row['Pot'], row['resultdate'][:10], row['timestamp'], row['observationcode'], row['observation']])

df = pd.DataFrame(df, columns=['exp', 'plantid', 'daydate', 'timestamp', 'observationcode', 'observation'])
df.to_csv('data/pheno.csv', index=False)

# ===== visualize data =============================================================================================

df = pd.read_csv('data/pheno.csv')

for code in ['visi_f', 'ligul_f']:
    plt.figure()
    for exp, col in zip(df['exp'].unique(), ['r', 'g', 'b']):
        s = df[(df['exp'] == exp) & (df['observationcode'] == code)]
        plt.plot(s['timestamp'] - np.min(s['timestamp']), s['observation'], 'o', color=col, alpha=0.03, label=exp)

        gb = s.groupby('timestamp').mean().reset_index()
        plt.plot(gb['timestamp'] - np.min(s['timestamp']), gb['observation'], '.-', color=col)

    leg = plt.legend()
    for lh in leg.legendHandles:
        lh._legmarker.set_alpha(1)
    plt.title(code)





