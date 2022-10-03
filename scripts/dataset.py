import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from openalea.phenomenal import object as phm_obj
from openalea.maizetrack.phenomenal_display import *

from alinea.phenoarch.cache import snapshot_index, load_collar_detection
from alinea.phenoarch.platform_resources import get_ressources
from alinea.phenoarch.meta_data import plant_data

exp = 'ZA22'

cache_client, image_client, binary_image_client, calibration_client = get_ressources(exp, cache='X:', studies='Z:', nasshare2='Y:')
index = snapshot_index(exp, image_client=image_client, cache_client=cache_client, binary_image_client=binary_image_client)

# ZA22 = 2506 pots
df_plant = plant_data(exp)

# ZA22 index = 2399 pots (pots >= 2400 not included, probably test or empty)
meta_exp = index.snapshot_index

for plantid in sorted(meta_exp['pot'].unique())[::10]:
    s = meta_exp[meta_exp['pot'] == plantid]
    plt.plot(s['timestamp'], s['pot'], 'k.-')

plant = df_plant.iloc[0]['plant']
query = index.filter(plant=plant)

meta_snapshots = index.get_snapshots(query, meta=True)
meta_snapshots = meta_snapshots[:-18]
