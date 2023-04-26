#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import NearestNeighbors
import math
import seaborn as sns
from sklearn.cluster import DBSCAN
import random
from scipy import ndarray
from matplotlib import pyplot as plt
from scipy.spatial import distance
import umap.umap_ as umap
from sklearn.cluster import KMeans
import pickle as pkl
from scipy import stats
from scipy.stats import ranksums
from config import *
from tools import *


# In[ ]:


#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# In[ ]:


imagesPath = "data/images/cluster_hypothesis_testing"
ensureDir("data/images")
ensureDir(imagesPath)


# In[ ]:


data=pd.read_csv(fileName_csv_preprocessed, index_col=0)


# In[ ]:


all_patient_ids = [0 for _ in range(1 + max(list(data["Patientennummer"].index)))]
for k in data["Patientennummer"].index:
    all_patient_ids[k] = data['Patientennummer'][k]

all_patient_ids = np.array(all_patient_ids)
all_patient_ids

data = data.drop(["Patientennummer"],axis=1)

updateMissingColumns(data)


# In[ ]:


list(data.columns)


# In[ ]:


len(data.columns)


# In[ ]:


len(cont_features)+len(nom_features)+len(ord_features)


# In[ ]:


with open(fileName_final_cluser_indexes, 'rb') as f:
    final_cluser_indexes = np.load(f,allow_pickle=True)


# In[ ]:


final_cluser_indexes


# In[ ]:


cluster_1 = data.loc[final_cluser_indexes[0]]
cluster_2 = data.loc[final_cluser_indexes[1]]
cluster_3 = data.loc[final_cluser_indexes[2]]
cluster_4 = data.loc[final_cluser_indexes[3]]
cluster_5 = data.loc[final_cluser_indexes[4]]


# new_column_names=[ 'KHK ', 'Herzerkrankung', 'pAVK', 'COPD/Asthma', 'Immunologische Erkrankungen', 'Malignom vor OP','Diabetes mellitus',
#        'Hypertonie', 'Diabetes Folgeschäden ', 'klinisches Grading', 'NTx-Anzahl',
#        'Immunologisches Grading', 'Grund für TX', 'Geburten und co. Ja/nein',
#        'Transfusionen in der Vergangenheit', 'Dauer in Minuten',
#        'Dialysezeit (Tage)', 'm = 1 ; f = 0', 'ZAHL Empfänger(CMV positiv=1)',
#        'donor m = 1 ; f = 0', 'Blutgruppe  Spender', 
#        'Donor CMV Positiv  = 1', 'Todspende',
#        'Todesursache', 'HLA A Mismatch', 'HLA B Mismatch',
#        'HLA DR Mismatch', 'Biospie ja/nein', 'erste Ausfuhr am:',
#        'Ausfuhr bei Entlassung (ml)',
#        'Kreatinin bei Entlassung (ymol/l)', '2 Wochen post OP', '3 Monate post OP ymol/l',
#        '1 Jahr post OP', 'GFR_bei_Entlassung ', 'GFR_2Wochen_postOP',
#        'GFR_3_Monate_post_OP', 'GFR_1_Jahr_post_OP', 'Abstoßungsreaktion', 'Banff',
#        'Grad', 'HWI',
#        'Atemwegsinfekt', 'M-D-T', 'Sepsis',
#        'CMV', 'Pilzinfektion', 'Virusinfektionen',
#        'Wundheilungsstörung', 'Katheterinfektion', 'Sonstige', 'Infektionen', 'Malignome nach OP', 'Explantation',
#        'Revisions OP', 'post OP Dialyse ja/nein', 'dialysefrei ja/nein', 'WIZ',
#        '(High risk drug;) Immunadsoprtionstherapie ','(High risk drug)  Rituximab/ Immunglobuline', 'MMF',
#        'Everolimus', 'Sirolimus', 'Azathioprin','Prednisolon', 'Basiliximab (Simulect)',
#        'ATG (Antithymozytenglobulin)', 'Urbanstoßtherapie',
#        'Prograf+Cellcept', 'Prograf+Myfortic',
#        'Cellcept+CyA', 'Cellcept+Myfortic', 'Cell. --> Myfortic', 'Konversion auf andere',
#        'Tacrolimus Spiegel Nachkontrolle',
#        'Cyc Spiegel bei Entlassung (yg/l)', 'Cyc Spiegel bei Nachsorge',
#         'Blutgruppe Empfänger', 'Range_Gestorben', 'Range_Explantation','Rh Compatibility',
#        'MaFr1FrMa2FrFr3MaMa4', 'TC_switch',
#        'Tacrolimus Spiegel bei Entlassung (ng/ml)',
#        'Alter bei Tx', 'BMI Empfänger', 'BMI Spender','Clusters']
# zip_iterator = zip(list(data.columns), new_column_names)
# feat_names_dict = dict(zip_iterator)
# feat_names_dict

# In[ ]:


def p_val(clustera,clusterb,feature,var_type):
    if var_type=='cont':
        p=stats.ttest_ind(np.array(clustera[feature]), np.array(clusterb[feature])).pvalue
    else:
        p=ranksums(np.array(clustera[feature]), np.array(clusterb[feature])).pvalue
    return p


# In[ ]:


cluster_list=[cluster_1,cluster_2,cluster_3,cluster_4,cluster_5]
def feature_p_val(feature,cluster_list,var_type):
    num_clusters=len(cluster_list)
    p_list_all_clusters=[]
    for i in range(num_clusters):
        p_list=[]
        for j in range(num_clusters):
            p=p_val(cluster_list[i],cluster_list[j],feature,var_type)
            p_list.append(p)
        p_list=np.array(p_list)
        p_list_all_clusters.append(p_list)  
    return(np.array(p_list_all_clusters))
        


# In[ ]:


import matplotlib as mpl

class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))


# In[ ]:


def p_map(feature, var_type, n):
    heatmap, ax = plt.subplots()
    norm = MidpointNormalize(vmin=0, vmax=1, midpoint=0.5)
    im = ax.imshow(feature_p_val(feature,cluster_list,var_type), cmap='coolwarm', norm=norm)
    ax.set_xticklabels(['','C1','C2','C3','C4','C5'])
    ax.set_yticklabels(['','C1','C2','C3','C4','C5'])

    for y in range(len(cluster_list)):
        for x in range(len(cluster_list)):
            plt.text(x , y , '%.2f' % feature_p_val(feature,cluster_list, var_type)[y, x], horizontalalignment='center',verticalalignment='center')

    cbar = heatmap.colorbar(im)
    cbar.ax.set_ylabel('p-value')
    plt.title(feature, fontsize=8)
    plt.savefig(f'{imagesPath}/{var_type}_{n}_{safeFilename(feature)}.pdf', dpi=700, bbox_inches='tight')
    plt.close()


# In[ ]:


for n, feature in enumerate(cont_features):
    p_map(feature, 'cont', n + 1)


# In[ ]:


for n, feature in enumerate(ord_features):
    p_map(feature, 'ord', n + 1)


# In[ ]:


for n, feature in enumerate(nom_features):
    p_map(feature, 'nom', n + 1)


# In[ ]:




