#!/usr/bin/env python
# coding: utf-8

# # Step 4: Hypothesis testing

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
from scipy.spatial import distance
import umap.umap_ as umap
from sklearn.cluster import KMeans
import pickle as pkl
from scipy import stats
from scipy.stats import ranksums
from config import *
from tools import *

imagePath = "data/images/stratification"
ensureDir(imagePath)



# In[ ]:


data = pd.read_csv('preproceesed_imputed_data_Hilke.csv',index_col=0)
data = data.drop(["Patientennummer"],axis=1)


# In[ ]:


print()
print("Columns:")
data.columns
updateMissingColumns(data)


# In[ ]:


len(cont_features)+len(nom_features)+len(ord_features)


# In[ ]:


with open('final_cluser_indexes.npy','rb') as f:
    final_cluser_indexes = np.load(f,allow_pickle=True)


# In[ ]:


final_cluser_indexes


# In[ ]:


cluster_1=data.loc[final_cluser_indexes[0]]
cluster_2=data.loc[final_cluser_indexes[1]]
cluster_3=data.loc[final_cluser_indexes[2]]
cluster_4=data.loc[final_cluser_indexes[3]]
cluster_5=data.loc[final_cluser_indexes[4]]
cluster_6=data.loc[final_cluser_indexes[5]]
cluster_7=data.loc[final_cluser_indexes[6]]


# In[ ]:


def p_val(clustera,clusterb,feature,var_type):
    if var_type=='cont':
        p=stats.ttest_ind(np.array(clustera[feature]), np.array(clusterb[feature])).pvalue
    else:
        p=ranksums(np.array(clustera[feature]), np.array(clusterb[feature])).pvalue
    return p


# In[ ]:


cluster_list=[cluster_1,cluster_2,cluster_3,cluster_4,cluster_5,cluster_6,cluster_7]
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


def p_map(n, feature,var_type):
    heatmap, ax = plt.subplots()
    norm = MidpointNormalize(vmin=0, vmax=1, midpoint=0.5)
    im = ax.imshow(feature_p_val(feature,cluster_list,var_type), cmap='coolwarm', norm=norm)
    ax.set_xticklabels(['','C1','C2','C3','C4','C5','C6','C7'])
    ax.set_yticklabels(['','C1','C2','C3','C4','C5','C6','C7'])

    for y in range(len(cluster_list)):
        for x in range(len(cluster_list)):
            plt.text(x , y , '%.2f' % feature_p_val(feature,cluster_list, var_type)[y, x], horizontalalignment='center',verticalalignment='center')

    cbar = heatmap.colorbar(im)
    cbar.ax.set_ylabel('p-value')
    plt.title(feature, fontsize=8)
    plt.savefig(f'{imagePath}/{n}_{safeFilename(feature)}.pdf', bbox_inches='tight')
    plt.close()


# In[ ]:

n = 0
for feature in cont_features:
    p_map(n, feature, 'cont')
    n += 1

for feature in ord_features:
    p_map(n, feature, 'ord')
    n += 1

for feature in nom_features:
    p_map(n, feature, 'nom')
    n += 1


# In[ ]:




