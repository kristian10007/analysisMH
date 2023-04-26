#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from config import *
from tools import *
from revdict import *


# In[2]:


#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# In[3]:


imagesPath = "data/images/Stratification"
ensureDir("data/images")
ensureDir(imagesPath)


# In[4]:


data=pd.read_csv(fileName_csv_preprocessed, index_col=0)


# In[5]:


all_patient_ids = [0 for _ in range(1 + max(list(data["Patientennummer"].index)))]
for k in data["Patientennummer"].index:
    all_patient_ids[k] = data['Patientennummer'][k]

all_patient_ids = np.array(all_patient_ids)
all_patient_ids

data = data.drop(["Patientennummer"],axis=1)

updateMissingColumns(data)


# In[6]:


list(data.columns)


# In[7]:


len(cont_features+nom_features+ord_features)


# ## The clustering paradigm
# 
# - After we have the pre-processed data we will employ a feature-distributed dimension reduction. Since the data has continuous, ordinal and nominal features, we adapt this strategy. The reason is, otherwise, the continuous features often tend to have more influence on the clustering. We use 3 kind of similarity measures as UMAP parameter (Euclidean for continuous features, Canberra for ordinal features and Cosine for nominal features)
# 
# - For each of the three feature-types the whole data is reduced to two dimensions, thes generating a 2x3=6 dimensional representation of the data. Note that each of these dimensions are normalized. 
# 
# - We use a clustering algorithm to extract the clusters from the 6-Dimensional data
# 
# - We again visualize the clusters in a further UMAP-reduced 2-dimensional version of the 6-dimensional feature-distriuted data embedding obtained from the second step.

# ## UMAP for features with ordinal values

# In[8]:


data_embedded = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2, metric='canberra', random_state=42).fit_transform(data[ord_features])
data_embedded[:,0]=(data_embedded[:,0]- np.mean(data_embedded[:,0]))/np.std(data_embedded[:,0])
data_embedded[:,1]=(data_embedded[:,1]- np.mean(data_embedded[:,1]))/np.std(data_embedded[:,1])
result_of = pd.DataFrame(data = data_embedded, columns = ['UMAP_0_of', 'UMAP_1_of'])


# In[9]:


sns.lmplot( x="UMAP_0_of", y="UMAP_1_of",
  data=result_of, 
  fit_reg=False, 
  legend=False,
  scatter_kws={"s": 10},palette="Set1") # specify the point size
plt.savefig(imagesPath + '/clusters_umap_of.pdf', dpi=700, bbox_inches='tight')
plt.close()


# ## UMAP for features with continuous values

# In[10]:


data_embedded = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2, metric='euclidean', random_state=42).fit_transform(data[cont_features])
data_embedded.shape
data_embedded[:,0]=(data_embedded[:,0]- np.mean(data_embedded[:,0]))/np.std(data_embedded[:,0])
data_embedded[:,1]=(data_embedded[:,1]- np.mean(data_embedded[:,1]))/np.std(data_embedded[:,1])
result_cf = pd.DataFrame(data=data_embedded, columns=['UMAP_0_cf', 'UMAP_1_cf'])


# In[11]:


sns.lmplot( x="UMAP_0_cf", y="UMAP_1_cf",
  data=result_cf, 
  fit_reg=False, 
  legend=False,
  scatter_kws={"s": 10},palette="Set1") # specify the point size
plt.savefig(imagesPath + '/clusters_umap_cf.pdf', dpi=700, bbox_inches='tight')
plt.close()


# ## UMAP for features with nominal values

# In[12]:


data_embedded = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, metric='hamming', random_state=42).fit_transform(data[nom_features])
data_embedded.shape
data_embedded[:,0]=(data_embedded[:,0]- np.mean(data_embedded[:,0]))/np.std(data_embedded[:,0])
data_embedded[:,1]=(data_embedded[:,1]- np.mean(data_embedded[:,1]))/np.std(data_embedded[:,1])
result_nf = pd.DataFrame(data = data_embedded, columns = ['UMAP_0_nf', 'UMAP_1_nf'])


# In[13]:


sns.lmplot( x="UMAP_0_nf", y="UMAP_1_nf",
  data=result_nf, 
  fit_reg=False, 
  legend=False,
  scatter_kws={"s": 10},palette="Set1") # specify the point size
plt.savefig(imagesPath + '/clusters_umap_nf.pdf', dpi=700, bbox_inches='tight')
plt.close()


# ## Integration of feature-distributed UMAP

# In[14]:


result=pd.concat([result_of, result_cf, result_nf.drop(columns=['UMAP_1_nf'])],axis=1)


# In[15]:


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        feat_mean = df[feature_name].mean()
        feat_sd = df[feature_name].std()
        result[feature_name] = (df[feature_name] - feat_mean) / feat_sd
    return result


# In[16]:


np.random.seed(42)
data_embedded = umap.UMAP(n_neighbors=5, min_dist=0.01, n_components=2, metric='euclidean', random_state=14).fit_transform(result)
result = pd.DataFrame(data=data_embedded, columns=['UMAP_0', 'UMAP_1'])


# In[17]:


result_mat=np.array(result)


# ## Finding clusters in feature-distributed UMAP with DBSCAN

# In[18]:


from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score, silhouette_samples
fig, ax = plt.subplots(4, 2, figsize=(10,8))
fig.tight_layout(pad=2.0)
for i in [2, 3, 4, 5, 6, 7, 8, 9]:
    '''
    Create KMeans instance for different number of clusters
    '''
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)
    '''
    Create SilhouetteVisualizer instance with KMeans instance
    Fit the visualizer
    '''
    labels=km.fit_predict(result_mat)
    s=silhouette_score(result_mat, labels)
    
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    
    ax[q-1][mod].title.set_text('k='+str(i)+' Silhouette Score: '+str(s))
    ax[q-1][mod].grid(False)
    visualizer.fit(result_mat)
fig.savefig(imagesPath + '/clusters_in_feature_distributed_umap_with_dbscan.pdf', dpi=700, bbox_inches='tight')
plt.close()


# In[19]:


alg_cluster= KMeans(n_clusters=5, random_state=0, n_init=10)

clusters=alg_cluster.fit_predict(result_mat)
(values,counts) = np.unique(clusters,return_counts=True)


# In[20]:


result['Cluster'] = clusters


# In[21]:


resultx=result
resultx['Cluster']=clusters+1
sns.lmplot( x="UMAP_0", y="UMAP_1",
  data=result, 
  fit_reg=False, 
  legend=True,
  hue='Cluster', # color by cluster
  scatter_kws={"s": 10},palette="Set1") # specify the point size
plt.grid(False)
plt.savefig(imagesPath + '/clusters_umap.pdf', dpi=700, bbox_inches='tight')
plt.close()


# In[22]:


values,counts=np.unique(clusters,return_counts=True)


# - After we visualize the obtained clusters we check their cardinalities
# - In the above plot you can see the cluster numbers

# In[23]:


plt.bar(values,counts,tick_label=values+1, )
plt.grid(False)
plt.xlabel('Clusters')
plt.ylabel('Number of patients')
plt.title('Distribution of clusters')
plt.savefig(imagesPath + '/Distribution_of_clusters.pdf', dpi=700, bbox_inches='tight')
plt.close()


# ## Summary statistics for clusters for every feature

# Here we provide feature-wise statistics for each cluster:
# - For discrete variables we provide value-wise frequency distribution
# - For each feature, we visualize feature value frequencies for each cluster
# - For each feature, we provide cluster mean, standard deviation and median for each cluster
# - For each feature, we visualize the cluster means with error bars
# - For discrete features, we visualize the distribution of the features mapped on the feature-distributed final UMAP embedding

# In[24]:


data['Clusters']=clusters


# In[25]:


data['Discrete_Explantation'] = discretizeDaysWithinYear(data['Range_Explantation'])


# In[26]:


data['Discrete_Gestorben'] = discretizeDaysWithinYear(data['Range_Gestorben'])


# In[27]:


cluster_df_list=[]
for cluster in values:
    cluster_df=data.loc[data['Clusters'] == cluster].drop(columns=['Clusters'])
    #cluster_df=cluster_df.drop(['Clusters'],axis=1)
    cluster_df_list.append(cluster_df)


# In[28]:


rev_dict = loadAutoRevDict()

rd = {}
for c in rev_dict.keys():
    print(f"'{c}':")
    values = set()
    d = {}
    for k in rev_dict[c]:
        v = rev_dict[c][k]
        print(f"   {v} -> '{k}'")
        if v not in values:
            values.add(v)
            d[k] = v
    rd[c] = d
rev_dict = rd


# In[29]:


def vizx(vizDir, feature_list, cluster_df_list, main_data, umap_data, cont_features):
    #get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
    vizlimit = 15
    plt.rcParams["figure.figsize"] = (12,6)
    sns.set_style("white")
    
    
    for featureNr, feature in enumerate(feature_list):
        
        fileBaseName = f"{vizDir}/{featureNr + 1}_{safeFilename(feature)}"
        logFile = open(fileBaseName + ".txt", "w")
        
        print(f'Feature {featureNr + 1} name: {feature.upper()}')
        print(f'Feature name: {feature.upper()}\n', file=logFile)
    
        if len(main_data[feature].value_counts())<=vizlimit:
            for cluster in range(len(cluster_df_list)):
                print(f'Cluster {cluster + 1} frequeny distribution', file=logFile)
                if feature in list(rev_dict.keys()):
                    feat_keys = rev_dict[feature]
                    r = dict(zip(feat_keys.values(), feat_keys.keys()))
                    print(cluster_df_list[cluster].replace({feature:r})[feature].value_counts(), file=logFile)
                else:
                    print(cluster_df_list[cluster][feature].value_counts(), file=logFile)

                print('\n', file=logFile)
        
        col = sns.color_palette("Set2")
        cluster_bar=[]     
        for cluster in range(len(cluster_df_list)):
            if len(main_data[feature].value_counts())<=vizlimit:
                if feature in list(rev_dict.keys()):
                    y = cluster_df_list[cluster].replace({feature:r})[feature].value_counts()
                else:
                    y = cluster_df_list[cluster][feature].value_counts().sort_index()
                x = y.index
                cluster_bar.append([[str(z) for z in x], np.array(y)])
                    
        rows = 1
        columns = len(cluster_df_list)
        
        if len(main_data[feature].value_counts()) <= vizlimit:
            figx, ax = plt.subplots(rows, columns)
            figx.set_size_inches(25, 5)
            cluster_in_subplot_axis_dict = np.array([0,1,2,3,4,5,6])
            
            for j in range(columns):
                ax[j].bar(cluster_bar[j][0], cluster_bar[j][1], color=col, width=.3)
                ax[j].tick_params(axis='x', which='major', labelsize=8, rotation=90)
                ax[j].set_title(f'Cluster: {j + 1}')
            plt.savefig(fileBaseName + '.pdf', dpi=700, bbox_inches='tight')
            plt.close()

            
        means = []
        sds = []
        cluster_labels = []
        cluster_counter = 1
        for cluster in range(len(cluster_df_list)):
            if feature in cont_features:
                print(f'Cluster {cluster_counter} summary statistics\n', file=logFile)
                cm = cluster_df_list[cluster][feature].mean()
                cs = cluster_df_list[cluster][feature].std()
                print(f'feature mean: {cm}', file=logFile)
                print(f'feature standard deviation: {cs}', file=logFile)
                print(f'feature median: {cluster_df_list[cluster][feature].median()}', file=logFile)
                print('\n', file=logFile)
                means.append(cm)
                sds.append(cs)
                cluster_labels.append(f'C{cluster_counter}')
            cluster_counter += 1
            
            
        means = np.array(means)
        sds = np.array(sds)
        cluster_labels = np.array(cluster_labels)
        
        if feature in cont_features:   
            #print('\n', file=logFile)
            #print('Distribution of feature across clusters', file=logFile)
            fig, ax7 = plt.subplots()
            ax7.bar(cluster_labels,means,yerr=sds,color=sns.color_palette("Set3"))
            ax7.tick_params(axis='both', which='major', labelsize=10)
            plt.xlabel(feature, fontsize=15)
            plt.savefig(fileBaseName + '_Distribution_of_feature_across_clusters.pdf', dpi=700, bbox_inches='tight')
            plt.close()
        
        
        colors_set = ['lightcoral','cornflowerblue','orange','mediumorchid', 'lightseagreen','olive', 'chocolate','steelblue',"paleturquoise",  "lightgreen",  'burlywood','lightsteelblue']
        customPalette_set = sns.set_palette(sns.color_palette(colors_set))
        
        if feature not in cont_features:
            #print('\n\n', file=logFile)
            #print('Feature distribution in UMAP embedding', file=logFile)
            if feature in list(rev_dict.keys()):
                umap_data[feature]=np.array(main_data.replace({feature:r})[feature])
            else:
                umap_data[feature]=np.array(main_data[feature])
            sns.lmplot(  x="UMAP_0", y="UMAP_1",
              data=umap_data, 
              fit_reg=False, 
              legend=True,
              hue=feature, # color by cluster
              scatter_kws={"s": 20},palette=customPalette_set) # specify the point size
            plt.savefig(fileBaseName + '_Feature_distribution_in_UMAP_embedding.pdf', dpi=700, bbox_inches='tight')
            plt.close()
        
        logFile.close()
        


# In[30]:


for_viz = list(set(data.columns) - set(['Range_Explantation', 'Range_Gestorben', 'Clusters']))


# ## Visualization

# In[31]:


vizDir = imagesPath + "/visuailzation"
ensureDir(vizDir)
vizx(vizDir, for_viz, cluster_df_list, data, result, cont_features)


# ## Saving patient indexes cluster-wise

# In[32]:


final_cluser_indexes = []
for i in range(len(cluster_df_list)):
    index_list = np.array((cluster_df_list[i].index))
    final_cluser_indexes.append(index_list)
final_cluser_indexes = np.array(final_cluser_indexes)


# In[33]:


with open(fileName_final_cluser_indexes,'wb') as f:
    np.save(f, final_cluser_indexes)


# In[34]:


np.load(fileName_final_cluser_indexes, allow_pickle=True)


# In[35]:


final_cluser_patient_ids = []
for cluster in cluster_df_list:
    index_list = np.array(cluster.index)
    patient_ids = all_patient_ids[index_list]
    final_cluser_patient_ids.append(patient_ids)

final_cluser_patient_ids = np.array(final_cluser_patient_ids, dtype=object)

with open(fileName_final_cluser_patient_id, 'wb') as f:
    np.save(f, final_cluser_patient_ids)
    
np.load(fileName_final_cluser_patient_id, allow_pickle=True)


# In[ ]:




