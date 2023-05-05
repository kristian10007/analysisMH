#!/usr/bin/env python
# coding: utf-8

# # Step 3: Stratification

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
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from config import *
from tools import *

imagePath = "data/images/stratification"
ensureDir(imagePath)




# In[ ]:


data=pd.read_csv('preproceesed_imputed_data_Hilke.csv',index_col=0)


# In[ ]:


all_patient_ids = [0 for _ in range(1 + max(list(data["Patientennummer"].index)))]
for k in data["Patientennummer"].index:
    all_patient_ids[k] = data['Patientennummer'][k]

all_patient_ids = np.array(all_patient_ids)

print()
print("Patient IDs:")
print(all_patient_ids)


# In[ ]:


data = data.drop(["Patientennummer"],axis=1)


# In[ ]:

print()
print("Columns:")
print(data.columns)

updateMissingColumns(data)

# In[ ]:


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

# In[ ]:


data_embedded = umap.UMAP(n_neighbors=5, min_dist=0.01, n_components=2, metric='canberra', random_state=42).fit_transform(data[ord_features])
data_embedded[:,0]=(data_embedded[:,0]- np.mean(data_embedded[:,0]))/np.std(data_embedded[:,0])
data_embedded[:,1]=(data_embedded[:,1]- np.mean(data_embedded[:,1]))/np.std(data_embedded[:,1])
result_of = pd.DataFrame(data = data_embedded , 
        columns = ['UMAP_0_of', 'UMAP_1_of'])


# ## UMAP for features with continuous values

# In[ ]:


data_embedded = umap.UMAP(n_neighbors=5, min_dist=0.01, n_components=2, metric='euclidean', random_state=42).fit_transform(data[cont_features])
data_embedded.shape
data_embedded[:,0]=(data_embedded[:,0]- np.mean(data_embedded[:,0]))/np.std(data_embedded[:,0])
data_embedded[:,1]=(data_embedded[:,1]- np.mean(data_embedded[:,1]))/np.std(data_embedded[:,1])
result_cf = pd.DataFrame(data = data_embedded , 
        columns = ['UMAP_0_cf', 'UMAP_1_cf'])


# ## UMAP for features with nominal values

# In[ ]:


data_embedded = umap.UMAP(n_neighbors=5, min_dist=0.01, n_components=2, metric='hamming', random_state=42).fit_transform(data[nom_features])
data_embedded.shape
data_embedded[:,0]=(data_embedded[:,0]- np.mean(data_embedded[:,0]))/np.std(data_embedded[:,0])
data_embedded[:,1]=(data_embedded[:,1]- np.mean(data_embedded[:,1]))/np.std(data_embedded[:,1])
result_nf = pd.DataFrame(data = data_embedded , 
        columns = ['UMAP_0_nf', 'UMAP_1_nf'])


# ## Integration of feature-distributed UMAP

# In[ ]:


result=pd.concat([result_of, result_cf, result_nf],axis=1)


# In[ ]:


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        feat_mean = df[feature_name].mean()
        feat_sd = df[feature_name].std()
        result[feature_name] = (df[feature_name] - feat_mean) / feat_sd
    return result


# In[ ]:


np.random.seed(42)
data_embedded = umap.UMAP(n_neighbors=20, min_dist=0.01, n_components=2, metric='euclidean', random_state=42).fit_transform(result)
result = pd.DataFrame(data = data_embedded , 
        columns = ['UMAP_0', 'UMAP_1'])


# In[ ]:


result_mat=np.array(result)


# ## Finding clusters in feature-distributed UMAP with DBSCAN

# In[ ]:


alg_cluster =  SpectralClustering(n_clusters=7, affinity= 'nearest_neighbors', n_neighbors=10, random_state=0)
clusters=alg_cluster.fit_predict(result_mat)
(values,counts) = np.unique(clusters,return_counts=True)


# In[ ]:


result['Cluster'] = clusters


# In[ ]:


resultx=result
resultx['Cluster']=clusters+1
sns.lmplot( x="UMAP_0", y="UMAP_1",
  data=result, 
  fit_reg=False, 
  legend=True,
  hue='Cluster', # color by cluster
  scatter_kws={"s": 10},palette="Set1") # specify the point size
plt.savefig(f'{imagePath}/clusters_umap.pdf', bbox_inches='tight')
plt.close()


# - After we visualize the obtained clusters we check their cardinalities
# - In the above plot you can see the cluster numbers

# In[ ]:


plt.bar(values,counts,tick_label=values+1, )
plt.xlabel('Clusters')
plt.ylabel('Number of patients')
plt.title('Distribution of clusters')
plt.savefig(f'{imagePath}/Distribution_of_clusters.pdf', bbox_inches='tight')
plt.close()


# ## Summary statistics for clusters for every feature

# Here we provide feature-wise statistics for each cluster:
# - For discrete variables we provide value-wise frequency distribution
# - For each feature, we visualize feature value frequencies for each cluster
# - For each feature, we provide cluster mean, standard deviation and median for each cluster
# - For each feature, we visualize the cluster means with error bars
# - For discrete features, we visualize the distribution of the features mapped on the feature-distributed final UMAP embedding

# In[ ]:


data['Clusters']=clusters


# In[ ]:


cluster_df_list=[]
for cluster in values:
    cluster_df=data.loc[data['Clusters'] == cluster].drop(columns=['Clusters'])
    cluster_df.columns=list(data.columns)[:-1]
    cluster_df_list.append(cluster_df)


# In[ ]:


data.columns


# In[ ]:


def vizx(feature_list, cluster_df_list, main_data,umap_data,cont_features):
    vizlimit=15
    plt.rcParams["figure.figsize"] = (12,6)
    
    
    ensureDir(f"{imagePath}/vizx")
    
    for featureNr, feature in enumerate(feature_list):
        featureFileName = f"{imagePath}/vizx/{featureNr}_{safeFilename(feature)}"
        featureLog = open(f"{featureFileName}.txt", "w")
        print('Feature name:', feature.upper())
        print('Feature name:', feature.upper(), file=featureLog)
        print('', file=featureLog)
    
        if len(main_data[feature].value_counts())<=vizlimit:
            cluster_counter=1
            for cluster in range(len(cluster_df_list)):
                print('Cluster '+ str(cluster_counter)+ ' frequeny distribution', file=featureLog)
                if feature in list(rev_dict.keys()):
                    feat_keys=rev_dict[feature]
                    r=dict(zip(feat_keys.values(), feat_keys.keys()))
                    print(cluster_df_list[cluster].replace({feature:r})[feature].value_counts(), file=featureLog)
                else:
                    print(cluster_df_list[cluster][feature].value_counts(), file=featureLog)
                cluster_counter=cluster_counter+1
                print('\n', file=featureLog)
        
        print('', file=featureLog)
        print('', file=featureLog)
        
        col=sns.color_palette("Set2")
        
        cluster_bar=[]     
        for cluster in range(len(cluster_df_list)):
            if len(main_data[feature].value_counts())<=vizlimit:
                if feature in list(rev_dict.keys()):
                    y=np.array(cluster_df_list[cluster].replace({feature:r})[feature].value_counts())
                    x=np.array(cluster_df_list[cluster].replace({feature:r})[feature].value_counts().index)
                    cluster_bar.append([x,y])
                else:
                    y=np.array(cluster_df_list[cluster][feature].value_counts().sort_index())
                    x=np.array(cluster_df_list[cluster][feature].value_counts().sort_index().index)
                    cluster_bar.append([x,y])
                
        rows=1
        columns=6
        
        if len(main_data[feature].value_counts())<=vizlimit:
            figx, ax = plt.subplots(rows, columns)
            figx.set_size_inches(25, 5)
            cluster_in_subplot_axis_dict=np.array([0,1,2,3,4,5])
            c=0
            
            for j in range(columns):
                ax[j].bar(cluster_bar[c][0],cluster_bar[c][1],color=col,width=.3)
                ax[j].tick_params(axis='x', which='major', labelsize=8, rotation= 90)
                ax[j].set_title('Cluster: '+str(c+1))
                if c>len(cluster_df_list):
                    break
                else:
                    c=c+1

            plt.savefig(f'{featureFileName}_clusters.pdf', bbox_inches='tight')
            plt.close()
            
        means=[]
        sds=[]
        cluster_labels=[]
        cluster_counter=1
        for cluster in range(len(cluster_df_list)):
            if feature in cont_features:
                print('Cluster '+ str(cluster_counter)+ ' summary statistics', file=featureLog)
                print('\n', file=featureLog)
                cm=cluster_df_list[cluster][feature].mean()
                cs=cluster_df_list[cluster][feature].std()
                print('feature mean:', cm, file=featureLog)
                print('feature standard deviation:', cs, file=featureLog)
                print('feature median:', cluster_df_list[cluster][feature].median(), file=featureLog)
                print('\n', file=featureLog)
                means.append(cm)
                sds.append(cs)
                cluster_labels.append('C'+str(cluster_counter))
            cluster_counter=cluster_counter+1
            
            
        means=np.array(means)
        sds=np.array(sds)
        cluster_labels=np.array(cluster_labels)
        
        
        if feature in cont_features:   
            print('', file=featureLog)  
            print('Distribution of feature across clusters', file=featureLog)
            fig, ax7 = plt.subplots()
            ax7.bar(cluster_labels,means,yerr=sds,color=sns.color_palette("Set3"))
            ax7.tick_params(axis='both', which='major', labelsize=10)
            plt.xlabel(feature, fontsize=15)
            plt.savefig(f'{featureFileName}.pdf', bbox_inches='tight')
            plt.close()
        else:
            print('', file=featureLog)
            print('Feature distribution in UMAP embedding', file=featureLog)
            colors_set = ['lightcoral','cornflowerblue','orange','mediumorchid', 'lightseagreen','olive', 'chocolate','steelblue',"paleturquoise",  "lightgreen",  'burlywood','lightsteelblue']
            customPalette_set = sns.set_palette(sns.color_palette(colors_set))
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
            plt.savefig(f'{featureFileName}.pdf', bbox_inches='tight')
            plt.close()
        
        featureLog.close()
        


# ## Visualize feature distributions on clusters

# In[ ]:


feats_viz=list(set(data.columns)-set(['Clusters']))


# In[ ]:


vizx(feats_viz,cluster_df_list,data,result,cont_features)


# ## Saving patient indexes cluster-wise

# In[ ]:


print()
print("# final_cluser_indexes: ")
final_cluser_indexes=[]
for i in range(len(cluster_df_list)):
    index_list=np.array((cluster_df_list[i].index))
    final_cluser_indexes.append(index_list)
final_cluser_indexes=np.array(final_cluser_indexes, dtype=object)


with open('final_cluser_indexes.npy','wb') as f:
    np.save(f,final_cluser_indexes)


print(np.load('final_cluser_indexes.npy',allow_pickle=True))


# In[ ]:

print()
print("# final_cluser_patient_ids: ")
final_cluser_patient_ids = []
for cluster in cluster_df_list:
    index_list = np.array(cluster.index)
    patient_ids = all_patient_ids[index_list]
    final_cluser_patient_ids.append(patient_ids)

final_cluser_patient_ids = np.array(final_cluser_patient_ids, dtype=object)

with open('final_cluser_patient_id.npy','wb') as f:
    np.save(f, final_cluser_patient_ids)
    
print(np.load('final_cluser_patient_id.npy', allow_pickle=True))

