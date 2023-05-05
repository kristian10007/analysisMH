#!/usr/bin/env python
# coding: utf-8

# # Step 2: Classification
# ## Load libraries

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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from tools import *

imagePath = "data/images/classification"
ensureDir(imagePath)




# ## Load data

# In[ ]:


data=pd.read_csv('preproceesed_data_Hilke.csv',index_col=0)


# In[ ]:


data.shape


# In[ ]:


data=data.drop(["Patientennummer"],axis=1)
data.shape


# In[ ]:


labels=['Banff Klassifikation (- means no rejection, other: categories of rejection)']


# In[ ]:


df=data.drop(labels,axis=1)


# In[ ]:


X=np.array(df)


# ## Analysis for switch between TC and Cyc

# In[ ]:


y_Banff=data['Banff Klassifikation (- means no rejection, other: categories of rejection)']


# In[ ]:


y_Banff_discretize=[]
for i in y_Banff:
    if i==0:
        y_Banff_discretize.append(0)
    else:
        y_Banff_discretize.append(1)
y_Banff_discretize=np.array(y_Banff_discretize)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y_Banff_discretize, test_size=0.30, random_state=42)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)
# In[ ]:


model = GradientBoostingClassifier()
parameters = {"learning_rate": sp_randFloat(),
                  "subsample"    : sp_randFloat(),
                  "n_estimators" : sp_randInt(100, 1000),
                  "max_depth"    : sp_randInt(4, 10)
                 }

randm = RandomizedSearchCV(estimator=model, param_distributions = parameters,
                               cv = 5, n_iter = 25, n_jobs=-1, random_state=42)
randm.fit(X_train, y_train)

print(" Results from Random Search " )
print("The best estimator across ALL searched params:", randm.best_estimator_)
print("The best score across ALL searched params:", randm.best_score_)
print("The best parameters across ALL searched params:", randm.best_params_)

# In[ ]:


model=GradientBoostingClassifier(learning_rate=0.07455064367977082, max_depth=4,
                           n_estimators=443, subsample=0.3948815181755697, random_state=15)
model.fit(X_train,y_train)


# In[ ]:


y_pred=model.predict(X_test)


# In[ ]:


y_pred_proba=model.predict_proba(X_test)
y_pred_proba=y_pred_proba[:,1]


# In[ ]:


print(confusion_matrix(y_test, y_pred))
print('F1-Score for classification:',f1_score(y_test, y_pred))


# In[ ]:


np.random.seed(20)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
area=auc(recall, precision)
plt.plot(recall, precision)
plt.title('Area under precision recall curve='+str(area*100)[:5]+'% for TC-Cyc switch classification')
plt.xlabel('recall', fontsize=12) 
plt.ylabel('precision', fontsize=12) 
plt.ylim(-0.05, 1.05)
plt.xlim(-0.05, 1.05)
plt.savefig(f'{imagePath}/Area under precision recall curve.pdf', bbox_inches='tight')
plt.close()

# axis labels


# In[ ]:


k=10 #select top k features
top_features=df.columns[model.feature_importances_.argsort()[-k:][::-1]]
top_feature_importance_scores=model.feature_importances_[model.feature_importances_.argsort()[-k:][::-1]]


# In[ ]:


fig, ax7 = plt.subplots()
ax7.bar(top_features,top_feature_importance_scores,color=sns.color_palette("Set3"))
ax7.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('Top features as per Gradiant Boosting Classifier for TC-Cyc switch classification', fontsize=12)   
plt.xticks(rotation=90)
plt.savefig(f'{imagePath}/Top features as per Gradiant Boosting Classifier for TC-Cyc switch classification.pdf', bbox_inches='tight')
plt.close()

