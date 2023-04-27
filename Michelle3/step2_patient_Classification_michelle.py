#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import math
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
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)
from config import *
from tools import *


md = MD("data/step2.md")

md.print("\\newpage")
md.heading("Classification")

global imageCount 
imageCount = 0

def saveImage(imageFile):
    global imageCount 
    plt.savefig(imageFile)
    plt.close()
    md.image(imageFile, 'Graphic')
    if imageCount < 4:
        imageCount += 1
    else:
        imageCount = 0
        md.print("\\newpage")

# In[ ]:


import warnings
warnings.filterwarnings("ignore", "X has feature names, but KNeighborsClassifier was fitted without feature names")


# In[ ]:


imagesPath = "data/images/Classification"
ensureDir("data/images")
ensureDir(imagesPath)
ensureDir(imagesPath + "/pdp")
ensureDir(imagesPath + "/pdp_TC")
ensureDir(imagesPath + "/pdp_Cyc")


# In[ ]:


#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# In[ ]:


data=pd.read_csv(fileName_csv_preprocessed, index_col=0)


# In[ ]:


data.shape


# In[ ]:


drops = [ c_Patientennummer
        , c_1_Jahr_post_OP
        , c_2_Wochen_post_OP
        , c_3_Monate_post_OP
        , c_Abstossungsreaktion
        , c_Atemwegsinfekt
        , c_ATG
        , c_Ausfuhr_bei_Entlassung
        , c_Azathioprin
        , c_Banff
        , c_Basiliximab_Simulect
        , c_Biospie
        , c_Cell_Myfortic
        , c_Cellcept_CyA
        , c_CMV
        , c_Cyc_Spiegel_bei_Entlassung
        , c_Cyc_Spiegel_bei_Nachsorge
        , c_dialysefrei
        , c_erste_Ausfuhr_am
        , c_Everolimus
        , c_Explantation
        #, 'GFR_bei_Entlassung '
        #, 'GFR_2Wochen_postOP'
        #, 'GFR_3_Monate_post_OP'
        #, 'GFR_1_Jahr_post_OP'
        , c_Grad
        , c_HWI
        , c_Immunadsoprtionstherapie
        #, 'Infektionen'
        , c_Katheterinfektion
        , c_Kreatinin_bei_Entlassung
        , c_Konversion_auf_andere
        , c_Malignom_nach_OP
        , c_MDT
        , c_MMF
        , c_Pilzinfektion
        , c_post_OP_Dialysen
        , c_Prednisolon
        , c_Prograf_Cellcept
        , c_Prograf_Myfortic
        , c_Range_Explantation
        , c_Range_Gestorben
        , c_Revisions_OP
        #, '(High risk drug)  Rituximab/ Immunglobuline'
        , c_Sepsis
        , c_Sirolimus
        , c_Sonstige
        , c_Tacrolimus_Spiegel_bei_Entlassung
        , c_Tacrolimus_Spiegel_Nachkontrolle
        , c_TC_switch
        , c_Urbanstosstherapie
        , c_Virusinfektionen
        , c_Wundheilungsstoerung
        ]


# In[ ]:


df=data.drop(drops,axis=1)


# In[ ]:


for c in data.columns:
    n = []
    k = 0
    for x in data[c]:
        if x == np.nan or math.isnan(x):
            n.append(k)
        k += 1
        
    if n != []:
        print(f"'{c}': {n}'")


# In[ ]:


X=np.array(df)


md.heading("Analysis for switch between TC and Cyc", 2)

# In[ ]:


y_TC_switch=np.array(data[c_TC_switch])


# In[ ]:


np.unique(y_TC_switch, return_counts=True)


# In[ ]:


y_TC_switch_discretize=[]
for i in y_TC_switch:
    if i==0:
        y_TC_switch_discretize.append(0)
    else:
        y_TC_switch_discretize.append(1)
y_TC_switch_discretize=np.array(y_TC_switch_discretize)


# In[ ]:


np.unique(y_TC_switch_discretize, return_counts=True)


# In[ ]:


pca = PCA()
Xt = pca.fit_transform(X)
plot = plt.scatter(Xt[:,0], Xt[:,1], c=y_TC_switch_discretize, s=4)
saveImage(imagesPath + "/pca_y_TC_switch_discretize.pdf")

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y_TC_switch_discretize, test_size=0.30, random_state=42)


# In[ ]:


print(X_train.shape)


# In[ ]:


def score_model(model, params, cv=None):
    """
    Creates folds manually, and upsamples within each fold.
    Returns an array of validation (recall) scores
    """
    if cv is None:
        cv = KFold(n_splits=5, random_state=42)

    # smoter = SMOTE(random_state=42, k_neighbors=30)
    smoter = SMOTE(random_state=42, k_neighbors=25) # n_samples was smaler than k_keighbors
    
    scores = []

    for train_fold_index, val_fold_index in cv.split(X_train, y_train):
        # Get the training data
        X_train_fold, y_train_fold = X_train[train_fold_index], y_train[train_fold_index]
        # Get the validation data
        X_val_fold, y_val_fold = X_train[val_fold_index], y_train[val_fold_index]

        # Upsample only the data in the training section
        X_train_fold_upsample, y_train_fold_upsample = smoter.fit_resample(X_train_fold,
                                                                           y_train_fold)
        # Fit the model on the upsampled training data
        model_obj = model(**params).fit(X_train_fold_upsample, y_train_fold_upsample)
        # Score the model on the (non-upsampled) validation data
        score = geometric_mean_score(y_val_fold, model_obj.predict(X_val_fold))
        scores.append(score)
    return np.array(scores)


# In[ ]:


kf = KFold(n_splits=5, random_state=42, shuffle=True)


# In[ ]:


example_params = { }


md.print()
md.print("Trying cross validation with gradient boosting")

# In[ ]:


CV_scores_GB=score_model(GradientBoostingClassifier, example_params, cv=kf)


# In[ ]:


md.code(CV_scores_GB)
md.code(CV_scores_GB.std())


md.print()
md.print("We observe incosistent results across the five folds, thus model might be overfitting")
md.print('Cross Validated geometric mean: ')
md.code(CV_scores_GB.mean())

md.heading("trying cross validation with k-Nearest Neighbours, a simpler model", 2)

CV_scores_kNN = score_model(KNeighborsClassifier, example_params, cv=kf)

md.code(CV_scores_kNN)
md.code(CV_scores_kNN.std())


md.print("We observe more cosistent results across the five folds, thus kNN would be more reliable than GB")

md.print()
md.print('Cross Validated geometric mean:')
md.code(CV_scores_kNN.mean())


md.print()
md.print("We therefore validate on the independent set with the kNN model")

# In[ ]:


sm = SMOTE(random_state=42,k_neighbors=40)
X_res, y_res = sm.fit_resample(X_train, y_train)


# In[ ]:


model=KNeighborsClassifier()
model.fit(X_res,y_res)


# In[ ]:


y_pred=model.predict(X_test)


# In[ ]:


y_pred_proba=model.predict_proba(X_test)
y_pred_proba=y_pred_proba[:,1]


# In[ ]:


md.code(confusion_matrix(y_test, y_pred))
md.print('G-Mean for classification:')
md.code(geometric_mean_score(y_test, y_pred))


# In[ ]:


np.random.seed(20)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
area=auc(recall, precision)
plt.plot(recall, precision)
plt.title('Area under precision recall curve='+str(area*100)[:5]+'% for TC-Cyc switch classification')
plt.xlabel('recall', fontsize=12) 
plt.ylabel('precision', fontsize=12) 
# axis labels
saveImage(f"{imagesPath}/precision_recall_curve.pdf")



md.print("\\newpage")
md.print()
md.print("Understand Partial Dependence Plots (PDPs) from the following video:")
md.print("<https://www.youtube.com/watch?v=uQQa3wQgG_s&ab_channel=ritvikmath>")

# In[ ]:


df_test=pd.DataFrame(data = X_test, 
                  columns = df.columns)


# In[ ]:

n = 0
for var in df.columns:
    p=partial_dependence(model, df_test, [var])
    sns.lineplot(x=p['values'][0], y=p['average'][0]/max(abs(p['average'][0])), style=0, markers=True, legend=False)
    plt.ylim(-1.02,1.02)
    plt.ylabel("Partial dependence for feature "+var)
    plt.xlabel(var)
    imageFile = f"{imagesPath}/pdp/{safeFilename(var)}.pdf"
    saveImage(imageFile)
    n += 1
    if n > 10:
        md.print("\\newpage")
        n = 0


md.print("\\newpage")
md.heading("Analysis for implementation of TC", 2)

# In[ ]:


y_Tacrolimus=data['Tacrolimus Spiegel bei Entlassung (ng/ml)']


# In[ ]:


y_Tacrolimus_discretize=[]
for i in y_Tacrolimus:
    if i==0:
        y_Tacrolimus_discretize.append(0)
    else:
        y_Tacrolimus_discretize.append(1)
y_Tacrolimus_discretize=np.array(y_Tacrolimus_discretize)


# In[ ]:


np.unique(y_Tacrolimus_discretize, return_counts=True)


# In[ ]:


pca = PCA()
Xt = pca.fit_transform(X)
plot = plt.scatter(Xt[:,0], Xt[:,1], c=y_Tacrolimus_discretize, s=4)
imageFile = imagesPath + "/pca_TC.pdf"
saveImage(imageFile)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y_Tacrolimus_discretize, test_size=0.30, random_state=42)


# In[ ]:


cross_val_GB=cross_val_score(GradientBoostingClassifier(), X_train, y_train, cv=5)


# In[ ]:


md.code(cross_val_GB)
md.code(cross_val_GB.mean())
md.code(cross_val_GB.std())


# In[ ]:


model=GradientBoostingClassifier()
model.fit(X_train,y_train)


# In[ ]:


y_pred=model.predict(X_test)


# In[ ]:


y_pred_proba=model.predict_proba(X_test)
y_pred_proba=y_pred_proba[:,1]


# In[ ]:


md.code(confusion_matrix(y_test, y_pred))
md.print('G-Mean for classification:')
md.code(geometric_mean_score(y_test, y_pred))


# In[ ]:


np.random.seed(20)
fpr, tpr, thresholds = roc_curve(y_test,y_pred_proba)
area=roc_auc_score(y_test,y_pred_proba)
plt.plot(fpr, tpr)
plt.title('Area under ROC curve='+str(area*100)[:5]+'% for classification of patients under TC')
plt.xlabel('false positive rate', fontsize=12) 
plt.ylabel('true positive rate', fontsize=12) 
# axis labels
imageFile = imagesPath + "/roc_curve_TC.pdf"
saveImage(imageFile)


# In[ ]:


k=10 #select top k features
top_features=df.columns[model.feature_importances_.argsort()[-k:][::-1]]
top_feature_importance_scores=model.feature_importances_[model.feature_importances_.argsort()[-k:][::-1]]


# In[ ]:


fig, ax7 = plt.subplots()
ax7.bar(top_features,top_feature_importance_scores,color=sns.color_palette("Set3"))
ax7.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('Top features as per Gradiant Boosting Classifier for classification of patients under TC', fontsize=12)   
plt.xticks(rotation=90)
imageFile = imagesPath + "/Top_features_as_per_Gradiant_Boosting_Classifier_for_classification_of_patients_under_TC.pdf"
plt.savefig(imageFile)
plt.close()
md.image(imageFile, imageFile)


# In[ ]:


df_test=pd.DataFrame(data = X_test, columns = df.columns)


# In[ ]:


for var in df.columns:
    p=partial_dependence(model, df_test, [var])
    sns.lineplot(x=p['values'][0], y=p['average'][0]/max(abs(p['average'][0])), style=0, 
                 markers=True, legend=False)
    plt.ylim(-1.02,1.02)
    plt.ylabel("Partial dependence for feature "+var)
    plt.xlabel(var)
    saveImage(f"{imagesPath}/pdp_TC/{safeFilename(var)}.pdf")



md.print("\\newpage")
md.heading("Analysis for implementation of Cyc", 2)

# In[ ]:


y_Cyc=data['Cyc Spiegel bei Entlassung (yg/l)']


# In[ ]:


y_Cyc_discretize=[]
for i in y_Cyc:
    if i==0:
        y_Cyc_discretize.append(0)
    else:
        y_Cyc_discretize.append(1)
y_Cyc_discretize=np.array(y_Cyc_discretize)


# In[ ]:


np.unique(y_Cyc_discretize, return_counts=True)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y_Cyc_discretize, test_size=0.30, random_state=42)


# In[ ]:


cross_val_GB=cross_val_score(GradientBoostingClassifier(), X_train, y_train, cv=5)


# In[ ]:


md.code(cross_val_GB)
md.code(cross_val_GB.std())
md.code(cross_val_GB.mean())


# In[ ]:


model=GradientBoostingClassifier(random_state=15)
model.fit(X_train,y_train)


# In[ ]:


y_pred=model.predict(X_test)


# In[ ]:


y_pred_proba=model.predict_proba(X_test)
y_pred_proba=y_pred_proba[:,1]


# In[ ]:


md.code(confusion_matrix(y_test, y_pred))
md.print('G-Mean for classification:')
md.code(geometric_mean_score(y_test, y_pred))


# In[ ]:


np.random.seed(20)
fpr, tpr, thresholds = roc_curve(y_test,y_pred_proba)
area=roc_auc_score(y_test,y_pred_proba)
plt.plot(fpr, tpr)
plt.title('Area under ROC curve='+str(area*100)[:5]+'% for classification of patients under Cyc')
plt.xlabel('false positive rate', fontsize=12) 
plt.ylabel('true positive rate', fontsize=12)
saveImage(imagesPath + "/roc_Cyc.pdf")


# In[ ]:


k=10 #select top k features
top_features=df.columns[model.feature_importances_.argsort()[-k:][::-1]]
top_feature_importance_scores=model.feature_importances_[model.feature_importances_.argsort()[-k:][::-1]]


# In[ ]:


fig, ax7 = plt.subplots()
ax7.bar(top_features,top_feature_importance_scores,color=sns.color_palette("Set3"))
ax7.tick_params(axis='both', which='major', labelsize=10)
plt.xlabel('Top features as per Gradiant Boosting Classifier for classification of patients under Cyc', fontsize=12)   
plt.xticks(rotation=90)
saveImage(imagesPath + "/Top_features_as_per_Gradiant_Boosting_Classifier_for_classification_of_patients_under_Cyc.pdf")



# In[ ]:


df_test=pd.DataFrame(data = X_test, 
                  columns = df.columns)


# In[ ]:


for var in df.columns:
    p=partial_dependence(model, df_test, [var])
    sns.lineplot(x=p['values'][0], y=p['average'][0]/max(abs(p['average'][0])), style=0, 
                 markers=True, legend=False)
    plt.ylim(-1.02,1.02)
    plt.ylabel("Partial dependence for feature "+var)
    plt.xlabel(var)
    saveImage(f"{imagesPath}/pdp_Cyc/{safeFilename(var)}.pdf")
    


md.print("\\newpage")
# In[ ]:





# In[ ]:


md.close()

