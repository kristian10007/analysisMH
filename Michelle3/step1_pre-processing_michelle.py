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
import seaborn as sn
import random
from scipy import ndarray
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.stats import entropy
from config import *
from tools import *
from revdict import *


# In[ ]:


from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


ensureDir("data")


# In[ ]:


filename='Tabelle_mit_Patientennummer_K.xlsx'
data=pd.read_excel(filename)


# In[ ]:


len(np.array(data.columns))


# In[ ]:


list(data.columns)


# In[ ]:


def set_pandas_display_options() -> None:
    """Set pandas display options."""
    # Ref: https://stackoverflow.com/a/52432757/
    display = pd.options.display

    display.max_columns = 1000
    display.max_rows = 1000
    display.max_colwidth = 199
    display.width = 1000
    # display.precision = 2  # set as needed

set_pandas_display_options()

print(data.dtypes)


# ## Remove not needed rows and columns

# In[ ]:


data.shape


# In[ ]:


indicesToDrop = np.where(data["Patient in Tabelle einfügen (1=ja, 0=nein)"] != 1)
data.drop(indicesToDrop[0], axis=0, inplace=True)
data=data.drop(["Patient in Tabelle einfügen (1=ja, 0=nein)"],axis=1)


# In[ ]:


data = data.drop(columns_to_remove, axis=1)


# In[ ]:


data.shape


# In[ ]:


print(data.dtypes)


# ## Cleanup data

# In[ ]:


data.rename(columns=rename_columns, inplace=True)


# In[ ]:


data.replace(data_mapping, inplace=True)
addToRevDict(data_mapping)


# In[ ]:


for n in data_mapping_bool_x_is_yes:
    data[n] = data[n].fillna(0)
    data.replace({n: {'x': 1, 'X': 1}}, inplace=True)
    addToRevDict({n: {'ja': 1, 'nein': 0}})


# In[ ]:


for n in data_mapping_bool_yes_no:
    data[n] = data[n].fillna(np.nan)
    data.replace({n: map_yes_no}, inplace=True)
    addToRevDict({n: {'ja': 1, 'nein': 0}})


# In[ ]:


for n in data_mapping_x_is_nan:
    data[n] = data[n].fillna(np.nan)
    data.replace({n: map_x_NaN}, inplace=True)


# In[ ]:


for name in data_mapping_auto_count:
    data[name] = data[name].fillna(0)
    mapping = {'x': 0, 'X': 0}
    n = 0
    for x in set(data[name]):
        if type(x) == type("abc") and n not in mapping.keys():
            n = n + 1
            mapping[x] = n

    data.replace({name: mapping}, inplace=True)
    addToRevDict({name: mapping})    


# In[ ]:


data.replace({
    c_Geburten: {'x': np.nan, 'ja (keine genauen angaben)': np.nan}
    , c_po_Tag_dialysefrei: {n: np.nan for n in ['x',  'weiter', 'immer noch', 'ja', '11?', '?', 'mehrere', 'nein']}
    , c_Diabetes_mellitus: {'ja, posttransplantionsdiabetes': 1}
    }, inplace=True)


# In[ ]:


data[c_Range_Gestorben] = -np.array(
    (data['OP-Tag'].apply(lambda x: x.date())
     - pd.to_datetime(data["gestorben"]).apply(lambda x: x.date())
    ).dt.days)
data[c_Range_Gestorben] = data[c_Range_Gestorben].fillna(-1)


# In[ ]:


def fixDate(x):
    if type(x) == type(7):
        x = pd.to_datetime(f"{x}-01-01")
    return x

data[c_Datum_Explantation] = data[c_Datum_Explantation].apply(fixDate)

range_explantation_array = -np.array(
        (data[c_OP_Tag].apply(lambda x: x.date())
         - pd.to_datetime(data[c_Datum_Explantation]).apply(lambda x: x.date())
        ).dt.days)

data[c_Range_Explantation] = range_explantation_array

patients_with_explantation_within_a_year = np.where(range_explantation_array<365)

data.drop(patients_with_explantation_within_a_year[0], axis=0, inplace=True)

data[c_Range_Explantation] = data[c_Range_Explantation].fillna(-1)

data = data.drop([c_OP_Tag, c_Datum_Explantation],axis=1)


# In[ ]:


reorderByCount(data, c_Grund_fuer_TX, splitBy=5)


# In[ ]:


cleanup_dict = {'x': np.nan}
for x in data[c_Dauer_in_Minuten]:
    if type(x) != type(7) and type(x) != type(7.3) and x != 'x':
        cleanup_dict[x] = (60.0 * x.hour) + (1.0 * x.minute)

data.replace({c_Dauer_in_Minuten: cleanup_dict}, inplace=True)


# In[ ]:


data[c_Rh_Compatibility] = 1 * (np.array(data[c_Rh_Empfaenger]) + np.array(data[c_Rh_Spender]) != 1)


# In[ ]:


reorderByCount(data, c_Todesursache, splitBy=5)


# In[ ]:


data['Mann auf Frau'] = data['Mann auf Frau'].fillna(0)
data['Frau auf Mann'] = data['Frau auf Mann'].fillna(0)
data['Frau auf Frau'] = data['Frau auf Frau'].fillna(0)
data['Mann auf Mann'] = data['Mann auf Mann'].fillna(0)

cleanup_nums = { 'Mann auf Frau': {'x': 1, ' ':0}
               , 'Frau auf Mann': {'x': 2, 'X':2}
               , 'Frau auf Frau': {'x': 3}
               , 'Mann auf Mann': {'x': 4}
               }
data.replace(cleanup_nums, inplace=True)

data['MaFr1FrMa2FrFr3MaMa4'] = data['Mann auf Mann'] + data['Frau auf Frau'] + data['Frau auf Mann'] + data['Mann auf Frau']
data=data.drop(['Mann auf Frau', 'Frau auf Mann', 'Frau auf Frau', 'Mann auf Mann'],axis=1)


# In[ ]:


splitByCount(data, c_Banff, splitBy=5, fillNaValueBefore=0)
set(data[c_Banff])


# In[ ]:


reorderByCount(data, c_Grad, splitBy=5, defaultValue=-1)

zero_indexes_Banff = data[c_Grad].loc[data[c_Grad]==0].index
for i in range(len(data[c_Grad])):
    if i in zero_indexes_Banff:
           data.at[i,c_Grad] = 0
            
data[c_Grad] = data[c_Grad].fillna(-1)

set(data[c_Grad])


# In[ ]:


cleanup_nums = {}

data[c_Malignom_nach_OP] = data[c_Malignom_nach_OP].fillna(0)
mapping = { 'x': 1, 'X': 1, 'kein': 0, 'keine': 0, 'kein ': 0, 'keine ': 0 }
for n in set(data[c_Malignom_nach_OP]):
    if n not in mapping.keys():
        mapping[n] = 1
        
cleanup_nums[c_Malignom_nach_OP] = mapping

mapping = {}
for n in set(data[c_post_OP_Dialysen]):
    if type(n) != type(7):
        mapping[n] = np.nan

if len(mapping.keys()) > 0:
    cleanup_nums[c_post_OP_Dialysen] = mapping

data.replace(cleanup_nums, inplace=True)
addToRevDict({c_Malignom_nach_OP: {'ja': 1, 'nein': 0}})


# In[ ]:


data[c_TC_switch] = data[c_T_C] + data[c_C_T]
data = data.drop([c_T_C, c_C_T],axis=1)


# In[ ]:


cleanup_nums = {c_Alter_bei_Spende: {"3 Monate": .25}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


for i in data[c_Ausfuhr_bei_Entlassung].index:
    if type(data[c_Ausfuhr_bei_Entlassung][i]) != int:
        data.at[i,c_Ausfuhr_bei_Entlassung] = np.nan


# In[ ]:


data[c_Tacrolimus_Spiegel_bei_Entlassung] = data[c_Tacrolimus_Spiegel_bei_Entlassung].fillna(0)


# In[ ]:


data[c_Cyc_Spiegel_bei_Entlassung] = data[c_Cyc_Spiegel_bei_Entlassung].fillna(0)
cleanup_nums = {c_Cyc_Spiegel_bei_Entlassung: {"###": np.nan, 'x': np.nan}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data=data.replace('###', np.nan)
data=data.replace(' - ', np.nan)


# In[ ]:


data=data.drop([ 'Demenz'
               , 'Lebendspende'
               , 'gestorben'
               , 'Rh Spender'
               , 'Rh Empfänger'
               ],axis=1)

vizlimit=20
plt.rcParams["figure.figsize"] = (12,6)
for feature in list(data.dtypes.index):
    if len(data[feature].value_counts())<=vizlimit:
        print('Feature name:', feature.upper())
        print('\n')
        print(data[feature].value_counts())
        col=sn.color_palette("Set2")
        ax=data[feature].value_counts().plot.bar(color=col)
        ax.tick_params(axis='both', which='major', labelsize=20)
        if len(ax.get_xticklabels())>vizlimit:
            plt.setp(ax.get_xticklabels(), visible=False)
        plt.xlabel(feature, fontsize=25)
        plt.show()

        if str(data[feature].dtype)=='float64':
            print('feature mean:', data[feature].mean())
            print('feature median:', data[feature].median())
        print('\n')
        print('\n')
        print('\n')
# In[ ]:


dense_data_pool = list(data.isna().sum().index[data.isna().sum()<4])
dense_data = data[dense_data_pool]
dense_data = dense_data.dropna()


# In[ ]:


dense_data.shape


# In[ ]:


data = data.loc[np.array(dense_data.index)]


# In[ ]:


data.shape


# In[ ]:


for x in data.dtypes.index:
    if str(data.dtypes[x]) not in ['float64', 'int64']:
        print(f"{x}: {str(data.dtypes[x])}: " + str([y for y in set(data[x]) if str(type(y)) != "<class 'datetime.time'>"]))
        print()


# In[ ]:


distance_matrix=[]
np_dense_data = np.array(dense_data)

for i in range(len(np_dense_data)):
    dist=[]
    for j in range(len(np_dense_data)):
        d = distance.euclidean(np_dense_data[i], np_dense_data[j])
        dist.append(d)
    
    neb_list = np.array(dense_data.index)[np.argsort(dist)]
    distance_matrix.append(neb_list)

distance_matrix = np.array(distance_matrix)


# In[ ]:


missing_value_list = [x for x in list(data.columns) if x not in dense_data_pool]


# In[ ]:


missing_value_list


# In[ ]:


total_impute_master=[]
for feature_name in missing_value_list:
    print(f"{feature_name} ...")
    missing_value_indices = data[data[feature_name].isnull()].index.tolist()
    feature_impute_master = []
    for index in missing_value_indices:
        index_in_dist_mat = np.where(distance_matrix[:,0] == index)[0][0]
        value_list = []
        for neb_index in distance_matrix[index_in_dist_mat][1:]:
            impute_value = data.loc[[neb_index]][feature_name]
            try:
                _v = float(impute_value) 
                if float(impute_value) != float(impute_value):
                    pass
                else:
                    value_list.append(float(impute_value))
            except  TypeError:
                pass
            finally:
                pass
            
            if len(value_list) >= 6:
                break
                
        feature_impute_master.append(np.array(value_list))
    total_impute_master.append(np.array(feature_impute_master))
print("done.")


# In[ ]:


total_impute = []
for tim_feature in total_impute_master:
    feature_impute=[]
    for tim_data in tim_feature:
        intcounter=0
        
        if len(tim_data) == 0:
            imputed_value = np.nan
        else:
            all_ints = True
            for v in tim_data:
                if v - int(v) != 0:
                    all_ints = False
                    break
        
            if all_ints:
                imputed_value = np.bincount(tim_data.astype(int)).argmax()
            else:
                imputed_value = np.mean(tim_data)
            
        feature_impute.append(np.array(imputed_value))

    total_impute.append(np.array(feature_impute))


# In[ ]:


for f, values in zip(missing_value_list, total_impute):
    missing_value_indices = data[data[f].isnull()].index.tolist()
    for i, v in zip(missing_value_indices, values):
        data.at[i, f] = v


# In[ ]:


data.shape


# In[ ]:


len(data.columns)


# In[ ]:


len(cont_features+nom_features+ord_features)


# In[ ]:


def bmi(groesse, gewicht):
    return data[gewicht] / ((data[groesse] / 100) * (data[groesse] / 100))

data[c_BMI_Empfaenger] = bmi(c_Gewicht_Empfaener, c_Groesse_Empfaener)
data[c_BMI_Spender] = bmi(c_Gewicht_Spender, c_Groesse_Spender)

#data=data.drop([c_Gewicht_Empfaener, c_Groesse_Empfaener, c_Gewicht_Spender, c_Groesse_Spender, c_Todspende], axis=1)


# In[ ]:


cat_feature_entropies=[]
for feature in nom_features+ord_features:
    x=entropy(data[feature].value_counts(), base=2)/(len(data[feature].value_counts().index)-1)
    cat_feature_entropies.append(x)
    print(feature+' '+str(x))
cat_feature_entropies=np.array(cat_feature_entropies)


# In[ ]:


cont_feature_spread=[]
for feature in cont_features:
    standardardized=(data[feature]-data[feature].mean())/data[feature].std()
    x=abs(standardardized.max()-standardardized.min())
    cont_feature_spread.append(x)
    print(feature+'  '+str(x))
cont_feature_spread=np.array(cont_feature_spread)


# In[ ]:


columns_to_save = [ c_Cell_Myfortic
                  , c_Konversion_auf_andere
                   , c_Tacrolimus_Spiegel_bei_Entlassung
                   , c_Abstossungsreaktion
                   , c_Alter_bei_Tx
                   , c_Range_Explantation
                   , c_Range_Gestorben
                   , c_Groesse_Spender
                  ]
column_backup = {}

for c in columns_to_save:
    column_backup[c] = data[c]

# removing low entropy categorical features
data=data.drop(np.array(nom_features+ord_features)[np.where(cat_feature_entropies<np.quantile(cat_feature_entropies,.25))],axis=1)

##removing low spread continuous features
data=data.drop(np.array(cont_features)[np.where(cont_feature_spread<np.quantile(cont_feature_spread,.25))],axis=1)

for c in columns_to_save:
    data[c] = column_backup[c]


# In[ ]:


data.shape


# In[ ]:


list(data.columns)


# In[ ]:


data.head(10)


# In[ ]:


data.to_csv(fileName_csv_preprocessed, index=True)


# In[ ]:


all_labeled_columns = [c_Patientennummer] + cont_features + nom_features + ord_features

print("Colums with unknown data type:")
for x in data.columns:
    if x not in all_labeled_columns:
        print(f"'{x}'")
print()
print()

print("Colums not in dataset:")
for x in all_labeled_columns:
    if x not in data.columns:
        print(f"'{x}'")


# In[ ]:


saveAutoRevDict()

