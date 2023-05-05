#!/usr/bin/env python
# coding: utf-8

# # Step 1: Preprocess data
# ## Load libraries

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
from collections import Counter
from config import *
from tools import *

imagePath = "data/images/preprocessing"
ensureDir(imagePath)

# In[ ]:


from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"


# ## Load Excel-Datasheet

# In[ ]:


filename='Doktorarbeit_Hilke_8.3.xlsx'
data=pd.read_excel(filename)


# In[ ]:


len(np.array(data.columns))


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


# ## Remove not needed rows

# In[ ]:


patients_with_explantation_within_a_year=np.where(-np.array((data['OP- Tag'].apply(lambda x: x.date())-pd.to_datetime(data["Datum Explantation"]).apply(lambda x: x.date())).dt.days)<365)


# In[ ]:


patients_with_death_within_a_year=np.where(-np.array((data['OP- Tag'].apply(lambda x: x.date())-pd.to_datetime(data["Gestorben datum"]).apply(lambda x: x.date())).dt.days)<365)


# In[ ]:


patients_with_explantation_within_a_year


# In[ ]:


patients_with_death_within_a_year


# In[ ]:


data.drop(patients_with_explantation_within_a_year[0], axis=0, inplace=True)
data.shape


# In[ ]:


indicesToDrop = np.where(data["Patient in Tabelle einfügen (1=ja, 0=nein)"] != 1)
data.drop(indicesToDrop[0], axis=0, inplace=True)
data=data.drop(["Patient in Tabelle einfügen (1=ja, 0=nein)"],axis=1)
data.shape


# ## Cleanup index datatype

# In[ ]:


data["Patientennummer"] = data["Patientennummer"].apply(lambda x: int(x))


# ## Cleanup data

# In[ ]:


cleanup_nums = {'Blutgruppe Empfänger.1': {"A ": 4, 'AB':1, 'B': 2, 0:3, 'A':4}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Banff Klassifikation': {' - ':0}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


range_gestorben_array=-np.array((data['OP- Tag'].apply(lambda x: x.date())-pd.to_datetime(data["Gestorben datum"]).apply(lambda x: x.date())).dt.days)


# In[ ]:


data['Range_Gestorben']=range_gestorben_array
data['Range_Gestorben']=data['Range_Gestorben'].fillna(-1)


# In[ ]:


range_dialysis_array=np.array((data['OP- Tag'].apply(lambda x: x.date())-pd.to_datetime(data["Dialyse Anfang "]).apply(lambda x: x.date())).dt.days)


# In[ ]:


data['Range_Dialysis']=range_dialysis_array
data['Range_Dialysis']=data['Range_Dialysis'].fillna(-1)


# In[ ]:


data=data.drop(['Gestorben datum','OP- Tag','Dialyse Anfang ','Lebendspende = x',"Datum Explantation"],axis=1)


# In[ ]:


list(data.columns)


# In[ ]:


cleanup_nums = {'Blutgruppe': {"A ": 4, 'AB':1, 'B': 2, '0':3, 'A':4}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'RH-Faktor (Empfänger)': {"D": 1, 'd':0}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


ICD_indexes_gr5=np.array(data['ICD-10 Code'].value_counts()[data['ICD-10 Code'].value_counts()>=10].index)
ICD_values_gr5=len(ICD_indexes_gr5)+1-np.arange(len(ICD_indexes_gr5))
ICD_indexes_l5=np.array(data['ICD-10 Code'].value_counts()[data['ICD-10 Code'].value_counts()<10].index)
ICD_values_l5=1+np.zeros(len(ICD_indexes_l5))
ICD_indexes=np.concatenate((ICD_indexes_gr5,ICD_indexes_l5))
ICD_values=np.concatenate((ICD_values_gr5,ICD_values_l5))

dict_ICD = {}
for A, B in zip(ICD_indexes, ICD_values):
    dict_ICD[A] = B
    
cleanup_nums = {'ICD-10 Code': dict_ICD}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data=data.drop(['Größe in cm','Gewicht in KG'],axis=1)


# In[ ]:


cleanup_nums = {'Diabetes mellitus': {"nein": 0, 'ja':1, 'nein ':0, 'ja ':1, 'ja (posttransplantation nach Leberspende)':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {
    name: {'nein': 0, 'nein ': 0, ' nein': 0, 'ja':1, 'ja ': 1, '###': np.nan, '?': np.nan}
    for name in
    [ 'Gestorben '
    , 'Blutgruppen inkompabilität'
    , 'Rhesus Inkompabilität'
    , 'Diabetes Folgeschäden '
    , 'arterielle Hypertonie '
    , 'Folgeschäden arterielle Hypertonie'
    , 'KHK '
    , 'COPD/Asthma'
    , 'Herzerkrankungen '
    , 'chronische Lebererkrankung'
    , 'pAVK'
    , 'Maligne Neoplasie '
    , 'Immunologische Erkr'
    , '1. NTx'
    , 'Transfusionen in der Vergangenheit'
    , 'Schwangerschaften'
    , 'Explantation'
    , 'Revisions OP'
    ] }
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data['Demenz '].value_counts()


# In[ ]:


data=data.drop(['Demenz '],axis=1)


# In[ ]:


cleanup_nums = {'Präformierte AK in %': {'###':np.nan,'?':np.nan}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Banff Klassifikation (- means no rejection, other: categories of rejection)': {" - ": 0,"  - ": 0, '###':np.nan,'?':np.nan}}
data.replace(cleanup_nums, inplace=True)
Banff_indexes_gr5=np.array(data['Banff Klassifikation (- means no rejection, other: categories of rejection)'].value_counts()[data['Banff Klassifikation (- means no rejection, other: categories of rejection)'].value_counts()>6].index)
Banff_values_gr5=Banff_indexes_gr5
Banff_indexes_l5=np.array(data['Banff Klassifikation (- means no rejection, other: categories of rejection)'].value_counts()[data['Banff Klassifikation (- means no rejection, other: categories of rejection)'].value_counts()<=6].index)
Banff_values_l5=-1+np.zeros(len(Banff_indexes_l5))
Banff_indexes=np.concatenate((Banff_indexes_gr5,Banff_indexes_l5))
Banff_values=np.concatenate((Banff_values_gr5,Banff_values_l5))

dict_Banff = {}
for A, B in zip(Banff_indexes, Banff_values):
    dict_Banff[A] = B
    
cleanup_nums = {'Banff Klassifikation (- means no rejection, other: categories of rejection)': dict_Banff}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


Grad_indexes_gr5=np.array(data['Gradeinteilung '].value_counts()[data['Gradeinteilung '].value_counts()>=5].index)
Grad_values_gr5=len(Grad_indexes_gr5)-np.arange(len(Grad_indexes_gr5))
Grad_indexes_l5=np.array(data['Gradeinteilung '].value_counts()[data['Gradeinteilung '].value_counts()<5].index)
Grad_values_l5=-1+np.zeros(len(Grad_indexes_l5))
Grad_indexes=np.concatenate((Grad_indexes_gr5,Grad_indexes_l5))
Grad_values=np.concatenate((Grad_values_gr5,Grad_values_l5))

dict_Grad = {}
for A, B in zip(Grad_indexes, Grad_values):
    dict_Grad[A] = B

cleanup_nums = {'Gradeinteilung ': dict_Grad}
data.replace(cleanup_nums, inplace=True)

zero_indexes_Banff=data['Banff Klassifikation (- means no rejection, other: categories of rejection)'].loc[data['Banff Klassifikation (- means no rejection, other: categories of rejection)']==0].index
for i in range(len(data['Gradeinteilung '])):
    if i in zero_indexes_Banff:
           data.at[i,'Gradeinteilung ']=0
            
data['Gradeinteilung '] = data['Gradeinteilung '].fillna(-1)


# In[ ]:


data=data.drop(['Uhrzeit OP Schnitt','Uhrzeit OP Ende'],axis=1)


# In[ ]:


cleanup_nums = {'WIZ': {'###':np.nan}}
data.replace(cleanup_nums, inplace=True)

ind_WIZ=np.array(data['WIZ'].value_counts().index)
val_WIZ=[]
for entry in  ind_WIZ:
    entry_num=int(''.join(filter(str.isdigit, entry)))
    val_WIZ.append(entry_num)
val_WIZ=np.array(val_WIZ)


dict_WIZ = {}
for A, B in zip(ind_WIZ, val_WIZ):
    dict_WIZ[A] = B
    
cleanup_nums = {'WIZ': dict_WIZ}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'kalte Ischämiezeit (Stunden)': {'###':np.nan}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


import re
ind_kalte=np.array(data['kalte Ischämiezeit (Stunden)'].value_counts().index)
val_kalte=[]
for entry in  ind_kalte:
    entry_list=re.findall(r'\d+',entry)
    if len(entry_list)==1:
        entry_list.append(str(0))
    entry_num=int(entry_list[0])*60+int(entry_list[1])
    val_kalte.append(entry_num)
val_kalte=np.array(val_kalte)


dict_kalte = {}
for A, B in zip(ind_kalte, val_kalte):
    dict_kalte[A] = B

cleanup_nums = {'kalte Ischämiezeit (Stunden)': dict_kalte}
data.replace(cleanup_nums, inplace=True)

data=data.rename(columns={'kalte Ischämiezeit (Stunden)': 'kalte Ischämiezeit (Minuen)'})


# In[ ]:


cleanup_nums = {'erste Ausfuhr': {"sofort": 2, 'verzögert':1, ' verzögert':1, 'keine':0, 'nie':0}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Kreatinin bei Entlassung µmol/l': {' - ':0, ' -  ':0, '###':np.nan}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Kreatinin 2 Wochen post OP µmol/l': {' - ':0}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Kreatinin 3 Monate post OP': {' - ':0}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Kreatinin 3 Monate post OP': {' - ':0, '###': np.nan}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Kreatinin 1 Jahr post OP (- dead or lost kidney)': {' - ':0, '###': np.nan,' -':0}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Post Op Dialyse notwendig ': {"nein": 0, 'ja':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Post Op Dialyse weiter ': {"nein": 0, 'ja':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'HWI': {"nein": 0, 'ja':1, 'Ja':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Atemwegsinfektion': {"nein": 0, 'ja':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'CMV': {"nein": 0, 'ja':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Pilzinfektion': {"nein": 0, 'ja':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Viereninfektion': {"nein": 0, 'ja':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Sepsis': {"nein": 0, 'ja':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Magen-Darm ': {"nein": 0, 'ja':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Katheterinfektionen': {"nein": 0, 'ja':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Wunddeheilungsstörungen': {"nein": 0, 'ja':1, '###':np.nan}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Sonstiges': {"nein": 0, 'ja':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Infektionen Anzahl': {'mind 5':5}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data['Rituximab prä OP'] = data['Rituximab prä OP'].fillna(0)
cleanup_nums = {'Rituximab prä OP': {'x':1, 'n':0,' ':0}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data['Immunabsorption prä OP'] = data['Immunabsorption prä OP'].fillna(0)
cleanup_nums = {'Immunabsorption prä OP': {'x':1, 'ja':1, 'ja ':1, ' ':0}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data['Urbanosonstoßtherapie'] = data['Urbanosonstoßtherapie'].fillna(0)
cleanup_nums = {'Urbanosonstoßtherapie': {'x':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data['ATG'] = data['ATG'].fillna(0)
cleanup_nums = {'ATG': {'x':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Tacrolimusspiegel bei Entlassung in ng/ml': {' - ':np.nan,'<2':2}}
data.replace(cleanup_nums, inplace=True)
data['Tacrolimusspiegel bei Entlassung in ng/ml'] = data['Tacrolimusspiegel bei Entlassung in ng/ml'].fillna(0)


# In[ ]:


cleanup_nums = {'Tacrolimusspiegel bei Nachkontrolle in ng/ml': {' - ':np.nan,'###':np.nan}}
data.replace(cleanup_nums, inplace=True)
data['Tacrolimusspiegel bei Nachkontrolle in ng/ml'] = data['Tacrolimusspiegel bei Nachkontrolle in ng/ml'].fillna(0)


# In[ ]:


data['MMF'] = data['MMF'].fillna(0)
cleanup_nums = {'MMF': {'x':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Ciclosporinspiegel bei Entlassung in µg/l': {' - ':np.nan,'###':np.nan}}
data.replace(cleanup_nums, inplace=True)
data['Ciclosporinspiegel bei Entlassung in µg/l'] = data['Ciclosporinspiegel bei Entlassung in µg/l'].fillna(0)


# In[ ]:


cleanup_nums = {'Ciclosporinspiegel bei Nachuntersuchung in µg/l ': {' - ':np.nan,'###':np.nan}}
data.replace(cleanup_nums, inplace=True)
data['Ciclosporinspiegel bei Nachuntersuchung in µg/l '] = data['Ciclosporinspiegel bei Nachuntersuchung in µg/l '].fillna(0)


# In[ ]:


data['Prednisolon'] = data['Prednisolon'].fillna(0)
cleanup_nums = {'Prednisolon': {'x':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data['Basiliximab'] = data['Basiliximab'].fillna(0)
cleanup_nums = {'Basiliximab': {'x':1, ' ':0}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data['Sirolimus'] = data['Sirolimus'].fillna(0)
cleanup_nums = {'Sirolimus': {'x':1, 'x (9,43)':0}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data['Azathioprin'] = data['Azathioprin'].fillna(0)
cleanup_nums = {'Azathioprin': {'x':1, ' ':0}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data['Konversion auf Tacrolimus '] = data['Konversion auf Tacrolimus '].fillna(0)
cleanup_nums = {'Konversion auf Tacrolimus ': {'x':1, ' ':0}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data['Konversion auf CyA '] = data['Konversion auf CyA '].fillna(0)
cleanup_nums = {'Konversion auf CyA ': {'x':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data['Prograf + CellCept'] = data['Prograf + CellCept'].fillna(0)
cleanup_nums = {'Prograf + CellCept': {'x':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data=data.drop(['Prograf + Rapamune'],axis=1)


# In[ ]:


data=data.drop(['Prograf + CyA'],axis=1)


# In[ ]:


data['Prograf + Myofortic'] = data['Prograf + Myofortic'].fillna(0)
cleanup_nums = {'Prograf + Myofortic': {'x':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data['CellCept + CyA'] = data['CellCept + CyA'].fillna(0)
cleanup_nums = {'CellCept + CyA': {'x':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data=data.drop(['CellCept + Sanimmun'],axis=1)


# In[ ]:


data=data.drop(['Cellcept+Myfortic'],axis=1)


# In[ ]:


data['Alter bei Spende (Spender)'] = data['Alter bei Spende (Spender)'].fillna(0)
#cleanup_nums = {'Alter bei Spende (Spender)': {'3Monate': 0.25}}
#data.replace(cleanup_nums, inplace=True)


# In[ ]:


data=data.drop(['Größe in m Spender', 'Gewicht Spender',],axis=1)


# In[ ]:


cleanup_nums = {'Blutgruppe Spender': {"A": 4, 'AB':1, 'B': 2, 0:3, 'B ':2, 'AB ':1}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Blutgruppe Empfänger': {"A": 4, 'AB':1, 'B': 2, 0:3}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'RH-Faktor Spender': {"D": 1, 'd':0}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'CMV Empänger-Spender': {"pos auf pos": 1, 'pos auf neg':2, 'neg auf pos': 3, 'neg auf neg':4,
                                        'neg auf neg ':4, 'neg auf pos ':3, 'pos auf neg ':2, 'pos auf pos ':1,
                                        'neg auuf pos':3}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data['Todspende = x'] = data['Todspende = x'].fillna(1)
cleanup_nums = {'Todspende = x': {'x':0}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Todesursache': {' - ':np.nan,'###':np.nan}}
data.replace(cleanup_nums, inplace=True)

Todesursache_indexes_gr5=np.array(data['Todesursache'].value_counts()[data['Todesursache'].value_counts()>=5].index)
Todesursache_values_gr5=len(Todesursache_indexes_gr5)+1-np.arange(len(Todesursache_indexes_gr5))
Todesursache_indexes_l5=np.array(data['Todesursache'].value_counts()[data['Todesursache'].value_counts()<5].index)
Todesursache_values_l5=1+np.zeros(len(Todesursache_indexes_l5))
Todesursache_indexes=np.concatenate((Todesursache_indexes_gr5,Todesursache_indexes_l5))
Todesursache_values=np.concatenate((Todesursache_values_gr5,Todesursache_values_l5))

dict_Todesursache = {}
for A, B in zip(Todesursache_indexes, Todesursache_values):
    dict_Todesursache[A] = B
    
cleanup_nums = {'Todesursache': dict_Todesursache}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data['Mann auf Frau'] = data['Mann auf Frau'].fillna(0)
cleanup_nums = {'Mann auf Frau': {'x': 1, ' ':np.nan}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data['Frau auf Mann'] = data['Frau auf Mann'].fillna(0)
cleanup_nums = {'Frau auf Mann': {'x': 2, 'X':2}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data['Frau auf Frau '] = data['Frau auf Frau '].fillna(0)
cleanup_nums = {'Frau auf Frau ': {'x': 3}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data['Mann auf Mann'] = data['Mann auf Mann'].fillna(0)
cleanup_nums = {'Mann auf Mann': {'x': 4,' ':np.nan}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data['MaFr1FrMa2FrFr3MaMa4']=data['Mann auf Mann']+data['Frau auf Frau ']+data['Frau auf Mann']+data['Mann auf Frau']


# In[ ]:


cleanup_nums = {'MaFr1FrMa2FrFr3MaMa4': {0: np.nan}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data=data.drop(['Mann auf Frau','Frau auf Mann','Frau auf Frau ','Mann auf Mann'],axis=1)


# In[ ]:


cleanup_nums = {'Ausfuhr bei Entlassung (in ml)': {'1000+': 1000, '###':np.nan, ' - ':np.nan}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


cleanup_nums = {'Kreatinin 3 Monate post OP': {'87.0': 87}}
data.replace(cleanup_nums, inplace=True)


# In[ ]:


data=data.replace('###', np.nan)
data=data.replace(' - ', np.nan)


vizlimit=20
plt.rcParams["figure.figsize"] = (12,6)
for n, feature in enumerate(list(data.dtypes.index)):
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
        plt.savefig(f"{imagePath}/{n}_{safeFilename(feature)}.pdf")
        plt.close()

        if str(data[feature].dtype)=='float64':
            print('feature mean:', data[feature].mean())
            print('feature median:', data[feature].median())
        print('\n')
        print('\n')
        print('\n')
# In[ ]:


dense_data_pool=list(data.isna().sum().index[data.isna().sum()<4])


# In[ ]:


dense_data=data[dense_data_pool]


# In[ ]:


dense_data=dense_data.dropna()


# In[ ]:


dense_data.shape


# In[ ]:


data=data.loc[np.array(dense_data.index)]


# In[ ]:


data.shape


# In[ ]:


print(data.dtypes)


# In[ ]:


data.to_csv('preproceesed_unimputed_data_Hilke.csv',index=True)


# In[ ]:


distance_matrix=[]
for i in range(len(np.array(dense_data))):
    dist=[]
    for j in range(len(np.array(dense_data))):
        d=distance.euclidean(np.array(dense_data)[i],np.array(dense_data)[j])
        dist.append(d)
        neb_list=np.array(dense_data.index)[np.argsort(dist)]
    distance_matrix.append(neb_list)
distance_matrix=np.array(distance_matrix)


# In[ ]:


missing_value_list = [x for x in list(data.columns) if x not in dense_data_pool]


# In[ ]:


total_impute_master=[]
for f in range(len(missing_value_list)):
    missing_value_indices=data[data[missing_value_list[f]].isnull()].index.tolist()
    feature_impute_master=[]
    for index in missing_value_indices:
        index_in_dist_mat=np.where(distance_matrix[:,0]==index)[0][0]
        value_list=[]
        neb_index_counter=1
        while len(value_list)<6:
            neb_index=distance_matrix[index_in_dist_mat][neb_index_counter]
            neb_index_counter=neb_index_counter+1
            impute_value=data.loc[[neb_index]][missing_value_list[f]]
            if float(impute_value)!=float(impute_value):
                pass
            else:
                value_list.append(float(impute_value))
        feature_impute_master.append(np.array(value_list))
    total_impute_master.append(np.array(feature_impute_master))
#total_impute_master=np.array(total_impute_master)


# In[ ]:


total_impute=[]
for i in range(len(total_impute_master)):
    feature_impute=[]
    for j in range(len(total_impute_master[i])):
        intcounter=0
        for k in range(len(total_impute_master[i][j])):
            if total_impute_master[i][j][k]-int(total_impute_master[i][j][k])==0:
                intcounter=intcounter+1
        if intcounter==len(total_impute_master[i][j]):
            imputed_value=Counter(total_impute_master[i][j]).most_common(1)[0][0]
        else:
            imputed_value=np.mean(total_impute_master[i][j])
        feature_impute.append(np.array(imputed_value))
    total_impute.append(np.array(feature_impute))
    


# In[ ]:


for f in range(len(missing_value_list)):
    missing_value_indices=data[data[missing_value_list[f]].isnull()].index.tolist()
    for i in range(len(missing_value_indices)):
        data.at[missing_value_indices[i], missing_value_list[f]]=total_impute[f][i]


# In[ ]:


data.shape


# In[ ]:


from scipy.stats import entropy
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


data_Konversion_Tacro=data['Konversion auf Tacrolimus ']
data_Konversion_CyA=data['Konversion auf CyA ']

data=data.drop(np.array(nom_features+ord_features)[np.where(cat_feature_entropies<np.quantile(cat_feature_entropies,.2))],axis=1)##removing low entropy categorical featuresdata=data.drop(np.array(cont_features)[np.where(cont_feature_spread<np.quantile(cont_feature_spread,.25))],axis=1)##removing low spread continuous features
# In[ ]:


data['Konversion auf Tacrolimus ']=data_Konversion_Tacro
data['Konversion auf CyA ']=data_Konversion_CyA


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


data.head(10)


# In[ ]:


data.to_csv('preproceesed_imputed_data_Hilke.csv',index=True)


# In[ ]:


data.to_csv('preproceesed_data_Hilke.csv',index=True)

