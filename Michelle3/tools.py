import os
import os.path
import numpy as np

from revdict import addToRevDict

def splitByCount(data, name, splitBy=5, fillNaValueBefore=None, fillNaValueAfter=None):
    if fillNaValueBefore is not None:
        data[name] = data[name].fillna(fillNaValueBefore)

    indexes_gr5 = np.array(data[name].value_counts()[data[name].value_counts() >= splitBy].index)
    values_gr5 = indexes_gr5
    indexes_l5 = np.array(data[name].value_counts()[data[name].value_counts() < splitBy].index)
    values_l5 = -1 + np.zeros(len(indexes_l5))
    indexes = np.concatenate( (indexes_gr5, indexes_l5) )
    values = np.concatenate( (values_gr5, values_l5) )

    mapping = {}
    for A, B in zip(indexes, values):
        mapping[A] = B
    
    data.replace({name: mapping}, inplace=True)

    mapping = {'unbekannt': -1}
    for A, B in zip(indexes_gr5, values_gr5):
        mapping[A] = B
    addToRevDict({name: mapping})

    if fillNaValueAfter is not None:
        data[name] = data[name].fillna(fillNaValueAfter)


def reorderByCount(data, name, splitBy=5, defaultValue=1):
    indexes_gr5 = np.array(data[name].value_counts()[data[name].value_counts() >= splitBy].index)
    values_gr5 = len(indexes_gr5) + 1 - np.arange(len(indexes_gr5))
    indexes_l5 = np.array(data[name].value_counts()[data[name].value_counts() < splitBy].index)
    values_l5 = defaultValue + np.zeros(len(indexes_l5))
    indexes = np.concatenate( (indexes_gr5, indexes_l5) )
    values = np.concatenate( (values_gr5, values_l5) )

    mapping = {}
    for A, B in zip(indexes, values):
        mapping[A] = B
    
    data.replace({name: mapping}, inplace=True)

    mapping = {'unbekannt': defaultValue}
    for A, B in zip(indexes, values):
        mapping[A] = B
    addToRevDict({name: mapping})
        
        

def safeFilename(name):
    specialChars = {'ä': 'ae', 'Ä': 'Ae', 'ö': 'oe', 'Ö': 'Oe', 'ü': 'ue', 'Ü': 'Ue', 'ß': 'ss' }
    newName = ""
    for c in name:
        if c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-() _=0123456789,;.":
            newName += c
        elif c in specialChars:
            newName += specialChars[c]
        else:
            newName += '_'
    return newName


def ensureDir(pathname):
    if not os.path.exists(pathname):
        os.mkdir(pathname)


def discretizeDaysWithinYear(items):
    def discretize(x):
        if x == -1:
            return -1
        elif x <= 365:
            return 1
        else:
            return 2

    return np.array([discretize(x) for x in items]) 



class MD:
    def __init__(self, fileName):
        self.file = open(fileName, "w")

    def print(self, text=""):
        print(text, file=self.file)
        print(text)

    def close(self):
        self.file.close()


    def heading(self, title, n=1):
        self.print()
        self.print(("#" * n) + f" {title}")


    def image(self, url, title="graphic"):
        self.print()
        self.print(f"![{title}]({url})")
        self.print()


    def code(self, text):
        self.print()
        self.print("~~~~~~~")
        self.print(text)
        self.print()
        self.print("~~~~~~~")
        self.print()
