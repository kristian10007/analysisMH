import os
import os.path
import numpy as np
import json


global auto_rev_dict
auto_rev_dict = None

def addToRevDict(mapping):
    global auto_rev_dict

    if auto_rev_dict is None:
        auto_rev_dict = {}

    for k in mapping.keys():
        if k not in auto_rev_dict:
            auto_rev_dict[k] = {}
            
        for mk in mapping[k].keys():
            v = mapping[k][mk]
            if v not in auto_rev_dict[k]:
                auto_rev_dict[k][mk] = v
                
def saveAutoRevDict():
    global auto_rev_dict
    if auto_rev_dict is not None:
        with open("data/rev_dict.json", "w") as f:
            json.dump(auto_rev_dict, f)


def loadAutoRevDict():
    rev_dict = {}
    fileName = "data/rev_dict.json"
    if os.path.exists(fileName):
        with open(fileName) as f:
            rev_dict = json.load(f)
    else:
        print(f"File '{fileName}' is missing. Did you run step 1?")
    return rev_dict
