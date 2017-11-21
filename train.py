# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:44:28 2017

@author: carmelr
"""

import numpy as np
import scipy as sc
from itertools import compress

data_path = 'D:\\TECHNION\\NLP\\Wet1\\data\\train.wtag'
test_path = 'D:\\TECHNION\\NLP\\Wet1\\data\\test.wtag'

data = []   # holds the data
data_tag = []  # holds the taging 
f = open(data_path, "r")
for line in f:
    linesplit = []
    tagsplit = []
    for word in line.split():
        word,tag = word.split('_')
        linesplit.append(word)
        tagsplit.append(tag)
    data_tag.append(tagsplit)
    data.append(linesplit)
    
# tag options
from itertools import chain
T = sorted(list(set(chain(*data_tag))))
#T.append('*')
#T.append('STOP')

# vocanulary
V = sorted(list(set(chain(*data))))


# test
test = []   # holds the data
test_tag = []  # holds the taging 
f = open(test_path, "r")
for line in f:
    linesplit = []
    tagsplit = []
    for word in line.split():
        word,tag = word.split('_')
        linesplit.append(word)
        tagsplit.append(tag)
    test_tag.append(tagsplit)
    test.append(linesplit)
    
# parameters: V - vocabulary, T - list of possible tags, 
#             xi - the word, ti - the tag
#             th - tag history vector <t(i-2),t(i-1)>
#             mode - base / complex
# return a feature vector representing p(yi|xi)
def get_features(V, T, xi, ti, th, mode):
    V_size = len(V)
    T_size = len(T)
    if mode == 'base':
        # 1 if xi = x and ti = t
        features_100 = np.zeros(V_size * T_size, dtype = bool)
        features_100[V.index(xi) * T_size + T.index(ti)] = True
        
        # 1 if <t(i-2),t(i-1),t(i)> = <t1,t2,t3>
        features_103 = np.zeros(T_size**3, dtype = bool)
        features_103[T.index(ti)*(T_size**2) + T.index(th[1])*T_size + T.index(th[0])] = True

        # 1 if <t(i-1),t(i)> = <t1,t2>
        features_104 = np.zeros(T_size**2, dtype = bool)
        features_104[T.index(ti) * T_size + T.index(th[1])] = True
        
        features = np.concatenate((features_100, features_103, features_104))
        return np.nonzero(features)
#        return features
        
    

