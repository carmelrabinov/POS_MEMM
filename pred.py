# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:14:36 2017

@author: amir
"""

import numpy as np
import scipy as sc
from itertools import compress
import time
import argparse
try: import cPickle as pickle
except: import pickle
import copy
import os
import random

############# parcing data ###############
project_dir = os.path.dirname(os.path.realpath('__file__'))
data_path = project_dir + '\\data\\train.wtag' 
test_path = project_dir + '\\data\\test.wtag'
comp_path = project_dir + '\\data\\comp.words'


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

# add start and stop signs to each sentence
for sentence,tags in zip(data,data_tag):
    sentence.append('/STOP')
    sentence.insert(0, '/*')
    sentence.insert(0, '/*')
    tags.append('/STOP')
    tags.insert(0, '/*')
    tags.insert(0, '/*')

   
# tag options
from itertools import chain
T = sorted(list(set(chain(*data_tag))))
T_size = len(T)

T_no_start = [x for x in T if (x != '/*' and x != '/STOP')]
#T.append('*')
#T.append('STOP')

# vocanulary
V = sorted(list(set(chain(*data))))
#T = [x for x in T if (x != '/*' and x != '/STOP')]
V_size = len(V)


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


# comp
comp = []   # holds the data
comp_tag = []  # holds the taging 

#DEBUG:
line1 = 'The black dog chased the cat'
line2 = 'The black dog chased the cat'
f=[]
f.append(line1)
f.append(line2)

#f = open(comp_path, "r")
for line in f:
    linesplit = []
    for word in line.split():
        linesplit.append(word)
    comp.append(linesplit)
    
############# end of parcing data ###############

# TODO: write the function!
# tag - tag of word at position k
# word - word at position k
# u = tag at word k-1
# t = tag at word k-2    
def calc_probability(tag,word,u,t):
#    return random.uniform(0, 1)
    if word == 'The':
        if tag == 'TO':
            if u == '/*' and t == '/*':
                return 1
        return 0.5
        
    if word == 'black':
        if tag == 'JJ':
            if u == 'TO' and t == '/*':
                return 1

    if word == 'dog':
        if tag == 'NN':
            if u == 'JJ' and t == 'TO':
                return 1
        return 0.5
    
    if word == 'chased':
        if tag == 'VB':
            if u == 'NN' and t == 'JJ':
                return 1

    if word == 'the':
        if tag == 'TO':
            if u == 'VB' and t == 'NN':
                return 1
        else:
            return 0.5
    if word == 'cat':
        if tag == 'NN':
            if u == 'TO' and t == 'VB':
                return 1   

    return 0

    

    
mode = 'comp' #TODO: add args extraction

if mode == 'test':
    corpus = test
else:
    corpus = comp

weights = [];

# init set of possible tags:
S={}
tmp_list = [];
tmp_list.append('/*')
S[-2] = tmp_list
S[-1] = tmp_list
S[0] = tmp_list


# run Viterbi algorithm for each sentence:
for sentence in corpus:
    # init empty array of strings to save the tag for each word in the sentance    
    sentence_len = len(sentence)
    sentence_tags = [''  for x in range(sentence_len)]

    # init dynamic matrix with size: 
    # pi_matrix[k,t,u] is the value of word number *k*, preciding tag u and t accordingly
    pi_matrix = np.zeros((sentence_len+1,T_size,T_size))
    pi_matrix[0,T.index('/*'),T.index('/*')] = 1
    
    # init back pointers matrix:
    #bp[k,t,u] is the tag index of word number *k-2*, following tag t and u accordingly
    bp = np.zeros((sentence_len+1,T_size,T_size),dtype=np.int)
    
    # init relevant tags set for each word in the sentence:
    for i in range(1,sentence_len+1):
        S[i] = T_no_start
    
    
    # u = word at position k-1
    # t = word at position k-2   
    for k in range (1,sentence_len+1): # for each word in the sentence
        for current_tag in S[k]: # for each t possible tag
            for u in S[k-1]: # for each t-1 possible tag
                for t in S[k-2]: # for each t-2 possible tag:
                    #calculate pi value, and check if it exeeds the current max:
                    tmp_val = pi_matrix[k-1,T.index(t),T.index(u)] * calc_probability(current_tag,sentence[k-1],u,t)
                    if tmp_val > pi_matrix[k,T.index(u),T.index(current_tag)]:
                        # update max:
                        pi_matrix[k,T.index(u),T.index(current_tag)] = tmp_val;
                        
                        # update back pointers:
                        bp[k,T.index(u),T.index(current_tag)] = T.index(t)
                        
                        #if its the last word in the sentence, save the last two tags:
                        if k == (sentence_len):
                            sentence_tags[k-1] = current_tag
                            sentence_tags[k-2] = u
    
    # extracting MEMM tags path from back pointers matrix:
    for i in range(sentence_len-3,-1,-1):
        # calculate the idx of tag i in T db:
        # reminder - bp[k,t,u] is the tag of word *k-2*, following tag t and u accordingly
        k_tag_idx = bp[i+3,T.index(sentence_tags[i+1]),T.index(sentence_tags[i+2])]

        # update the i-th tag to the list of tags
        sentence_tags[i] = T[k_tag_idx]

    # build tagged sentence:
    tagged_sentence = ''
    for i in range(sentence_len):
        tagged_sentence += (sentence[i] +'_')
        tagged_sentence += sentence_tags[i] + (' ')
    print(tagged_sentence)
    
