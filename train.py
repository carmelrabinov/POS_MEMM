# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:44:28 2017

@author: carmelr
"""

import numpy as np
import scipy as sc
from itertools import compress
import time
import argparse
try: import cPickle as pickle
except: import pickle
import copy


############# parcing data ###############
data_path = 'D:\\TECHNION\\NLP\\part_of_speech_taging_MEMM\\data\\train.wtag'
test_path = 'D:\\TECHNION\\NLP\\part_of_speech_taging_MEMM\\data\\test.wtag'

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

############# end of parcing data ###############



# parameters: V - vocabulary, T - list of possible tags, 
#             xi - the word, ti - the tag
#             th - tag history vector <t(i-2),t(i-1)>
#             mode - base / complex
# return a feature vector representing p(yi|xi)
def get_features(xi, ti, th, mode, to_compress):
    if mode == 'base':
        # 1 if xi = x and ti = t
        features_100 = np.zeros(V_size * T_size, dtype = bool)
        try: features_100[V.index(xi) * T_size + T.index(ti)] = True
        except: pass
    
        # 1 if <t(i-2),t(i-1),t(i)> = <t1,t2,t3>
        features_103 = np.zeros(T_size**3, dtype = bool)
        try: features_103[T.index(ti)*(T_size**2) + T.index(th[1])*T_size + T.index(th[0])] = True
        except: pass
    
        # 1 if <t(i-1),t(i)> = <t1,t2>
        features_104 = np.zeros(T_size**2, dtype = bool)
        try: features_104[T.index(ti) * T_size + T.index(th[1])] = True
        except: pass
    
        features = np.concatenate((features_100, features_103, features_104))
        
        if to_compress: return np.nonzero(features)
        else: return features
        

def get_sentence_features(sentence, tags, mode, to_compress):
    if mode == 'base':
        
        if to_compress:
            sentence_features_shape = (len(sentence)-3,3)
        else:
            sentence_features_shape = (len(sentence)-3,T_size**3 + T_size**2 + V_size*T_size)

        sentence_features=np.empty(shape = sentence_features_shape, dtype = np.int64)
        # iterate over all the words in the sentence besides START and STOP special signs
        for i, word in enumerate(sentence[:-1]):
            if i==0 or i==1:
                continue
            # 1 if xi = x and ti = t
            features_100 = np.zeros(V_size * T_size, dtype = np.bool)
            try: features_100[V.index(word) * T_size + T.index(tags[i])] = True
            except: pass
        
            # 1 if <t(i-2),t(i-1),t(i)> = <t1,t2,t3>
            features_103 = np.zeros(T_size**3, dtype = bool)
            try: features_103[T.index(tags[i])*(T_size**2) + T.index(tags[i-1])*T_size + T.index(tags[i-2])] = True
            except: pass
            # 1 if <t(i-1),t(i)> = <t1,t2>
            features_104 = np.zeros(T_size**2, dtype = bool)
            try: features_104[T.index(tags[i]) * T_size + T.index(tags[i-1])] = True
            except: pass
        
            features = np.concatenate((features_100, features_103, features_104))

            if to_compress:
                sentence_features[i-2,:] = np.array(np.nonzero(features))
            else:
                sentence_features[i-2,:] = features
        
        return sentence_features
    
    
def get_all_tags_for_word_features(xi, th, mode, to_compress):
    if mode == 'base':
        if to_compress:
            sentence_features_shape = (T_size-2,3)
        else:
            sentence_features_shape = (T_size-2, T_size**3 + T_size**2 + V_size*T_size)

        sentence_features=np.empty(shape = sentence_features_shape, dtype = np.int64)
        # iterate over all the words in the sentence besides START and STOP special signs
        for i, tag in enumerate(T_no_start):

            # 1 if xi = x and ti = t
            features_100 = np.zeros(V_size * T_size, dtype = np.bool)
            try: features_100[V.index(xi) * T_size + T.index(tag)] = True
            except: pass
        
            # 1 if <t(i-2),t(i-1),t(i)> = <t1,t2,t3>
            features_103 = np.zeros(T_size**3, dtype = bool)
            try: features_103[T.index(tag)*(T_size**2) + T.index(th[1])*T_size + T.index(th[0])] = True
            except: pass
            # 1 if <t(i-1),t(i)> = <t1,t2>
            features_104 = np.zeros(T_size**2, dtype = bool)
            try: features_104[T.index(tag) * T_size + T.index(th[1])] = True
            except: pass
        
            features = np.concatenate((features_100, features_103, features_104))

            if to_compress:
                sentence_features[i,:] = np.array(np.nonzero(features))
            else:
                sentence_features[i,:] = features
        
        return sentence_features

    
    

#def sentence_(sentence, tag_sentence):   
#    features_dict = dict()
#    for i, word in enumerate(sentence[:-1]):
#        if i==0 or i==1:
#            continue
#        d = dict()
#        for tag in T:
#            d[tag] = get_features(word, tag, [tag_sentence[i-2],tag_sentence[i-1]], 'base', True)
#        features_dict[word] = d
#    
#    return features_dict

   



########### main ###########

max_epoch = 4


# init
l = 0.001
lr = 0.2
feature_size = T_size**3 + T_size**2 + V_size*T_size
w = np.zeros(feature_size, dtype = np.float32)
w_grads = np.zeros(feature_size, dtype = np.float32)


weightsL = []
timeL = [0]
weightsL.append(copy.deepcopy(w))

start_time = time.time()

for epoch in range(max_epoch):
    # calc grads
    for h, sentence in enumerate(data):
        if h%50 == 0:
            print('finished sentence {} in {} min'.format(h, (time.time() - start_time)/60))
        tag_sentence = data_tag[h]
        
        normalization_conts = l*w
        
        sentence_features = get_sentence_features(sentence, tag_sentence, 'base', False)
        empirical_counts = np.sum(sentence_features, axis = 0) 
    
        expected_counts = np.zeros(feature_size, dtype = np.float32)
        
        # calculate p_array which contains p(y|x,w)
        p_array = np.zeros((len(sentence[2:-1]),T_size-2), dtype = np.float32)
        for i, word in enumerate(sentence[:-1]):
                if i==0 or i==1:
                    continue
                feats = get_all_tags_for_word_features(word, [tag_sentence[i-1],tag_sentence[i-2]], 'base', True)
                for j, tag in enumerate(T_no_start):
                    tag_feat = feats[j,:]              
                    p_array[i-2,j] = np.exp(np.sum(w[tag_feat])) / np.sum(np.exp(np.sum(w[feats], axis = 1)))
    
        # calc f_array which contains f(x,y)
        f_array = np.zeros((len(sentence[2:-1]),T_size-2,3), dtype = np.int64)
        for i, word in enumerate(sentence[:-1]):
            if i==0 or i==1:
                continue
            f_array[i-2,:,:] = get_all_tags_for_word_features(word, [tag_sentence[i-1],tag_sentence[i-2]], 'base', True)
    
        # calc empirical counts
        for i, word in enumerate(sentence[:-1]):
            if i==0 or i==1:
                continue
            for j, tag in enumerate(T_no_start):
                expected_counts[f_array[i-2,j,:]] += p_array[i-2,j]
    
    
        w_grads = empirical_counts - expected_counts - normalization_conts        
#        print('expected_counts: ',np.sum(expected_counts),', empirical_counts: ',np.sum(empirical_counts),', normalization_conts: ',np.sum(normalization_conts))        
        np.sum(empirical_counts)
        # update wieghts
#        print('tot_grad: ',np.sum(w_grads * lr))
        w += w_grads * lr
        
    weightsL.append(copy.deepcopy(w))
    timeL.append(time.time() - start_time)
    print('finished epoch {} in {} min'.format(epoch+1, (time.time() - start_time)/60))
    with open('./results.log', 'wb') as f:
        pickle.dump([weightsL, timeL], f)



    