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


def get_base_features(word, tags):
    """
    :param word: the word
    :param tags: POS tags of the trigram as as a list <t(i-2), t(i-1), t(i)>
    :return: features - a binary feature vector
    """
    # 1 if xi = x and ti = t
    features_100 = np.zeros(V_size * T_size, dtype=np.bool)
    try: features_100[V.index(word) * T_size + T_with_start.index(tags[2])] = True
    except: pass 

    # trigram feature - 1 if <t(i-2),t(is),t(i)> = <t1,t2,t3>
    features_103 = np.zeros(T_size ** 3, dtype=bool)
    features_103[T_with_start.index(tags[2]) * (T_size ** 2) +
                 T_with_start.index(tags[1]) * T_size + T_with_start.index(tags[0])] = True
    
    # bigram feature - 1 if <t(i-1),t(i)> = <t1,t2>
    features_104 = np.zeros(T_size ** 2, dtype=bool)
    features_104[T_with_start.index(tags[2]) * T_size + T_with_start.index(tags[1])] = True

    return np.concatenate((features_100, features_103, features_104))


def get_word_all_possible_tags_features(xi, th, mode, to_compress):
    """
    computes all feature vectors of a given word with all possible POS tags
    :param xi: the word
    :param th: tag history vector <t(i-2),t(i-1)>
    :param mode: base / complex
    :param to_compress: True if return value as a list of index, False if return value as binary feature vector
    :return: feature matrix as a matrix of indexes or as a binary matrix, each row represent a possible POS tag
    """
    if mode == 'base':
        if to_compress:
            sentence_features_shape = (T_size - 2, 3)
            si_features = np.empty(shape=sentence_features_shape, dtype=np.int64)

        else:
            sentence_features_shape = (T_size - 2, T_size ** 3 + T_size ** 2 + V_size * T_size)
            si_features = np.empty(shape=sentence_features_shape, dtype=bool)


        # iterate over all the words in the sentence besides START and STOP special signs
        for i, tag in enumerate(T):
            features = get_base_features(xi, [th[0], th[1], tag])

            if to_compress:
                si_features[i, :] = np.array(np.nonzero(features))
            else:
                si_features[i, :] = features

        return si_features

#def get_word_all_possible_tags_features(xi, th, mode, to_compress):
#    """
#    computes all feature vectors of a given word with all possible POS tags
#    :param xi: the word
#    :param th: tag history vector <t(i-2),t(i-1)>
#    :param mode: base / complex
#    :param to_compress: True if return value as a list of index, False if return value as binary feature vector
#    :return: feature matrix as a matrix of indexes or as a binary matrix, each row represent a possible POS tag
#    """
#    if mode == 'base':
#        if to_compress:
#            sentence_features_shape = (T_size - 2, 3)
#            si_features = np.empty(shape=sentence_features_shape, dtype=np.int64)
#
#        else:
#            sentence_features_shape = (T_size - 2, T_size ** 3 + T_size ** 2 + V_size * T_size)
#            si_features = np.empty(shape=sentence_features_shape, dtype=bool)
#
#
#        # iterate over all the words in the sentence besides START and STOP special signs
#        for i, tag in enumerate(T_no_start):          
#            features = get_base_features(xi, [th[0], th[1], tag])
#
#            if to_compress:
#                si_features[i, :] = np.array(np.nonzero(features))
#            else:
#                si_features[i, :] = features
#
#        return si_features


#def viterby_predictor(sentence, weights):
#    # init set of possible tags:
#    S={}
#    tmp_list = [];
#    tmp_list.append('/*')
#    S[-2] = tmp_list
#    S[-1] = tmp_list
#    S[0] = tmp_list
#
#    # init empty array of strings to save the tag for each word in the sentance    
#    sentence_len = len(sentence)
#    sentence_tags = [''  for x in range(sentence_len)]
#
#    # init dynamic matrix with size: 
#    # pi_matrix[k,t,u] is the value of word number *k*, preciding tag u and t accordingly
#    pi_matrix = np.zeros((sentence_len+1,T_size,T_size))
#    pi_matrix[0, T_with_start.index('/*'), T_with_start.index('/*')] = 1
#    
#    # init back pointers matrix:
#    #bp[k,t,u] is the tag index of word number *k-2*, following tag t and u accordingly
#    bp = np.zeros((sentence_len+1,T_size,T_size),dtype=np.int)
#    
#    # init relevant tags set for each word in the sentence:
#    for i in range(1,sentence_len+1):
#        S[i] = T
#    
#    t0 = time.time()
#    # u = word at position k-1
#    # t = word at position k-2   
#    for k in range (1,sentence_len+1): # for each word in the sentence
#        for current_tag in S[k]: # for each t possible tag
#            for u in S[k-1]: # for each t-1 possible tag
#                for t in S[k-2]: # for each t-2 possible tag:
#                    
#                    #calculate pi value, and check if it exeeds the current max:
#                    tmp_val = pi_matrix[k - 1, T_with_start.index(t), T_with_start.index(u)] * calc_probability(current_tag, sentence[k - 1], u, t, weights, False)
#                    
#                    if tmp_val > pi_matrix[k, T_with_start.index(u), T_with_start.index(current_tag)]:
#                        
#                        # update max:
#                        pi_matrix[k, T_with_start.index(u), T_with_start.index(current_tag)] = tmp_val;
#                        
#                        # update back pointers:
#                        bp[k, T_with_start.index(u), T_with_start.index(current_tag)] = T_with_start.index(t)
#                        
##                        #if its the last word in the sentence, save the last two tags:
##                        if k == (sentence_len):   
##                            sentence_tags[k-1] = current_tag 
##                            sentence_tags[k-2] = u 
#        
#        print('finished word ',k,' time: ',time.time() - t0)
#        
#    u_ind, curr_ind = np.unravel_index(pi_matrix[sentence_len-1,:,:].argmax(), pi_matrix[sentence_len-1,:,:].shape)
#    sentence_tags[-2:] = [T_with_start[u_ind], T_with_start[curr_ind]]
#
#    
#    # extracting MEMM tags path from back pointers matrix:
#    for i in range(sentence_len-3,-1,-1):
#        # calculate the idx of tag i in T db:
#        # reminder - bp[k,t,u] is the tag of word *k-2*, following tag t and u accordingly
#        k_tag_idx = bp[i + 3, T_with_start.index(sentence_tags[i + 1]), T_with_start.index(sentence_tags[i + 2])]
#
#        # update the i-th tag to the list of tags
#        sentence_tags[i] = T_with_start[k_tag_idx]
#
#    # build tagged sentence:
#    tagged_sentence = ''
#    for i in range(sentence_len):
#        tagged_sentence += (sentence[i] +'_')
#        tagged_sentence += sentence_tags[i] + (' ')
#    
#    return(tagged_sentence, sentence_tags)


def calc_probability(ti, xi, t1, t2, w, to_compress):
    """
    calculate probability p(ti|xi,w)
    :param ti: POS tag
    :param xi: the word[i]
    :param t1: POS tag for word[i-1]
    :param t2: POS tag for word[i-2]
    :param w: weights vector
    :return: probability p(ti|xi,w) as float64
    """
    if to_compress:
        all_y_feats = get_word_all_possible_tags_features(xi, [t2, t1], 'base', True)
        tag_feat = all_y_feats[T.index(ti)]
        return np.exp(np.sum(w[tag_feat])) / np.sum(np.exp(np.sum(w[all_y_feats], axis=1)))

    else:
        all_y_feats = get_word_all_possible_tags_features(xi, [t2, t1], 'base', False)
        tag_feat = all_y_feats[T.index(ti)]
        return np.exp(np.sum(w*tag_feat)) / np.sum(np.exp(np.sum(w*all_y_feats, axis=1)))


def calc_all_possible_tags_probabilities(xi, t1, t2, w):
    """
    calculate probability p(ti|xi,w)
    :param xi: the word[i]
    :param t1: POS tag for word[i-1]
    :param t2: POS tag for word[i-2]
    :param w: weights vector
    :return: a list for all posbible ti probabilities p(ti|xi,w) as float64
    """
    all_y_feats = get_word_all_possible_tags_features(xi, [t2, t1], 'base', False)
    return np.exp(np.sum(w*all_y_feats, axis = 1)) / np.sum(np.exp(np.sum(w*all_y_feats, axis=1)))


def viterby_predictor(sentence, weights):
    # init empty array of strings to save the tag for each word in the sentance    
    sentence_len = len(sentence)
    sentence_tags = [''  for x in range(sentence_len)]

    # init dynamic matrix with size: 
    # pi_matrix[k,t(i-1),t(i)] is the value of word number *k*, preciding tag u and t accordingly
    pi_matrix = np.zeros((sentence_len,T_size-2,T_size-2))
    
    # init back pointers matrix:
    #bp[k,t,u] is the tag index of word number *k-2*, following tag t and u accordingly
    bp = np.zeros((sentence_len,T_size-2,T_size-2),dtype=np.int)
          
    # holds all p(t(i),t(i-1),t(i-2))
    prob_mat = np.zeros((T_size - 2,T_size - 2,T_size - 2))

    for k in range (0,sentence_len): # for each word in the sentence

        if k > 1: # seceond word and above                         
            
            # u = word at position k-1
            # t = word at position k-2   
            for u in T: # for each t-1 possible tag
                for t in T: # for each t-2 possible tag:
                    prob_mat[:, T.index(u), T.index(t)] = calc_all_possible_tags_probabilities(sentence[k], u, t, weights)
       
        for current_tag in T: # for each t possible tag
            
            if k == 0:
                pi_matrix[k, T_with_start.index('/*'), :] = calc_all_possible_tags_probabilities(sentence[k], '/*', '/*', weights)

            elif k == 1:
                for u in T: # for each t-1 possible tag
                    pi_matrix[k, T.index(u), :] = pi_matrix[k - 1, T_with_start.index('/*'), T.index(u)] * calc_all_possible_tags_probabilities(sentence[k], u, '/*', weights)

            else:      
                for u in T: # for each t-1 possible tag
                
                    #calculate pi value, and check if it exeeds the current max:
                    pi_values = pi_matrix[k-1, :, T.index(u)] * prob_mat[T.index(current_tag), T.index(u), :]
                    ind = np.argmax(pi_values)
                    if pi_values[ind] > pi_matrix[k, T.index(u), T.index(current_tag)]:
                        
                        # update max:
                        pi_matrix[k, T.index(u), T.index(current_tag)] = pi_values[ind]
                        
                        # update back pointers:
                        bp[k, T.index(u), T.index(current_tag)] = ind
             

#                #calculate pi value, and check if it exeeds the current max:
#                pi_values = pi_matrix[k-1, :, :] * np.transpose(prob_mat[T.index(current_tag), :, :])
#                ind = np.argmax(pi_values, axis = 0)
#                bool_list = pi_values[ind] > pi_matrix[k, :, T.index(current_tag)]
#                if pi_values[ind] > pi_matrix[k, T.index(u), T.index(current_tag)]:
#                    
#                    # update max:
#                    pi_matrix[k, T.index(u), T.index(current_tag)] = pi_values[ind]
#                    
#                    # update back pointers:
#                    bp[k, T.index(u), T.index(current_tag)] = ind



    u_ind, curr_ind = np.unravel_index(pi_matrix[sentence_len-1,:,:].argmax(), pi_matrix[sentence_len-1,:,:].shape)
    sentence_tags[-2:] = [T[u_ind], T[curr_ind]]
        
        
    # extracting MEMM tags path from back pointers matrix:
    for i in range(sentence_len-3,-1,-1):
        # calculate the idx of tag i in T db:
        # reminder - bp[k,t,u] is the tag of word *k-2*, following tag t and u accordingly
        k_tag_idx = bp[i + 2, T.index(sentence_tags[i + 1]), T.index(sentence_tags[i + 2])]

        # update the i-th tag to the list of tags
        sentence_tags[i] = T[k_tag_idx]

    # build tagged sentence:
    tagged_sentence = ''
    for i in range(sentence_len):
        tagged_sentence += (sentence[i] +'_')
        tagged_sentence += sentence_tags[i] + (' ')
    
    return(tagged_sentence, sentence_tags)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('resultsFn', help='output results')
    parser.parse_args(namespace=sys.modules['__main__'])


    project_dir = 'D:\\TECHNION\\NLP\\part_of_speech_taging_MEMM'
    resultsFn = 'results_pycharm2'

#    project_dir = os.path.dirname(os.path.realpath('__file__'))
    test_path = project_dir + '\\data\\test.wtag'
    comp_path = project_dir + '\\data\\comp.words'
    data_path = project_dir + '\\data\\train.wtag'
    
  
    # load all data:
    with open(project_dir + '\\train_results\\' + resultsFn + '.log', 'rb') as f:
        [weights, _, V, T_with_start, data, data_tag, test, test_tag] = pickle.load(f)
    
    V_size = len(V)
    T_size = len(T_with_start)
    T = [x for x in T_with_start if (x != '/*' and x != '/STOP')]
  
   
    for i, line in enumerate(data):
        data[i] = data[i][2:-1]
    for i, line in enumerate(data):
        data_tag[i] = data_tag[i][2:-1]

    corpus = data
           
    all_sentence_tags = []
    all_tagged_sentence = []
    
 
    # run Viterbi algorithm for each sentence:
    curr_time = time.time()
    for k, sentence in enumerate(corpus):
        (tagged_sentence, sentence_tags) = viterby_predictor(sentence, weights)
        all_sentence_tags.append(sentence_tags)
        all_tagged_sentence.append(tagged_sentence)
        print('finished sentence ',k,' in ',time.time() - curr_time)
        curr_time = time.time()

    line_accuracy = []
    tot_length = 0
    tot_correct = 0
    for i, tag_line in enumerate(all_sentence_tags):       
        res = np.sum([x==y for x,y in zip(tag_line, data_tag[i])])
        line_accuracy.append(res/len(tag_line))
        tot_length += len(tag_line)
        tot_correct += res
        
    tot_accuracy = tot_correct/tot_length
        
    # save predictions:
    with open(project_dir + '\\train_results\\' + resultsFn + '_predictions.log', 'wb') as f:
        pickle.dump([all_sentence_tags, all_tagged_sentence, line_accuracy, tot_accuracy],f)

