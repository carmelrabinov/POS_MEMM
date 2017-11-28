# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:14:36 2017

@author: amir
"""

import numpy as np
import time
import argparse
try: import cPickle as pickle
except: import pickle
import os
import sys
from itertools import chain
from scipy.sparse import csr_matrix


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

        return csr_matrix(si_features)


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
    tmp = np.exp(csr_matrix.sum(w.multiply(all_y_feats), axis = 1)).reshape(T_size-2)
    return tmp / np.sum(tmp)


def viterby_predictor(corpus, w, prob_mat = None):
    """
    calculate the tags for the corpus
    :param corpus: a list of sentences (each sentence as a list of words) 
    :param w: trained weights
    :param prob_mat: the propability matrix for this corpus if exist
    :return: all_sentence_tags: a list of tagged sentences (each sentence as a list of tags
             all_tagged_sentence: a list of tagged sentences in form of "word_tag"
             prob_mat: the propability matrix for this corpus
    """
    # compress weights using sparse matrix representation
    weights = csr_matrix(w)
    
    # init a list of singular words in the target corpus:
    V_COMP = sorted(list(set(chain(*corpus))))
    V_COMP_size = len(V_COMP)
       
    # init probability matrix:
    # holds all p(word,t(i),t(i-1),t(i-2))
    if prob_mat == None:
        prob_mat = np.zeros((V_COMP_size, T_size - 2,T_size - 2,T_size - 2))
    
    all_sentence_tags = []
    all_tagged_sentence = []
 
    for sentence in corpus:
        t0 = time.time()

        # init empty array of strings to save the tag for each word in the sentance
        sentence_len = len(sentence)
        sentence_tags = [''  for x in range(sentence_len)]
    
        # init dynamic matrix with size: 
        # pi_matrix[k,t(i-1),t(i)] is the value of word number *k*, preciding tag u and t accordingly
        pi_matrix = np.zeros((sentence_len,T_size-2,T_size-2))
        
        # init back pointers matrix:
        # bp[k,t,u] is the tag index of word number *k-2*, following tag t and u accordingly
        bp = np.zeros((sentence_len,T_size-2,T_size-2),dtype=np.int)
              
        # holds all p(t(i),t(i-1),t(i-2))
        # prob_mat = np.zeros((T_size - 2,T_size - 2,T_size - 2))
    
        for k in range (0,sentence_len): # for each word in the sentence
            # if havn't seen the word before - update the probebility matrix for all possible tagsL
            if k > 1 and not prob_mat[V_COMP.index(sentence[k]),0,0,0].any():
                for u in T: # for each t-1 possible tag
                    for t in T: # for each t-2 possible tag:
                        prob_mat[V_COMP.index(sentence[k]),:, T.index(u), T.index(t)] = calc_all_possible_tags_probabilities(sentence[k], u, t, weights)
            for current_tag in T: # for each t possible tag
                
                if k == 0:
                    # at the first two words there is no meaning to the k-1 tag index. pi[k-1]
                    pi_matrix[k, 0, :] = 1 * calc_all_possible_tags_probabilities(sentence[k], '/*', '/*', weights)
    
                elif k == 1:
                    for u in T: # for each t-1 possible tag
                        pi_matrix[k, T.index(u), :] = pi_matrix[k - 1, 0, T.index(u)] * calc_all_possible_tags_probabilities(sentence[k], u, '/*', weights)
    
                else:      
                    for u in T: # for each t-1 possible tag
                        # calculate pi value, and check if it exeeds the current max:
                        pi_values = pi_matrix[k-1, :, T.index(u)] * prob_mat[V_COMP.index(sentence[k]),T.index(current_tag), T.index(u), :]
                        ind = np.argmax(pi_values)
                        if pi_values[ind] > pi_matrix[k, T.index(u), T.index(current_tag)]:
                            
                            # update max:
                            pi_matrix[k, T.index(u), T.index(current_tag)] = pi_values[ind]
                            
                            # update back pointers:
                            bp[k, T.index(u), T.index(current_tag)] = ind

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
        all_sentence_tags.append(sentence_tags)
        all_tagged_sentence.append(tagged_sentence)
        print(tagged_sentence, ' ,time: ', time.time() - t0)

    return(all_tagged_sentence, all_sentence_tags, prob_mat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('trained_model_Fn', help='output results')
    parser.add_argument('-input_path', type=str, default=None)
    parser.add_argument('-end', type=int, default=0)
    parser.add_argument('-test_accuracy', type=int, default=0)
    parser.parse_args(namespace=sys.modules['__main__'])

    project_dir = os.path.dirname(os.path.realpath('__file__'))

#    test_path = project_dir + '\\data\\test.wtag'
#    comp_path = project_dir + '\\data\\comp.words'
#    data_path = project_dir + '\\data\\train.wtag'
  
    # load all data:
    with open(project_dir + '\\' + trained_model_Fn + '.log', 'rb') as f:
        [weights, _, V, T_with_start, data, data_tag, test, test_tag] = pickle.load(f)
        print('loaded trained model')

    V_size = len(V)
    T_size = len(T_with_start)
    T = [x for x in T_with_start if (x != '/*' and x != '/STOP')]

    # case input file is not given, use loaded test
    if input_path == None:
            # remove /* and /Stop word
        for i, line in enumerate(data):
            test[i] = test[i][2:-1]
        for i, line in enumerate(test):
            test_tag[i] = test_tag[i][2:-1]

    # case input file is given
    else:
        test = []  # holds the data
        test_tag = []  # holds the taging
        with open(input_path, 'r') as f:
            for line in f:
                linesplit = []
                tagsplit = []
                for word in line.split():
                    word, tag = word.split('_')
                    linesplit.append(word)
                    tagsplit.append(tag)
                test_tag.append(tagsplit)
                test.append(linesplit)

    if end:
        corpus = test[:end]
    else:
        corpus = test

    # run Viterbi algorithm
    (all_tagged_sentence, all_sentence_tags, _) = viterby_predictor(corpus, weights)

    line_accuracy = []
    tot_accuracy = 0
    if test_accuracy:
        tot_length = 0
        tot_correct = 0
        for i, tag_line in enumerate(all_sentence_tags):
            res = np.sum([x == y for x, y in zip(tag_line, test_tag[i])])
            line_accuracy.append(res/len(tag_line))
            tot_length += len(tag_line)
            tot_correct += res

        tot_accuracy = tot_correct/tot_length
        print("Total accuracy is: ", tot_accuracy)

    # save predictions:
    with open(project_dir + '\\' + trained_model_Fn + '_predictions.log', 'wb') as f:
        pickle.dump([all_sentence_tags, all_tagged_sentence, line_accuracy, tot_accuracy], f)
    with open(project_dir + '\\' + trained_model_Fn + '_predictions.txt') as f2:
        for s in all_tagged_sentence:
            f2.writelines(s)
