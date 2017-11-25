# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:44:28 2017

@author: carmelr
"""
import sys
import os
import numpy as np
import scipy as sc
from itertools import compress
import time
import argparse

try: import cPickle as pickle
except: import pickle
import copy
from itertools import chain
import random


# def load_results(Fn):
##Fn = 'D:\\TECHNION\\NLP\\part_of_speech_taging_MEMM\\results2.log'
#    with open(Fn, 'rb') as f:
#        [weightsL, timeL] = pickle.load(f)



def data_preprocessing(data_path, test_path):
    """
    :param data_path: path to train data
    :param test_path: path to test data
    :return: V: vocabulary
             T: list of possible tags
             data: list of sentences, each sentece as a list of words,
             data_tag: list of sentences tags, each sentece as a list of tags
             test: list of test sentences, each sentece as a list of words
             test_tag: list of test sentences tags, each sentece as a list of tags
    """
    data = []  # holds the data
    data_tag = []  # holds the taging 
    f = open(data_path, "r")
    for line in f:
        linesplit = []
        tagsplit = []
        for word in line.split():
            word, tag = word.split('_')
            linesplit.append(word)
            tagsplit.append(tag)
        data_tag.append(tagsplit)
        data.append(linesplit)

    # add start and stop signs to each sentence
    for sentence, tags in zip(data, data_tag):
        sentence.append('/STOP')
        sentence.insert(0, '/*')
        sentence.insert(0, '/*')
        tags.append('/STOP')
        tags.insert(0, '/*')
        tags.insert(0, '/*')
        

    # tag options
    T = sorted(list(set(chain(*data_tag))))

    # vocanulary
    V = sorted(list(set(chain(*data))))

    # test
    test = []  # holds the data
    test_tag = []  # holds the taging 
    f = open(test_path, "r")
    for line in f:
        linesplit = []
        tagsplit = []
        for word in line.split():
            word, tag = word.split('_')
            linesplit.append(word)
            tagsplit.append(tag)
        test_tag.append(tagsplit)
        test.append(linesplit)

    return (V, T, data, data_tag, test, test_tag)


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


def get_sentence_features(si, tags, mode, to_compress):
    """
    computes all feature vectors of a sentence
    :param si: a sentence represented by a list of words
    :param tags: a list of POS tags for each word in the sentence
    :param mode: base / complex
    :param to_compress: True if return value as a list of index, False if return value as binary feature vector
    :return: feature matrix as a matrix of indexes or as a binary matrix, each row represent a word
    """
    if mode == 'base':

        if to_compress:
            sentence_features_shape = (len(si) - 3, 3)
            si_features = np.empty(shape=sentence_features_shape, dtype=np.int64)

        else:
            sentence_features_shape = (len(si) - 3, T_size ** 3 + T_size ** 2 + V_size * T_size)
            si_features = np.empty(shape=sentence_features_shape, dtype=bool)

        # iterate over all the words in the sentence besides START and STOP special signs
        for i, word in enumerate(si[:-1]):
            if i == 0 or i == 1:
                continue          
            features = get_base_features(word, tags[i-2:i+1])         
            
            if to_compress:
                si_features[i - 2, :] = np.array(np.nonzero(features))
            else:
                si_features[i - 2, :] = features

        return si_features

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


def calc_probability_for_all_possible_t2_tags(ti, xi, t1, w, is_start):
    """
    calculate a list of probabilities p(ti|xi,w) for all t(i-2) options
    :param ti: POS tag
    :param xi: the word[i]
    :param t1: POS tag for word[i-1]
    :param w: weights vector
    :param is_start: True if i=0 or 1, False else, use for dealing with /* start tag
    :return: a list of probability p(ti|xi,w) as float64 for all t(i-2) options
    """


############
#(ti, xi, t1, t2, w) = ('DT', 'The', '/*', '/*', w)
#
#sys.getsizeof(all_y_feats)
#
#
#But_CC wire_NN transfers_NNS
#
#
##some tests:
#t0=time.time()
#p = calc_probability('NN', 'But', '/*', '/*', w, True)
#
#end = time.time() - t0
#
#p = []
#for t in T_no_start:
#    p.append((calc_probability(t, 'But', 'NN', 'DT', w, False),t))
#
#end2 = time.time() - end - t0
#
#print(p, '  ', end)
#print(p2, '  ', end2)
#
################



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('resultsFn', help='output results')
    parser.add_argument('-max_epoch', type=int, default=4)
    parser.add_argument('-lr', type=np.float64, default=0.2)
    parser.add_argument('-lambda_rate', type=np.float64, default=0.1)
    parser.add_argument('-noShuffle', action='store_false')
    parser.add_argument('-test', action='store_true')
    parser.parse_args(namespace=sys.modules['__main__'])

#    ##############
#    max_epoch = 40
#    lr = 0.2
#    lambda_rate = 0.1
#    noShuffle = True
#    test = True
#    ##############

    project_dir = 'D:\\TECHNION\\NLP\\part_of_speech_taging_MEMM'
    
#    project_dir = os.path.dirname(os.path.realpath('__file__'))
    test_path = project_dir + '\\data\\test.wtag'
    comp_path = project_dir + '\\data\\comp.words'
    data_path = project_dir + '\\data\\train.wtag'

    # run on very small corpus to test the algorithm
    if test:
        data_path = project_dir + '\\data\\carmel_test2.txt'

    (V, T_with_start, data, data_tag, test, test_tag) = data_preprocessing(data_path, test_path)

    V_size = len(V)
    T_size = len(T_with_start)
    T = [x for x in T_with_start if (x != '/*' and x != '/STOP')]

    # init
    feature_size = T_size ** 3 + T_size ** 2 + V_size * T_size
    w = np.zeros(feature_size, dtype=np.float64)
    w_grads = np.zeros(feature_size, dtype=np.float64)

    weightsL = []
    timeL = [0]
    weightsL.append(copy.deepcopy(w))

    start_time = time.time()

    for epoch in range(max_epoch):

        # each epoch shuffle the order of the train sentences in the data
        if not noShuffle:
            s = list(zip(data, data_tag))
            random.shuffle(s)
            data, data_tag = zip(*s)

            # calc grads
        for h, sentence in enumerate(data):
            tag_sentence = data_tag[h]

            normalization_counts = lambda_rate * w

            sentence_features = get_sentence_features(sentence[:-1], tag_sentence, 'base', False)
            empirical_counts = np.sum(sentence_features, axis=0)

            expected_counts = np.zeros(feature_size, dtype=np.float64)

            # calculate p_array which contains p(y|x,w)
            p_array = np.zeros((len(sentence[2:-1]), T_size - 2), dtype=np.float64)
            for i, word in enumerate(sentence[:-1]):
                if i == 0 or i == 1:
                    continue
                feats = get_word_all_possible_tags_features(word, [tag_sentence[i - 2], tag_sentence[i - 1]], 'base', True)
                for j, tag in enumerate(T):
                    tag_feat = feats[j, :]
                    p_array[i - 2, j] = np.exp(np.sum(w[tag_feat])) / np.sum(np.exp(np.sum(w[feats], axis=1)))
                    # need to insert something that checks for inf or nan like:  np.isinf(a).any()

            # calc f_array which contains f(x,y)
            f_array = np.zeros((len(sentence[2:-1]), T_size - 2, 3), dtype=np.int64)
            for i, word in enumerate(sentence[:-1]):
                if i == 0 or i == 1:
                    continue
                f_array[i - 2, :, :] = get_word_all_possible_tags_features(word, [tag_sentence[i - 2], tag_sentence[i - 1]], 'base', True)

            # calc empirical counts
            for i, word in enumerate(sentence[:-1]):
                if i == 0 or i == 1:
                    continue
                for j, tag in enumerate(T):
                    expected_counts[f_array[i - 2, j, :]] += p_array[i - 2, j]

            w_grads = empirical_counts - expected_counts - normalization_counts

            # update weights
            w += w_grads * lr

            if h != 0 and h % 50 == 0:
                print('finished sentence {} after {} minutes'.format(h, round((time.time() - start_time) / 60, 1)))
                print('    expected_counts: ', np.sum(expected_counts), ', empirical_counts: ',
                      np.sum(empirical_counts), ', normalization_counts: ', np.sum(normalization_counts))
                print('    tot_grad: ', np.sum(w_grads * lr))
                print('    non zero params: ', np.count_nonzero(w))
                print('    non zero index: ', np.nonzero(w))
                print('    non zero values: ', w[np.nonzero(w)])

#        weightsL.append(copy.deepcopy(w))
        timeL.append(time.time() - start_time)
        print('finished epoch {} in {} min'.format(epoch + 1, (time.time() - start_time) / 60))

    # dump all results:
    with open(project_dir + '\\train_results\\' + resultsFn + '.log', 'wb') as f:
        pickle.dump([w, timeL, V, T_with_start, data, data_tag, test, test_tag], f)
