# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:44:28 2017

@author: carmelr
"""
import sys
import numpy as np
import time
import argparse
try: import cPickle as pickle
except: import pickle
import copy
from itertools import chain
import random
import os
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import logsumexp

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
    try: features_100 = V_dict[word] * T_size + T_with_start_dict[tags[2]]
    except: features_100 = None 
    features_100_length = V_size * T_size
    
    # trigram feature - 1 if <t(i-2),t(is),t(i)> = <t1,t2,t3>
    features_103 = T_with_start_dict[tags[2]] * (T_size ** 2) + T_with_start_dict[tags[1]] * T_size + T_with_start_dict[tags[0]]
    features_103_length = features_100_length + T_size**3
    
    # bigram feature - 1 if <t(i-1),t(i)> = <t1,t2>
    features_104 = T_with_start_dict[tags[2]] * T_size + T_with_start_dict[tags[1]]

    if features_100 == None:
        return [features_100_length + features_103, features_103_length + features_104]
    else:
        return [features_100, features_100_length + features_103, features_103_length + features_104]


def get_word_all_possible_tags_features(xi, th, mode):
    """
    computes all feature vectors of a given word with all possible POS tags
    :param xi: the word
    :param th: tag history vector <t(i-2),t(i-1)>
    :param mode: base / complex
    :param to_compress: True if return value as a list of index, False if return value as binary feature vector
    :return: feature matrix as a matrix of indexes or as a binary matrix, each row represent a possible POS tag
    """
    if mode == 'base':
        si_features = np.empty(shape=(T_size - 2, 3), dtype=np.int64)

        # iterate over all the words in the sentence besides START and STOP special signs
        for i, tag in enumerate(T):
            si_features[i, :] = get_base_features(xi, [th[0], th[1], tag])

        return si_features


def calc_all_possible_tags_probabilities(xi, t1, t2, w):
    """
    calculate probability p(ti|xi,w)
    :param xi: the word[i]
    :param t1: POS tag for word[i-1]
    :param t2: POS tag for word[i-2]
    :param w: weights vector
    :return: a list for all possible ti probabilities p(ti|xi,w) as float64
    """
    denominator = np.zeros(len(T))
    for i, tag in enumerate(T):
        denominator[i] = np.sum(w[get_base_features(xi, [t2, t1, tag])])
    return softmax(denominator,denominator)


def train(max_epoch, data, data_tag, lambda_rate, lr):

    # init
    feature_size = T_size ** 3 + T_size ** 2 + V_size * T_size
    w = np.zeros(feature_size, dtype=np.float64)

    start_time = time.time()

    for epoch in range(max_epoch):
        w_grads = loss_grads(w, data, data_tag, lambda_rate, feature_size, T, T_size)
        print('finished epoch {} in {} min'.format(epoch + 1, (time.time() - start_time) / 60))
        w += w_grads * lr

    return (w)


def train_online(max_epoch, data, data_tag, lambda_rate, lr):

    # init
    feature_size = T_size ** 3 + T_size ** 2 + V_size * T_size
    w = np.zeros(feature_size, dtype=np.float64)

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


            empirical_counts = np.zeros(feature_size, dtype=np.float64)
            for i, word in enumerate(sentence[:-1]):
                if i == 0 or i == 1:
                    continue
                empirical_counts[get_base_features(word, tag_sentence[i-2:i+1])] += 1

            expected_counts = np.zeros(feature_size, dtype=np.float64)

            # calculate p_array which contains p(y|x,w)
            p_array = np.zeros((len(sentence[2:-1]), T_size - 2), dtype=np.float64)
            for i, word in enumerate(sentence[:-1]):
                if i == 0 or i == 1:
                    continue
                feats = get_word_all_possible_tags_features(word, [tag_sentence[i - 2], tag_sentence[i - 1]], 'base')
                for j, tag in enumerate(T):
                    tag_feat = feats[j, :]
                    p_array[i - 2, j] = np.exp(np.sum(w[tag_feat])) / np.sum(np.exp(np.sum(w[feats], axis=1)))
                    # need to insert something that checks for inf or nan like:  np.isinf(a).any()

            # calc f_array which contains f(x,y)
            f_array = np.zeros((len(sentence[2:-1]), T_size - 2, 3), dtype=np.int64)
            for i, word in enumerate(sentence[:-1]):
                if i == 0 or i == 1:
                    continue
                f_array[i - 2, :, :] = get_word_all_possible_tags_features(word, [tag_sentence[i - 2], tag_sentence[i - 1]], 'base')

            # calc empirical counts
            for i, word in enumerate(sentence[:-1]):
                if i == 0 or i == 1:
                    continue
                for j, tag in enumerate(T):
                    expected_counts[f_array[i - 2, j, :]] += p_array[i - 2, j]

            w_grads = empirical_counts - expected_counts - normalization_counts

            # update weights
            w += w_grads * lr

        weightsL.append(copy.deepcopy(w))
        timeL.append(time.time() - start_time)
        print('finished epoch {} in {} min'.format(epoch + 1, (time.time() - start_time) / 60))

        # dump all results:
        with open(project_dir + '\\train_results\\' + resultsFn + '.log', 'wb') as f:
            pickle.dump([weightsL, timeL], f)

    return (w, timeL)


def loss(w, data, data_tag, lambda_rate, feature_size, T, T_size):
    
    loss_ = 0
    for h, sentence in enumerate(data):
        tag_sentence = data_tag[h]
        normalization_loss = np.sum(np.square(w)) * lambda_rate/2
    
        empirical_loss = 0
        expected_loss = 0
    
        for i, word in enumerate(sentence[:-1]):
            if i == 0 or i == 1:
                continue 
            features_inx = get_base_features(word, tag_sentence[i-2:i+1])
            empirical_loss += np.sum(w[features_inx])
            
            feats = get_word_all_possible_tags_features(word, [tag_sentence[i - 2], tag_sentence[i - 1]], 'base')
            expected_loss += logsumexp(np.sum(w[feats], axis=1))
        loss_ += empirical_loss - expected_loss - normalization_loss
    return loss_


def softmax(numerator, denominator):
    denominator_max = np.max(denominator)
    denominator -= denominator_max
    numerator -= denominator_max
    return np.exp(numerator) / np.sum(np.exp(denominator))


def loss_grads(w, data, data_tag, lambda_rate, feature_size, T, T_size):

    w_grads = np.zeros(feature_size, dtype=np.float64)
    print('max w: {}, {}'.format(np.max(w), np.sum(w > 20)))
    for h, sentence in enumerate(data):
        tag_sentence = data_tag[h]
    
        normalization_counts = lambda_rate * w
        empirical_counts = np.zeros(feature_size, dtype=np.float64)
        for i, word in enumerate(sentence[:-1]):
            if i == 0 or i == 1:
                continue          
            empirical_counts[get_base_features(word, tag_sentence[i-2:i+1])] += 1
    
        expected_counts = np.zeros(feature_size, dtype=np.float64)
    
        # calculate p_array which contains p(y|x,w)
        p_array = np.zeros((len(sentence[2:-1]), T_size - 2), dtype=np.float64)
        for i, word in enumerate(sentence[:-1]):
            if i == 0 or i == 1:
                continue
            feats = get_word_all_possible_tags_features(word, [tag_sentence[i - 2], tag_sentence[i - 1]], 'base')
            for j, tag in enumerate(T):
                tag_feat = feats[j, :]
                # denominator = np.sum(w[feats], axis=1)
                # numerator = np.sum(w[tag_feat])
                p_array[i - 2, j] = softmax(np.sum(w[tag_feat]), np.sum(w[feats], axis=1))
                # need to insert something that checks for inf or nan like:  np.isinf(a).any()

        # calc f_array which contains f(x,y)
        f_array = np.zeros((len(sentence[2:-1]), T_size - 2, 3), dtype=np.int64)
        for i, word in enumerate(sentence[:-1]):
            if i == 0 or i == 1:
                continue
            f_array[i - 2, :, :] = get_word_all_possible_tags_features(word, [tag_sentence[i - 2], tag_sentence[i - 1]], 'base')
    
        # calc empirical counts
        for i, word in enumerate(sentence[:-1]):
            if i == 0 or i == 1:
                continue
            for j, tag in enumerate(T):
                expected_counts[f_array[i - 2, j, :]] += p_array[i - 2, j]
    
        w_grads += empirical_counts - expected_counts - normalization_counts
    print('max_grads w: {}, {}'.format(np.max(w_grads), np.sum(w_grads > 0.1)))
    # np.clip(w_grads, None, 400, out=w_grads)
    # print('max_cliped_grads w: {}, {}'.format(np.max(w_grads), np.sum(w_grads > 20)))
    return w_grads


def train_bfgs(data, data_tag, lambda_rate, T, T_size):
    feature_size = T_size ** 3 + T_size ** 2 + V_size * T_size
    w0 = np.zeros(feature_size, dtype=np.float64)
    return fmin_l_bfgs_b(func=loss, x0=w0, fprime=loss_grads, args=(data, data_tag, lambda_rate, feature_size, T, T_size))


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
    weights = w
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
                  
        for k in range (0,sentence_len): # for each word in the sentence

            # if havn't seen the word before - update the probebility matrix for all possible tagsL
            if k > 1 and not prob_mat[V_COMP.index(sentence[k]),0,0,0].any():
                for u in T: # for each t-1 possible tag
                    for t in T: # for each t-2 possible tag:
                        prob_mat[V_COMP.index(sentence[k]),:, T_dict[u], T_dict[t]] = calc_all_possible_tags_probabilities(sentence[k], u, t, weights)

            for current_tag in T: # for each t possible tag
                
                if k == 0:
                    # at the first two words there is no meaning to the k-1 tag index. pi[k-1]
                    pi_matrix[k, 0, :] = 1 * calc_all_possible_tags_probabilities(sentence[k], '/*', '/*', weights)
                    break
                elif k == 1:
                    for u in T: # for each t-1 possible tag
                        pi_matrix[k, T_dict[u], :] = pi_matrix[k - 1, 0, T_dict[u]] * calc_all_possible_tags_probabilities(sentence[k], u, '/*', weights)
                    break
                else:      
                    for u in T: # for each t-1 possible tag
                        # calculate pi value, and check if it exeeds the current max:
                        pi_values = pi_matrix[k-1, :, T_dict[u]] * prob_mat[V_COMP.index(sentence[k]),T_dict[current_tag], T_dict[u], :]
                        ind = np.argmax(pi_values)
                        if pi_values[ind] > pi_matrix[k, T_dict[u], T_dict[current_tag]]:
                            
                            # update max:
                            pi_matrix[k, T_dict[u], T_dict[current_tag]] = pi_values[ind]
                            
                            # update back pointers:
                            bp[k, T_dict[u], T_dict[current_tag]] = ind

        u_ind, curr_ind = np.unravel_index(pi_matrix[sentence_len-1,:,:].argmax(), pi_matrix[sentence_len-1,:,:].shape)
        sentence_tags[-2:] = [T[u_ind], T[curr_ind]]

        # extracting MEMM tags path from back pointers matrix:
        for i in range(sentence_len-3,-1,-1):
            # calculate the idx of tag i in T db:
            # reminder - bp[k,t,u] is the tag of word *k-2*, following tag t and u accordingly
            k_tag_idx = bp[i + 2, T_dict[sentence_tags[i + 1]], T_dict[sentence_tags[i + 2]]]
    
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
    parser.add_argument('resultsFn', help='output results')
    parser.add_argument('-max_epoch', type=int, default=4)
    parser.add_argument('-lr', type=np.float64, default=0.1)
    parser.add_argument('-lambda_rate', type=np.float64, default=0.5)
    parser.add_argument('-noShuffle', action='store_false')
    parser.add_argument('-toy', action='store_true')
    parser.add_argument('-input_path', type=str, default=None)
    parser.add_argument('-end', type=int, default=0)
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
    if toy:
        data_path = project_dir + '\\data\\carmel_test3.txt'
        test_path = project_dir + '\\data\\carmel_test3.txt'


    (V, T_with_start, data, data_tag, test, test_tag) = data_preprocessing(data_path, test_path)

    V_size = len(V)
    T_size = len(T_with_start)
    T = [x for x in T_with_start if (x != '/*' and x != '/STOP')]

    T_dict = {}
    for i,tag in enumerate(T):
        T_dict[tag] = i
    T_with_start_dict = {}
    for i,tag in enumerate(T_with_start):
        T_with_start_dict[tag] = i
    V_dict = {}
    for i,tag in enumerate(V):
        V_dict[tag] = i  

    
    t0 = time.time()
    lambda_rate = 0.1
    w_opt = train(100, data, data_tag, lambda_rate, 0.005)
    # optimal_params = train_bfgs(data, data_tag, lambda_rate, T, T_size)
    # if optimal_params[2]['warnflag']:
    #     print('Error in training')
    #     exit()
    # else:
    print('Done in time: {}'.format(time.time() - t0))
    # w_opt = optimal_params[0]
    print(w_opt)

    results_path = project_dir + '\\train_results\\' + resultsFn
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # dump all results:
    with open(results_path +'\\weights_and_corpus.pkl', 'wb') as f:
        pickle.dump([w_opt, V, T_with_start, data, data_tag, test, test_tag], f)

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
    (all_tagged_sentence, all_sentence_tags, _) = viterby_predictor(corpus, w_opt)

    line_accuracy = []
    tot_accuracy = 0
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
    with open(results_path +'\\predictions_logs.pkl', 'wb') as f:
        pickle.dump([all_sentence_tags, all_tagged_sentence, line_accuracy, tot_accuracy], f)
    with open(results_path +'\\predictions.txt', 'w') as f2:
        for s in all_tagged_sentence:
            f2.writelines(s+'\n')
