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
from itertools import chain
import os
from scipy.optimize import fmin_l_bfgs_b, fmin_bfgs, fmin_tnc
from scipy.misc import logsumexp

def data_preprocessing(data_path, test_path, pred_path):
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
    print('Data preprocessing...')
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

    # vocabulary
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

    V_test = sorted(list(set(chain(*test))))

    comp = []  # holds the data
    f = open(comp_path, "r")
    for line in f:
        linesplit = []
        for word in line.split():
            linesplit.append(word)
        comp.append(linesplit)

    print('Done preprocessing!')
    return (V, T, data, data_tag, test, test_tag, comp, V_test)


def init_suffix_dicts(Vocabulary, threshold):

        #declare histograms that will countain count in the corpus for each prefix and suffix
        histogram_suffix2 = {}
        histogram_suffix3 = {}
        histogram_suffix4 = {}

        histogram_prefix2 = {}
        histogram_prefix3 = {}
        histogram_prefix4 = {}

        #init the histograms to 0 for all relevant prefixes and suffixes
        for word in Vocabulary:
            if word[-1].isdigit(): continue
            if (len(word) > 1):
                histogram_suffix2[word[-2:]] = 0
                histogram_prefix2[word[:2]] = 0
            if (len(word) > 2):
                histogram_suffix3[word[-3:]] = 0
                histogram_prefix3[word[:3]] = 0
            if (len(word) > 3):
                histogram_suffix4[word[-4:]] = 0
                histogram_prefix4[word[:4]] = 0

        for word in Vocabulary:
            if word[-1].isdigit(): continue
            if len(word) > 1:
                histogram_suffix2[word[-2:]] += 1

            if len(word) > 2:
                histogram_suffix3[word[-3:]] += 1

            if len(word) > 3:
                histogram_suffix4[word[-4:]] += 1

        suffix_2 = {}
        i = 0
        for key in histogram_suffix2.keys():
            if histogram_suffix2[key] > threshold:
                suffix_2[key] = i;
                i += 1

        i = 0
        suffix_3 = {}
        for key in histogram_suffix3.keys():
            if histogram_suffix3[key] > threshold:
                suffix_3[key] = i
                i += 1

        i = 0
        suffix_4 = {}
        for key in histogram_suffix4.keys():
            if histogram_suffix4[key] > threshold:
                suffix_4[key] = i;
                i += 1

        return (suffix_2, suffix_3, suffix_4)

def get_feature_vector_size(mode):
    size_dict = {}
    size_dict['F100'] = V_size * T_size # represens word ant tag for all possible combinations
    size_dict['F103'] = T_size**3 # trigram of tags
    size_dict['F104'] = T_size**2 # bigram of tags
    if mode == 'complex':
        size_dict['F101_2'] = T_size*len(suffix_2) # all posible tags for each word in importnat suffix list
        size_dict['F101_3'] = T_size*len(suffix_3) # all posible tags for each word in importnat suffix list
        size_dict['F101_4'] = T_size*len(suffix_4) # all posible tags for each word in importnat suffix list
        size_dict['F105'] = T_size # unigram of tag
        size_dict['G1'] = T_size  # is current word a number + the current tag
        size_dict['G2'] = T_size  # is current word starts with Upper case + the current tag

    return sum(size_dict.values())

def get_features(word, tags, mode):
    """
    :param word: the word
    :param tags: POS tags of the trigram as as a list <t(i-2), t(i-1), t(i)>
    :return: features - a binary feature vector
    """
    features = []

    # 1 if xi = x and ti = t
    try:
        F100 = V_dict[word] * T_size + T_with_start_dict[tags[2]]
        features.append(F100)
    except:
        tmp = 0
    F100_len = V_size * T_size

    # trigram feature - 1 if <t(i-2),t(is),t(i)> = <t1,t2,t3>
    F103 = T_with_start_dict[tags[2]] * (T_size ** 2) + T_with_start_dict[tags[1]] * T_size + T_with_start_dict[tags[0]]
    features.append(F103 + F100_len)
    F103_len = F100_len + T_size**3

    # bigram feature - 1 if <t(i-1),t(i)> = <t1,t2>
    F104 = T_with_start_dict[tags[2]] * T_size + T_with_start_dict[tags[1]]
    features.append(F104 + F103_len)
    F104_len = F103_len + T_size**2

    if mode=='complex':

        # F101: suffix in last 2/3/4 letters suffix lists && tag <t(i)>
        if len(word) > 1 and word[-2:] in suffix_2.keys():
            F101_2 = suffix_2[word[-2:]]*T_size + T_with_start_dict[tags[2]]
            features.append(F101_2 + F104_len)

#            #debug:
#            if F101_2 + F104_len > feature_size:
#                print('F101_2: ', F101_2 + F104_len)

        F101_2_len = F104_len + T_size*len(suffix_2)
        if len(word) > 2 and word[-3:] in suffix_3.keys():
            F101_3 = suffix_3[word[-3:]]*T_size + T_with_start_dict[tags[2]]
            features.append(F101_3 + F101_2_len)
#            #debug:
#            if F101_3 + F101_2_len > feature_size:
#                print('F101_3: ', F101_3 + F101_2_len)
        F101_3_len = F101_2_len + T_size*len(suffix_3)
        if len(word) > 3 and word[-4:] in suffix_4.keys():
            F101_4 = suffix_4[word[-4:]]*T_size + T_with_start_dict[tags[2]]
            features.append(F101_4 + F101_3_len)
            #debug:
#            if F101_4 + F101_3_len > feature_size:
#                print('F101_4: ', F101_4 + F101_3_len)
        F101_4_len = F101_3_len + T_size*len(suffix_4)

        F101_len = F101_4_len

        # F102: suffix in last 2/3/4 letters suffix lists && tag <t(i)>

        F102_len = F101_len + 0#TODO

        # F105: tag is <t(i)>
        F105 = T_with_start_dict[tags[2]]
        features.append(F105 + F102_len)

        #debug:
#        if F105 + F102_len > feature_size:
#            print('F101_5: ', F105 + F102_len)

        F105_len = F102_len + T_size

        # F106: 
        F106_len = F105_len + 0

        # F107:
        F107_len = F106_len + 0

        # G1 : is the cuurent word a number and tag is t_i?
        if word[0].isdigit():
            G1 = T_with_start_dict[tags[2]]
            features.append(G1 + F107_len)

            #debug:
#            if G1
        G1_len = F107_len + T_size

        # G2 : is the cuurent word starts in Upper case and tag is t_i?
        if word[0].isupper() and word[0].isalpha():
            G2 = T_with_start_dict[tags[2]]
            features.append(G2 + G1_len)
        G2_len = G1_len + T_size

        # G3 : is the cuurent word starts in Upper case and tag is t_i?

        ##debug - all:




    return features


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
        denominator[i] = np.sum(w[get_features(xi, [t2, t1, tag], mode)])
    return softmax(denominator,denominator)


def loss(w, data, data_tag, lambda_rate, feature_size, T, T_size):

    loss_ = 0
    for h, sentence in enumerate(data):
        tag_sentence = data_tag[h]
        empirical_loss = 0
        expected_loss = 0

        # calculate normalization loss term
        normalization_loss = np.sum(np.square(w)) * lambda_rate/2

        for i, word in enumerate(sentence[:-1]):
            if i == 0 or i == 1:
                continue
            # calculate empirical loss term
            features_inx = get_features(word, tag_sentence[i-2:i+1], mode)
            empirical_loss += np.sum(w[features_inx])

            # calculate expected_loss term
            exp_term = np.zeros(len(T))
            for j, tag in enumerate(T):
                exp_term[j] = np.sum(w[get_features(word, [tag_sentence[i-2], tag_sentence[i-1], tag], mode)])
            expected_loss += logsumexp(exp_term)

        loss_ += empirical_loss - expected_loss - normalization_loss
    print('Loss is: {}'.format((-1)*loss_))
    return (-1)*loss_


def softmax(numerator, denominator):
    denominator_max = np.max(denominator)
    denominator -= denominator_max
    numerator -= denominator_max
    return np.exp(numerator) / np.sum(np.exp(denominator))


def loss_grads(w, data, data_tag, lambda_rate, feature_size, T, T_size):

    t0 = time.time()
    w_grads = np.zeros(feature_size, dtype=np.float64)
    for h, sentence in enumerate(data):
        tag_sentence = data_tag[h]

    # calculate weights normalization term
        normalization_counts = lambda_rate * w

    # calculate empirical counts term
        empirical_counts = np.zeros(feature_size, dtype=np.float64)
        for i, word in enumerate(sentence[:-1]):
            if i == 0 or i == 1:
                continue
            empirical_counts[get_features(word, tag_sentence[i-2:i+1], mode)] += 1

    # calculate expected counts term
        expected_counts = np.zeros(feature_size, dtype=np.float64)

        # go over all words in sentence
        for i, word in enumerate(sentence[:-1]):
            # 2 first words are /* /* start symbols
            if i == 0 or i == 1:
                continue

            # calculate p(y|x,w) for word x and for all possible tag[i]
            p = calc_all_possible_tags_probabilities(word, tag_sentence[i - 1], tag_sentence[i - 2], w)

            for j, tag in enumerate(T):
                # take features indexes for tag[i] = j
                tag_feat = get_features(word, [tag_sentence[i - 2], tag_sentence[i - 1], tag], mode)

                # add p[j] to all features indexes that are equal to 1 (f_array[i - 2, j, :] is a list of indexes)
                expected_counts[tag_feat] += p[j]
            # TODO: need to insert something that checks for inf or nan like:  np.isinf(a).any()

        # update grads for the sentence
        w_grads += empirical_counts - expected_counts - normalization_counts
    print('Done calculate grads in {}, max abs grad is {}, max abs w is {}'.format((time.time()-t0)/60, np.max(np.abs(w_grads)), np.max(np.abs(w))))
    return (-1)*w_grads


def train_bfgs(data, data_tag, lambda_rate, T, T_size):
    w0 = np.zeros(feature_size, dtype=np.float64)
    return fmin_l_bfgs_b(loss, x0=w0, fprime=loss_grads, args=(data, data_tag, lambda_rate, feature_size, T, T_size))


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
    parser.add_argument('-lambda_rate', type=np.float64, default=0.5)
    parser.add_argument('-toy', action='store_true')
    parser.add_argument('-input_path', type=str, default=None)
    parser.add_argument('-end', type=int, default=0)
    parser.add_argument('-suff_threshold', type=int, default=10)
    parser.add_argument('-pref_threshold', type=int, default=10)
    parser.add_argument('-mode', type=str, default='complex')

    parser.parse_args(namespace=sys.modules['__main__'])

#    ##############
#    max_epoch = 40
#    lr = 0.2
#    lambda_rate = 0.1
#    noShuffle = True
#    test = True
#    ##############

    project_dir = 'D:\\TECHNION\\NLP\\part_of_speech_taging_MEMM'
    # project_dir = 'C:\\Users\\amirli\\Desktop\\amir\\part_of_speech_taging_MEMM-carmel\\POS_MEMM'
#    project_dir = os.path.dirname(os.path.realpath('__file__'))
    test_path = project_dir + '\\data\\test.wtag'
    comp_path = project_dir + '\\data\\comp.words'
    data_path = project_dir + '\\data\\train.wtag'

    # run on very small corpus to test the algorithm
    if toy:
        data_path = project_dir + '\\data\\carmel_test3.txt'
        test_path = project_dir + '\\data\\carmel_test3.txt'
        resultsFn = 'test1'
        mode = 'complex'
        end = 10
        # data_path = project_dir + '\\data\\debug.wtag'
        # test_path = project_dir + '\\data\\debug.wtag'

    (V, T_with_start, data, data_tag, test, test_tag, comp, V_dev) = data_preprocessing(data_path, test_path, comp_path)
    V_Total = set(V+V_dev)
    # (suffix_2, suffix_3, suffix_4) = init_suffix_dicts(V_Total, suff_threshold)

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

    feature_size = get_feature_vector_size(mode)

    t0 = time.time()
    lambda_rate = 0.1

    optimal_params = train_bfgs(data, data_tag, lambda_rate, T, T_size)
    print('Finished training with code: {}\nIterations number: {}\nCalls number: {}'.format(optimal_params[2]['warnflag'],
                                                                                   optimal_params[2]['nit'],
                                                                                   optimal_params[2]['funcalls']))
    if optimal_params[2]['warnflag']:
        print('Error in training:\n{}'.format(optimal_params[2]['task']))
        exit()
    else:
        print('Done in time: {}'.format(time.time() - t0))
        w_opt = optimal_params[0]

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