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
from scipy.sparse import csr_matrix


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

        if to_compress:
            return si_features
        else:
            return csr_matrix(si_features)


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


def train(max_epoch, data, data_tag, lambda_rate, lr):

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

        weightsL.append(copy.deepcopy(w))
        timeL.append(time.time() - start_time)
        print('finished epoch {} in {} min'.format(epoch + 1, (time.time() - start_time) / 60))

        # dump all results:
        with open(project_dir + '\\train_results\\' + resultsFn + '.log', 'wb') as f:
            pickle.dump([weightsL, timeL], f)

    return (w, timeL)


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


def viterby_predictor(corpus, w, prob_mat=None):
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
        prob_mat = np.zeros((V_COMP_size, T_size - 2, T_size - 2, T_size - 2))

    all_sentence_tags = []
    all_tagged_sentence = []

    for sentence in corpus:
        t0 = time.time()

        # init empty array of strings to save the tag for each word in the sentance
        sentence_len = len(sentence)
        sentence_tags = ['' for x in range(sentence_len)]

        # init dynamic matrix with size:
        # pi_matrix[k,t(i-1),t(i)] is the value of word number *k*, preciding tag u and t accordingly
        pi_matrix = np.zeros((sentence_len, T_size - 2, T_size - 2))

        # init back pointers matrix:
        # bp[k,t,u] is the tag index of word number *k-2*, following tag t and u accordingly
        bp = np.zeros((sentence_len, T_size - 2, T_size - 2), dtype=np.int)

        # holds all p(t(i),t(i-1),t(i-2))
        # prob_mat = np.zeros((T_size - 2,T_size - 2,T_size - 2))

        for k in range(0, sentence_len):  # for each word in the sentence
            # if havn't seen the word before - update the probebility matrix for all possible tagsL
            if k > 1 and not prob_mat[V_COMP.index(sentence[k]), 0, 0, 0].any():
                for u in T:  # for each t-1 possible tag
                    for t in T:  # for each t-2 possible tag:
                        prob_mat[V_COMP.index(sentence[k]), :, T.index(u),
                        T.index(t)] = calc_all_possible_tags_probabilities(sentence[k], u, t, weights)
            for current_tag in T:  # for each t possible tag

                if k == 0:
                    # at the first two words there is no meaning to the k-1 tag index. pi[k-1]
                    pi_matrix[k, 0, :] = 1 * calc_all_possible_tags_probabilities(sentence[k], '/*', '/*', weights)

                elif k == 1:
                    for u in T:  # for each t-1 possible tag
                        pi_matrix[k, T.index(u), :] = pi_matrix[
                                                          k - 1, 0, T.index(u)] * calc_all_possible_tags_probabilities(
                            sentence[k], u, '/*', weights)

                else:
                    for u in T:  # for each t-1 possible tag
                        # calculate pi value, and check if it exeeds the current max:
                        pi_values = pi_matrix[k - 1, :, T.index(u)] * prob_mat[V_COMP.index(sentence[k]),
                                                                      T.index(current_tag), T.index(u), :]
                        ind = np.argmax(pi_values)
                        if pi_values[ind] > pi_matrix[k, T.index(u), T.index(current_tag)]:
                            # update max:
                            pi_matrix[k, T.index(u), T.index(current_tag)] = pi_values[ind]

                            # update back pointers:
                            bp[k, T.index(u), T.index(current_tag)] = ind

        u_ind, curr_ind = np.unravel_index(pi_matrix[sentence_len - 1, :, :].argmax(),
                                           pi_matrix[sentence_len - 1, :, :].shape)
        sentence_tags[-2:] = [T[u_ind], T[curr_ind]]

        # extracting MEMM tags path from back pointers matrix:
        for i in range(sentence_len - 3, -1, -1):
            # calculate the idx of tag i in T db:
            # reminder - bp[k,t,u] is the tag of word *k-2*, following tag t and u accordingly
            k_tag_idx = bp[i + 2, T.index(sentence_tags[i + 1]), T.index(sentence_tags[i + 2])]

            # update the i-th tag to the list of tags
            sentence_tags[i] = T[k_tag_idx]

        # build tagged sentence:
        tagged_sentence = ''
        for i in range(sentence_len):
            tagged_sentence += (sentence[i] + '_')
            tagged_sentence += sentence_tags[i] + (' ')
        all_sentence_tags.append(sentence_tags)
        all_tagged_sentence.append(tagged_sentence)
        print(tagged_sentence, ' ,time: ', time.time() - t0)

    return (all_tagged_sentence, all_sentence_tags, prob_mat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('resultsFn', help='output results')
#    parser.add_argument('project_dir', help='output results')
#    parser.add_argument('input_path', help='output results')
    parser.add_argument('-max_epoch', type=int, default=4)
#    parser.add_argument('-lr', type=np.float64, default=0.2)
#    parser.add_argument('-lambda_rate', type=np.float64, default=0.1)
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

    lr_list = [0.01, 0.1, 1]
    lambda_rate_list = [0.01, 0.1, 1]

    input_path = project_dir + '\\data\\test_data_carmel.txt'

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

    corpus = test

    for lr in lr_list:
        for lambda_rate in lambda_rate_list:
            resultsFn_ = resultsFn + '_lr_' + str(lr) + '_lambda_rate_' + str(lambda_rate)
            (w, timeL) = train(max_epoch, data, data_tag, lambda_rate, lr)

            (all_tagged_sentence, all_sentence_tags, _) = viterby_predictor(corpus, w)

            line_accuracy = []
            tot_accuracy = 0
            if test_accuracy:
                tot_length = 0
                tot_correct = 0
                for i, tag_line in enumerate(all_sentence_tags):
                    res = np.sum([x == y for x, y in zip(tag_line, test_tag[i])])
                    line_accuracy.append(res / len(tag_line))
                    tot_length += len(tag_line)
                    tot_correct += res

                tot_accuracy = tot_correct / tot_length
                print("Total accuracy is: ", tot_accuracy)

            # dump all results:
            with open(project_dir + '\\train_results\\' + resultsFn_ + '_final.log', 'wb') as f:
                pickle.dump([lambda_rate, lr, all_sentence_tags, all_tagged_sentence, line_accuracy, tot_accuracy, w, timeL, V, T_with_start, data, data_tag, test, test_tag], f)
