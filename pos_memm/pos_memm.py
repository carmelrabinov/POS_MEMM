# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 01:02:59 2017
@author: carmelr
"""
import time
import os

try:
    import cPickle as pickle
except:
    import pickle

import pandas as pd
import numpy as np
from itertools import chain
from scipy.misc import logsumexp
from scipy.optimize import fmin_l_bfgs_b
from collections import Counter
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool



def analyze_results(pred_path, real_path, train_path, results_path):
    (_, _, _, tag_pred) = data_preprocessing(pred_path, 'test')
    (_, _, data_real, tag_real) = data_preprocessing(real_path, 'test')
    (V, _, _, _) = data_preprocessing(train_path, 'test')

    res = []
    for i, tag_line in enumerate(tag_pred):
        j = 0
        for pred, real in zip(tag_line, tag_real[i]):
            if pred != real:
                res.append([pred, real, data_real[i][j], j, i, int(data_real[i][j] in V)])
            j += 1

    df = pd.DataFrame(res, columns=['pred', 'real', 'word', 'word index', 'sentence index', 'unknown word'])
    df.to_csv(results_path, header=True, sep=',')


def softmax(numerator, denominator):
    denominator_max = np.max(denominator)
    denominator -= denominator_max
    numerator -= denominator_max
    return np.exp(numerator) / np.sum(np.exp(denominator))


def load_model(Fn):
    with open(Fn + '\\model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def data_preprocessing(data_path, mode):
    # mode can be 'train', 'test', or 'comp'
    data = []  # holds the data
    data_tag = []  # holds the taging
    f = open(data_path, "r")

    if mode == 'comp':
        for line in f:
            linesplit = []
            for word in line.split():
                linesplit.append(word)
            data.append(linesplit)

    else:
        for line in f:
            linesplit = []
            tagsplit = []
            for word in line.split():
                word, tag = word.split('_')
                linesplit.append(word)
                tagsplit.append(tag)
            data_tag.append(tagsplit)
            data.append(linesplit)

        if mode == 'train':
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

    return V, T, data, data_tag


class POS_MEMM:
    def __init__(self):
        self.weights = 0
        self.feature_size = 0
        self.data = []
        self.data_tag = []
        self.verbosity = 0
        self.mode = 'base'
        self.regularization = 0.1
        self.V = []
        self.T = []
        self.T_with_start = []
        self.T_size = 0
        self.V_size = 0
        self.T_dict = {}
        self.V_dict = {}
        self.T_with_start_dict = {}
        self.suffix_2 = {}
        self.suffix_3 = {}
        self.suffix_4 = {}
        self.prefix_2 = {}
        self.prefix_3 = {}
        self.prefix_4 = {}
        self.features_dict = {}
        self.features_dict_all_tags = {}
        self.features_count_dict = {}

    def build_features_dict(self):
        for h, sen in enumerate(self.data):
            tag_sen = self.data_tag[h]
            isfirst = False
            for i, word in enumerate(sen[:-1]):
                if i == 0 or i == 1:
                    continue
                if i==2:
                    isfirst = True
                if (word, tag_sen[i - 2], tag_sen[i - 1], tag_sen[i], isfirst) in self.features_count_dict:
                    self.features_count_dict[(word, tag_sen[i - 2], tag_sen[i - 1], tag_sen[i], isfirst)] += 1
                else:
                    self.features_dict[(word, tag_sen[i - 2], tag_sen[i - 1], tag_sen[i], isfirst)] = \
                        self.get_features(word, [tag_sen[i - 2], tag_sen[i - 1], tag_sen[i]], isfirst)                        
                    self.features_count_dict[(word, tag_sen[i - 2], tag_sen[i - 1], tag_sen[i], isfirst)] = 1
                for tag in self.T:
                    self.features_dict_all_tags[(word, tag_sen[i - 2], tag_sen[i - 1], tag, isfirst)] = \
                        self.get_features(word, [tag_sen[i - 2], tag_sen[i - 1], tag], isfirst)


#            for i, word in enumerate(sen[:-1]):
#                if i == 0 or i == 1:
#                    continue
#                if (word, tag_sen[i - 2], tag_sen[i - 1], tag_sen[i]) in self.features_count_dict:
#                    self.features_count_dict[(word, tag_sen[i - 2], tag_sen[i - 1], tag_sen[i])] += 1
#                else:
#                    self.features_dict[(word, tag_sen[i - 2], tag_sen[i - 1], tag_sen[i])] = \
#                        self.get_features(word, [tag_sen[i - 2], tag_sen[i - 1], tag_sen[i]])
#                    self.features_count_dict[(word, tag_sen[i - 2], tag_sen[i - 1], tag_sen[i])] = 1
#                for tag in self.T:
#                        self.features_dict_all_tags[(word, tag_sen[i - 2], tag_sen[i - 1], tag)] = \
#                                            self.get_features(word, [tag_sen[i - 2], tag_sen[i - 1], tag])

    def train(self, data_path, regularization=0.1, mode='base', spelling_threshold=10, verbosity=0, log_path=None):
        self.regularization = regularization
        self.mode = mode
        self.verbosity = verbosity

        (self.V, self.T_with_start, self.data, self.data_tag) = data_preprocessing(data_path, 'train')

        self.T = [x for x in self.T_with_start if (x != '/*' and x != '/STOP')]

        for i, tag in enumerate(self.T):
            self.T_dict[tag] = i
        for i, tag in enumerate(self.T_with_start):
            self.T_with_start_dict[tag] = i
        for i, tag in enumerate(self.V):
            self.V_dict[tag] = i

        self.T_size = len(self.T_with_start)
        self.V_size = len(self.V)

        self.init_spelling_dicts(spelling_threshold)

        self.feature_size = self.get_feature_size()
        self.weights = np.zeros(self.feature_size, dtype=np.float64)

        print('Building features...')
        self.build_features_dict()
        print('Done!')

        print('Start training...')
        t0 = time.time()
        optimal_params = fmin_l_bfgs_b(func=self.loss, x0=self.weights, fprime=self.loss_grads)
        training_time = (time.time() - t0) / 60
        print('Finished training with code: ', optimal_params[2]['warnflag'])
        print('Training time: {} minutes'.format(training_time))
        print('Iterations number: ', optimal_params[2]['nit'])
        print('Calls number: ', optimal_params[2]['funcalls'])

        if optimal_params[2]['warnflag']:
            print('Error in training:\n{}\\n'.format(optimal_params[2]['task']))
        else:
            self.weights = optimal_params[0]
            if log_path is not None:
                with open(log_path, 'a') as f:
                    f.writelines('\nTrain data:')
                    f.writelines('Number of sentences trained on: {}\n'.format(len(self.data)))
                    f.writelines('T size: {}\n'.format(self.T_size - 2))
                    f.writelines('V size: {}\n'.format(self.V_size))
                    f.writelines('Training time: {}\n'.format(training_time))
                    f.writelines('Iterations number: '.format(optimal_params[2]['nit']))
                    f.writelines('Calls number: '.format(optimal_params[2]['funcalls']))

        del self.data, self.data_tag

    def create_confusion_matrix(self, all_sentence_tags, test_tag, results_path):
        # init confusion matrix - [a,b] if we tagged word as "a" but the real tag is "b"
        confusion_matrix = np.zeros((self.T_size - 2, self.T_size - 2), dtype=np.int)

        max_num = 10
        if self.T_size - 2 < max_num:
            max_num = self.T_size - 2
        max_failure_matrix = np.zeros((max_num, self.T_size - 2), dtype=np.int)

        failure_dict = {}
        for tag in self.T:
            failure_dict[tag] = 0
        for sen_idx, sen_tags in enumerate(all_sentence_tags):
            for pred_tag, real_tag in zip(sen_tags, test_tag[sen_idx]):
                if pred_tag != real_tag:
                    failure_dict[real_tag] += 1
                confusion_matrix[self.T_dict[real_tag], self.T_dict[pred_tag]] += 1

        common_failure_tags = dict(Counter(failure_dict).most_common(max_num))

        i = 0
        for key in common_failure_tags.keys():
            common_failure_tags[key] = i
            i += 1

        if not os.path.exists(results_path):
            os.makedirs(results_path)
        print('Saving confusion matrix to {}'.format(results_path))

        confusion_matrix_fn = results_path + '\\confusion_matrix.csv'
        df = pd.DataFrame(confusion_matrix, index=self.T, columns=self.T)
        df.to_csv(confusion_matrix_fn, index=True, header=True, sep=',')

        mat = confusion_matrix
        np.fill_diagonal(mat, 0)
        for tag in common_failure_tags.keys():
            max_failure_matrix[common_failure_tags[tag]] = mat[self.T_dict[tag]]

        max_failure_matrix_fn = results_path + '\\max_failure_matrix.csv'
        df = pd.DataFrame(max_failure_matrix, index=list(common_failure_tags.keys()), columns=self.T)
        df.to_csv(max_failure_matrix_fn, index=True, header=True, sep=',')

    def calc_all_possible_tags_probabilities(self, xi, t1, t2, w, isfirst = False):
        """
        calculate probability p(ti|xi,w)
        :param xi: the word[i]
        :param t1: POS tag for word[i-1]
        :param t2: POS tag for word[i-2]
        :param w: weights vector
        :return: a list for all possible ti probabilities p(ti|xi,w) as float64
        """
        denominator = np.zeros(self.T_size - 2)
        for i, tag in enumerate(self.T):
            if xi[0].islower() and tag == 'NNP': #debug
                denominator[i] = 0 #debug
            else: #debug
                denominator[i] = np.sum(w[self.get_features(xi, [t2, t1, tag], isfirst)])
        return softmax(denominator, denominator)

#    def loss_grads(self, w):

    def calc_all_possible_tags_probabilities_train(self, xi, t1, t2, w):
        """
        calculate probability p(ti|xi,w)
        :param xi: the word[i]
        :param t1: POS tag for word[i-1]
        :param t2: POS tag for word[i-2]
        :param w: weights vector
        :return: a list for all possible ti probabilities p(ti|xi,w) as float64
        """
        denominator = np.zeros(self.T_size - 2)
        for i, tag in enumerate(self.T):
            denominator[i] = np.sum(w[self.features_dict_all_tags[(xi, t2, t1, tag)]])
        return softmax(denominator, denominator)

    def loss_grads(self, w):
        t0 = time.time()
        empirical_counts = np.zeros(self.feature_size, dtype=np.float64)
        expected_counts = np.zeros(self.feature_size, dtype=np.float64)

        # calculate normalization loss term
        normalization_counts = self.regularization * w * len(self.data)

        for key, features_inx in self.features_dict.items():
            (word, t2, t1, t, isfirst) = key
            count = self.features_count_dict[key]

            # calculate empirical loss term
            empirical_counts[features_inx] += count

            # calculate p(y|x,w) for word x and for all possible tag[i]
            p = self.calc_all_possible_tags_probabilities_train(word, t1, t2, w, isfirst)

            # calculate expected_loss term
            for j, tag in enumerate(self.T):
                tag_feat = self.features_dict_all_tags[(word, t2, t1, tag, isfirst)]
                expected_counts[tag_feat] += p[j] * count

        w_grads = empirical_counts - expected_counts - normalization_counts

        if self.verbosity:
            print('Done calculate grads in {}, max abs grad is {}, max abs w is {}'.format((time.time()-t0)/60, np.max(np.abs(w_grads)), np.max(np.abs(w))))
        return (-1) * w_grads

    def loss(self, w):
        t0 = time.time()
        empirical_loss = 0
        expected_loss = 0

        # calculate normalization loss term
        normalization_loss = (np.sum(np.square(w)) * self.regularization / 2) * len(self.data)

        for key, features_inx in self.features_dict.items():
            (word, t2, t1, t, isfirst) = key
            count = self.features_count_dict[key]

            # calculate empirical loss term
            empirical_loss += np.sum(w[features_inx]) * count

            # calculate expected_loss term
            exp_term = np.zeros(self.T_size - 2)
            for j, tag in enumerate(self.T):
                exp_term[j] = np.sum(w[self.features_dict_all_tags[(word, t2, t1, tag, isfirst)]])
            expected_loss += logsumexp(exp_term) * count

        loss_ = empirical_loss - expected_loss - normalization_loss
        if self.verbosity:
            print('Done calculate Loss in {} minutes, Loss is: {}'.format((time.time() - t0)/60, (-1) * loss_))
        return (-1) * loss_

    def get_feature_size(self):
        size_dict = {}
        size_dict['F100'] = self.V_size * self.T_size # represens word ant tag for all possible combinations
        size_dict['F103'] = self.T_size**3 # trigram of tags
        size_dict['F104'] = self.T_size**2 # bigram of tags
        if self.mode == 'complex':
            size_dict['F101_2'] = self.T_size*len(self.suffix_2) # all posible tags for each word in importnat suffix list
            size_dict['F101_3'] = self.T_size*len(self.suffix_3) # all posible tags for each word in importnat suffix list
            size_dict['F101_4'] = self.T_size*len(self.suffix_4) # all posible tags for each word in importnat suffix list
            size_dict['F102_2'] = self.T_size*len(self.prefix_2) # all posible tags for each word in importnat prefix list
            size_dict['F102_3'] = self.T_size*len(self.prefix_3) # all posible tags for each word in importnat prefix list
            size_dict['F102_4'] = self.T_size*len(self.prefix_4) # all posible tags for each word in importnat prefix list
            size_dict['F105'] = self.T_size # unigram of tag
            size_dict['F106'] = self.V_size * self.T_size # representes last word and current tag
            size_dict['F107'] = self.V_size * self.T_size # representes next word and current tag
            size_dict['G1'] = self.T_size  # is current word of the form <number-word> and tagged as t(i)
            size_dict['G2'] = self.T_size  # is current word starts with Upper case + the current tag
            size_dict['G3'] = self.T_size  # is Upper case and first word in sentance, with tag t(i)
            size_dict['G4'] = self.T_size  # is Upper case and *not* first word in sentance, with tag t(i)
            size_dict['G5'] = self.T_size  # is the all word in uppercase, with tag t(i)
            size_dict['G6'] = self.T_size  # is the all word in number, with tag t(i)

        return sum(size_dict.values())

    # def predict_parallel(self, corpus, verbosity=0):
    #
    #     print('Start predicting...')
    #     t0 = time.time()
    #     self.verbosity = verbosity
    #
    #     self.build_predict_prob_matrix(corpus)
    #
    #     pool = ThreadPool()     #Pool(processes=5, maxtasksperchild=3)
    #     res = pool.map(self.predict_sentence, corpus)
    #     pool.close()
    #     pool.join()
    #
    #     all_sentence_tags = [x[1] for x in res]
    #     all_tagged_sentence = [x[0] for x in res]
    #
    #     del self.prob_mat, self.V_COMP_dict
    #     self.verbosity = 0
    #     print('predict_parallel in time: {} min'.format((time.time() - t0)/60))
    #     return all_tagged_sentence, all_sentence_tags
    #
    # def build_word_prob_matrix(self, word):
    #
    #     prob_mat = np.zeros((self.T_size - 2, self.T_size - 2, self.T_size - 2))
    #
    #     for u in self.T:  # for each t-1 possible tag
    #         for t in self.T:  # for each t-2 possible tag:
    #             prob_mat[:, self.T_dict[u],
    #             self.T_dict[t]] = self.calc_all_possible_tags_probabilities(word, u, t, self.weights)
    #
    #     return word, prob_mat
    #
    # def build_predict_prob_matrix(self, corpus):
    #     print('Building prob matrix...')
    #     t0 = time.time()
    #     # init a list of singular words in the target corpus:
    #     V_COMP = sorted(list(set(chain(*corpus))))
    #     V_COMP_size = len(V_COMP)
    #     self.V_COMP_dict = {}
    #     for i, v in enumerate(V_COMP):
    #         self.V_COMP_dict[v] = i
    #
    #     # init probability matrix:
    #     # holds all p(word,t(i),t(i-1),t(i-2))
    #     self.prob_mat = np.zeros((V_COMP_size, self.T_size - 2, self.T_size - 2, self.T_size - 2))
    #
    #     pool = ThreadPool() #Pool(processes=2, maxtasksperchild=1)
    #     res = pool.map(self.build_word_prob_matrix, V_COMP)
    #     pool.close()
    #     pool.join()
    #
    #     for touple in res:
    #         self.prob_mat[self.V_COMP_dict[touple[0]], :, :, :] = touple[1]
    #     print('Build prob matrix in time: {} min'.format((time.time() - t0)/60))
    #
    # def predict_sentence(self, sentence):
    #     # init empty array of strings to save the tag for each word in the sentance
    #     sentence_len = len(sentence)
    #     sentence_tags = ['' for x in range(sentence_len)]
    #
    #     # init dynamic matrix with size:
    #     # pi_matrix[k,t(i-1),t(i)] is the value of word number k, preciding tag u and t accordingly
    #     pi_matrix = np.zeros((sentence_len, self.T_size - 2, self.T_size - 2))
    #
    #     # init back pointers matrix:
    #     # bp[k,t,u] is the tag index of word number k-2, following tag t and u accordingly
    #     bp = np.zeros((sentence_len, self.T_size - 2, self.T_size - 2), dtype=np.int)
    #
    #     for k in range(0, sentence_len):  # for each word in the sentence
    #
    #         for current_tag in self.T:  # for each t possible tag
    #
    #             if k == 0:
    #                 # at the first two words there is no meaning to the k-1 tag index. pi[k-1]
    #                 pi_matrix[k, 0, :] = 1 * self.calc_all_possible_tags_probabilities(sentence[k], '/*', '/*',
    #                                                                                    self.weights)
    #                 break
    #             elif k == 1:
    #                 for u in self.T:  # for each t-1 possible tag
    #                     pi_matrix[k, self.T_dict[u], :] = pi_matrix[k - 1, 0, self.T_dict[
    #                         u]] * self.calc_all_possible_tags_probabilities(sentence[k], u, '/*', self.weights)
    #                 break
    #             else:
    #                 for u in self.T:  # for each t-1 possible tag
    #
    #                     # calculate pi value, and check if it exeeds the current max:
    #                     pi_values = pi_matrix[k - 1, :, self.T_dict[u]] * self.prob_mat[self.V_COMP_dict[sentence[k]],
    #                                                                       self.T_dict[current_tag], self.T_dict[u],
    #                                                                       :]
    #                     ind = np.argmax(pi_values)
    #                     if pi_values[ind] > pi_matrix[k, self.T_dict[u], self.T_dict[current_tag]]:
    #                         # update max:
    #                         pi_matrix[k, self.T_dict[u], self.T_dict[current_tag]] = pi_values[ind]
    #
    #                         # update back pointers:
    #                         bp[k, self.T_dict[u], self.T_dict[current_tag]] = ind
    #
    #     u_ind, curr_ind = np.unravel_index(pi_matrix[sentence_len - 1, :, :].argmax(),
    #                                        pi_matrix[sentence_len - 1, :, :].shape)
    #     sentence_tags[-2:] = [self.T[u_ind], self.T[curr_ind]]
    #
    #     # extracting MEMM tags path from back pointers matrix:
    #     for i in range(sentence_len - 3, -1, -1):
    #         # calculate the idx of tag i in T db:
    #         # reminder - bp[k,t,u] is the tag of word k-2, following tag t and u accordingly
    #         k_tag_idx = bp[i + 2, self.T_dict[sentence_tags[i + 1]], self.T_dict[sentence_tags[i + 2]]]
    #
    #         # update the i-th tag to the list of tags
    #         sentence_tags[i] = self.T[k_tag_idx]
    #
    #     # build tagged sentence:
    #     tagged_sentence = ''
    #     for i in range(sentence_len):
    #         tagged_sentence += (sentence[i] + '_')
    #         tagged_sentence += sentence_tags[i] + (' ')
    #
    #     if self.verbosity:
    #         print(tagged_sentence)
    #
    #     return tagged_sentence, sentence_tags

    def predict(self, corpus, verbosity=0, log_path=None):
        """
        calculate the tags for the corpus
        :param corpus: a list of sentences (each sentence as a list of words) 
        :return: all_sentence_tags: a list of tagged sentences (each sentence as a list of tags
                 all_tagged_sentence: a list of tagged sentences in form of "word_tag"
        """
        self.verbosity = verbosity
        # case corpus is only 1 sentence:
        if len(corpus) == 1:
            corpus = [corpus]
        # init a list of singular words in the target corpus:
        V_COMP = sorted(list(set(chain(*corpus))))
        V_COMP_size = len(V_COMP)
        V_COMP_dict = {}
        for i, v in enumerate(V_COMP):
            V_COMP_dict[v] = i

        # init probability matrix:
        # holds all p(word,t(i),t(i-1),t(i-2))
        prob_mat = np.zeros((V_COMP_size, self.T_size - 2, self.T_size - 2, self.T_size - 2))

        all_sentence_tags = []
        all_tagged_sentence = []

        print('Start predicting...')
        t0 = time.time()
        for sentence in corpus:
            # init empty array of strings to save the tag for each word in the sentance
            sentence_len = len(sentence)
            sentence_tags = ['' for x in range(sentence_len)]

            # init dynamic matrix with size: 
            # pi_matrix[k,t(i-1),t(i)] is the value of word number k, preciding tag u and t accordingly
            pi_matrix = np.zeros((sentence_len, self.T_size - 2, self.T_size - 2))

            # init back pointers matrix:
            # bp[k,t,u] is the tag index of word number k-2, following tag t and u accordingly
            bp = np.zeros((sentence_len, self.T_size - 2, self.T_size - 2), dtype=np.int)

            for k in range(0, sentence_len):  # for each word in the sentence

                # if havn't seen the word before - update the probebility matrix for all possible tagsL
                if k > 1 and not prob_mat[V_COMP_dict[sentence[k]], 0, 0, 0].any():
                    for u in self.T:  # for each t-1 possible tag
                        for t in self.T:  # for each t-2 possible tag:
                            prob_mat[V_COMP_dict[sentence[k]], :, self.T_dict[u],
                            self.T_dict[t]] = self.calc_all_possible_tags_probabilities(sentence[k], u, t, self.weights)

                for current_tag in self.T:  # for each t possible tag

                    if k == 0:
                        # at the first two words there is no meaning to the k-1 tag index. pi[k-1]
                        pi_matrix[k, 0, :] = 1 * self.calc_all_possible_tags_probabilities(sentence[k], '/*', '/*',
                                                                                           self.weights)
                        break
                    elif k == 1:
                        for u in self.T:  # for each t-1 possible tag
                            pi_matrix[k, self.T_dict[u], :] = pi_matrix[k - 1, 0, self.T_dict[
                                u]] * self.calc_all_possible_tags_probabilities(sentence[k], u, '/*', self.weights)
                        break
                    else:
                        for u in self.T:  # for each t-1 possible tag
                            # calculate pi value, and check if it exeeds the current max:
                            pi_values = pi_matrix[k - 1, :, self.T_dict[u]] * prob_mat[V_COMP_dict[sentence[k]],
                                                                              self.T_dict[current_tag], self.T_dict[u],
                                                                              :]
                            ind = np.argmax(pi_values)
                            if pi_values[ind] > pi_matrix[k, self.T_dict[u], self.T_dict[current_tag]]:
                                # update max:
                                pi_matrix[k, self.T_dict[u], self.T_dict[current_tag]] = pi_values[ind]

                                # update back pointers:
                                bp[k, self.T_dict[u], self.T_dict[current_tag]] = ind

            u_ind, curr_ind = np.unravel_index(pi_matrix[sentence_len - 1, :, :].argmax(),
                                               pi_matrix[sentence_len - 1, :, :].shape)
            sentence_tags[-2:] = [self.T[u_ind], self.T[curr_ind]]

            # extracting MEMM tags path from back pointers matrix:
            for i in range(sentence_len - 3, -1, -1):
                # calculate the idx of tag i in T db:
                # reminder - bp[k,t,u] is the tag of word k-2, following tag t and u accordingly
                k_tag_idx = bp[i + 2, self.T_dict[sentence_tags[i + 1]], self.T_dict[sentence_tags[i + 2]]]

                # update the i-th tag to the list of tags
                sentence_tags[i] = self.T[k_tag_idx]

            # build tagged sentence:
            tagged_sentence = ''
            for i in range(sentence_len):
                tagged_sentence += (sentence[i] + '_')
                tagged_sentence += sentence_tags[i] + (' ')
            all_sentence_tags.append(sentence_tags)
            all_tagged_sentence.append(tagged_sentence)
            if self.verbosity:
                print(tagged_sentence)

        prediction_time = (time.time() - t0) / 60
        if log_path is not None:
            with open(log_path, 'a') as f:
                f.writelines('Prediction time: {}\n'.format(prediction_time))

        print('Done predicting in {} minutes'.format(prediction_time))
        return all_tagged_sentence, all_sentence_tags

    def get_features(self, word, tags, is_first=False):
        """
        :param word: the word
        :param tags: POS tags of the trigram as as a list <t(i-2), t(i-1), t(i)>
        :return: features - list of the features vector's indexes which are "true" 
        """
        features = []
        word_len = len(word)

        # base features:
        # 1 if xi = x and ti = t
        try:
            F100 = self.V_dict[word] * self.T_size + self.T_with_start_dict[tags[2]]
            features.append(F100)
        except:
            tmp = 0  # must do something in except
        F100_len = self.V_size * self.T_size

        # trigram feature - 1 if <t(i-2),t(is),t(i)> = <t1,t2,t3>
        F103 = self.T_with_start_dict[tags[2]] * (self.T_size ** 2) + self.T_with_start_dict[tags[1]] * self.T_size + \
               self.T_with_start_dict[tags[0]]
        features.append(F103 + F100_len)
        F103_len = F100_len + self.T_size ** 3

        # bigram feature - 1 if <t(i-1),t(i)> = <t1,t2>
        F104 = self.T_with_start_dict[tags[2]] * self.T_size + self.T_with_start_dict[tags[1]]
        features.append(F104 + F103_len)
        F104_len = F103_len + self.T_size ** 2

        # complex features:
        if self.mode == 'complex':

            # F101: suffix of length  2/3/4 which is in suffix lists && tag <t(i)>
            if word_len > 2 and word[-2:] in self.suffix_2.keys():
                F101_2 = self.suffix_2[word[-2:]] * self.T_size + self.T_with_start_dict[tags[2]]
                features.append(F101_2 + F104_len)
            F101_2_len = F104_len + self.T_size * len(self.suffix_2)
            if word_len > 3 and word[-3:] in self.suffix_3.keys():
                F101_3 = self.suffix_3[word[-3:]] * self.T_size + self.T_with_start_dict[tags[2]]
                features.append(F101_3 + F101_2_len)
            F101_3_len = F101_2_len + self.T_size * len(self.suffix_3)
            if word_len > 4 and word[-4:] in self.suffix_4.keys():
                F101_4 = self.suffix_4[word[-4:]] * self.T_size + self.T_with_start_dict[tags[2]]
                features.append(F101_4 + F101_3_len)
            F101_4_len = F101_3_len + self.T_size * len(self.suffix_4)
            F101_len = F101_4_len

            # F102: prefix of length 2/3/4 letters which is in prefix list && tag <t(i)>
            if word_len > 2 and word[:2] in self.prefix_2.keys():
                F102_2 = self.prefix_2[word[:2]] * self.T_size + self.T_with_start_dict[tags[2]]
                features.append(F102_2 + F101_len)
            F102_2_len = F101_len + self.T_size * len(self.prefix_2)

            if word_len > 3 and word[:3] in self.prefix_3.keys():
                F102_3 = self.prefix_3[word[:3]] * self.T_size + self.T_with_start_dict[tags[2]]
                features.append(F102_3 + F102_2_len)
            F102_3_len = F102_2_len + self.T_size * len(self.prefix_3)

            if word_len > 4 and word[:4] in self.prefix_4.keys():
                F102_4 = self.prefix_4[word[:4]] * self.T_size + self.T_with_start_dict[tags[2]]
                features.append(F102_4 + F102_3_len)
            F102_4_len = F102_3_len + self.T_size * len(self.prefix_4)
            F102_len = F102_4_len

            # F105: tag is <t(i)>
            F105 = self.T_with_start_dict[tags[2]]
            features.append(F105 + F102_len)
            F105_len = F102_len + self.T_size

            # F106: is last word w[i-1] and tag t[i]
            F106_len = F105_len + self.V_size * self.T_size            

            # F107: is next word w[i+1] and tag t[i]
            F107_len = F106_len + self.V_size * self.T_size   

            # G1 : is the current word in the form of <number-noun> (e.g 12-inch) and tagged as t(i)?
            number_check = word.partition('-')
            if number_check[0].isdigit() and number_check[1] == '-' and not number_check[2].isdigit():
                G1 = self.T_with_start_dict[tags[2]]
                features.append(G1 + F107_len)
            G1_len = F107_len + self.T_size

            # G2 : is the current word starts in Upper case and tag is t_i?
            if word[0].isupper() and word[0].isalpha():
                G2 = self.T_with_start_dict[tags[2]]
                features.append(G2 + G1_len)
            G2_len = G1_len + self.T_size
            
            # G3 : is word tagged as t(i), is Capital letter and first word in sentance?
            if word[0].isupper() and is_first:
                G3 = self.T_with_start_dict[tags[2]]
                features.append(G3 + G2_len)
            G3_len = G2_len + self.T_size
            
            # G4 : is  word tagged as t(i), it Capital letter and first word in sentance?
            if word[0].isupper() and not is_first:
                G4 = self.T_with_start_dict[tags[2]]
                features.append(G4 + G3_len)
            G4_len = G3_len + self.T_size 
            
            # G5 : is the all word in uppercase and tagged as t(i)?
            if word.isupper():
                G5 = self.T_with_start_dict[tags[2]]
                features.append(G5 + G4_len)
            G5_len = G4_len + self.T_size 
            
            # G6 : is all the word is digits (even if seperated by '.'), and taged as t(i)?
            number_check = word.partition('.')
            if number_check[0].isdigit():
                if number_check[2] == '' or number_check[2].isdigit():
                    G6 = self.T_with_start_dict[tags[2]]
                    features.append(G6 + G5_len)
            G6_len = G5_len + self.T_size 

        return features

    def init_spelling_dicts(self, threshold):
        """
        :param threshold: min count for suffix in the train set, in order to be taken for feature
        :return: save 3 lists for suffix at length 2-4 and 3 for prefix as "self" items
        """
        # declare histograms that will countain count in the corpus for each prefix and suffix
        histogram_suffix2 = {}
        histogram_suffix3 = {}
        histogram_suffix4 = {}

        histogram_prefix2 = {}
        histogram_prefix3 = {}
        histogram_prefix4 = {}

        # init the histograms to 0 for all relevant prefixes and suffixes
        for word in self.V:
            word_len = len(word)
            if word[-1].isdigit() or word[0].isdigit(): continue
            if word_len > 2:
                histogram_suffix2[word[-2:]] = 0
                histogram_prefix2[word[:2]] = 0
            if word_len > 3:
                histogram_suffix3[word[-3:]] = 0
                histogram_prefix3[word[:3]] = 0
            if word_len > 4:
                histogram_suffix4[word[-4:]] = 0
                histogram_prefix4[word[:4]] = 0

        # fill the histogram with count of each relevant prefix / suffix in the corpus
        for word in self.V:
            word_len = len(word)
            if word[-1].isdigit() or word[0].isdigit(): continue
            if word_len > 2:
                histogram_suffix2[word[-2:]] += 1
                histogram_prefix2[word[:2]] += 1
            if word_len > 3:
                histogram_suffix3[word[-3:]] += 1
                histogram_prefix3[word[:3]] += 1
            if word_len > 4:
                histogram_suffix4[word[-4:]] += 1
                histogram_prefix4[word[:4]] += 1

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

        prefix_2 = {}
        i = 0
        for key in histogram_prefix2.keys():
            if histogram_prefix2[key] > threshold:
                prefix_2[key] = i;
                i += 1

        prefix_3 = {}
        i = 0
        for key in histogram_prefix3.keys():
            if histogram_prefix3[key] > threshold:
                prefix_3[key] = i;
                i += 1

        prefix_4 = {}
        i = 0
        for key in histogram_prefix4.keys():
            if histogram_prefix4[key] > threshold:
                prefix_4[key] = i;
                i += 1

        self.suffix_2 = suffix_2
        self.suffix_3 = suffix_3
        self.suffix_4 = suffix_4
        self.prefix_2 = prefix_2
        self.prefix_3 = prefix_3
        self.prefix_4 = prefix_4


    def test(self, test_data_path, end=0, start=0, verbosity=0, save_results_to_file=None, log_path=None):
        self.verbosity = verbosity
        (_, _, test, test_tag) = data_preprocessing(test_data_path, 'test')

        if end:
            corpus = test[start:end]
            corpus_tag = test_tag[start:end]
        else:
            corpus = test[start:]
            corpus_tag = test_tag[start:]

        # run Viterbi algorithm
        (all_tagged_sentence, all_sentence_tags) = self.predict(corpus, verbosity=verbosity, log_path=log_path)

        tot_length = 0
        tot_correct = 0
        for i, tag_line in enumerate(all_sentence_tags):
            res = np.sum([x == y for x, y in zip(tag_line, corpus_tag[i])])
            tot_length += len(tag_line)
            tot_correct += res

        tot_accuracy = tot_correct / tot_length
        print("Total accuracy is: ", tot_accuracy)

        if save_results_to_file is not None:
            print('Saving predictions to {}'.format(save_results_to_file))
            # creating directory
            if not os.path.exists(save_results_to_file):
                os.makedirs(save_results_to_file)

            # creating and saving confusion matrix
            self.create_confusion_matrix(all_sentence_tags, test_tag, save_results_to_file)

            # saving predictions results as pkl
            with open(save_results_to_file + '\\predictions_logs.pkl', 'wb') as f:
                pickle.dump([all_sentence_tags, all_tagged_sentence, tot_accuracy], f)

            # saving predictions in comp format: word_tag
            with open(save_results_to_file + '\\predictions.txt', 'w') as f:
                for s in all_tagged_sentence:
                    f.writelines(s + '\n')

        if log_path is not None:
            with open(log_path, 'a') as f:
                self.data
                f.writelines('\nTest data:')
                f.writelines('Number of sentences tested on: {}\n'.format(len(corpus)))
                f.writelines('Prediction time: {}\n'.format(training_time))
                f.writelines('Total accuracy: {}\n'.format(tot_accuracy))

        return tot_accuracy, all_sentence_tags, all_tagged_sentence, test_tag

    def save_model(self, resultsfn):
        print('Saving model to {}'.format(resultsfn))
        # creating directory
        if not os.path.exists(resultsfn):
            os.makedirs(resultsfn)

        # dump all results:
        with open(resultsfn + '\\model.pkl', 'wb') as f:
            pickle.dump(self, f)