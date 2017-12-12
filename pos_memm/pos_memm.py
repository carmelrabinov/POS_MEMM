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
                if j > 0:
                    prev_word = data_real[i][j - 1]
                    prev_tag = tag_real[i][j - 1]
                else:
                    prev_word = '/*'
                    prev_tag = '/*'
                if j < len(tag_line) - 1:
                    next_word = data_real[i][j + 1]
                    next_tag = tag_real[i][j + 1]
                else:
                    next_word = '/stop'
                    next_tag = '/stop'
                res.append([pred, prev_tag, real, next_tag, prev_word, data_real[i][j], next_word, j, i,
                            1 - int(data_real[i][j] in V)])
            j += 1

    df = pd.DataFrame(res, columns=['pred', 'prev tag', 'real', 'next tag', 'previous word', 'word', 'next word',
                                    'word index', 'sentence index', 'unknown word'])
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
        self.V_count = {}  # holds how many times a word is in the train data
        self.V_tags_dict = {}  # for each word in V hold all possible tags seen in train
        self.T_with_start_dict = {}
        self.suffix_1 = {}
        self.suffix_2 = {}
        self.suffix_3 = {}
        self.suffix_4 = {}
        self.prefix_1 = {}
        self.prefix_2 = {}
        self.prefix_3 = {}
        self.prefix_4 = {}
        self.features_dict = {}
        self.features_dict_all_tags = {}
        self.features_count_dict = {}
        self.spelling_threshold = 0
        self.word_count_threshold = 0
        self.train_or_predict_mode = 'train'
        self.use_106_107 = False

    def build_features_dict(self):
        for h, sen in enumerate(self.data):
            tag_sen = self.data_tag[h]
            isfirst = False
            for i, word in enumerate(sen[:-1]):
                if self.use_106_107:
                    words = tuple(sen[i - 1:i + 2])
                else:
                    words = ('/106_107', sen[i], '/106_107')

                if i == 0 or i == 1:
                    continue
                if i == 2:
                    isfirst = True
                if (words, tag_sen[i - 2], tag_sen[i - 1], tag_sen[i], isfirst) in self.features_count_dict:
                    self.features_count_dict[(words, tag_sen[i - 2], tag_sen[i - 1], tag_sen[i], isfirst)] += 1
                else:
                    feature = self.get_features(words, [tag_sen[i - 2], tag_sen[i - 1], tag_sen[i]], isfirst)
                    self.features_dict[(words, tag_sen[i - 2], tag_sen[i - 1], tag_sen[i], isfirst)] = feature

                    self.features_count_dict[(words, tag_sen[i - 2], tag_sen[i - 1], tag_sen[i], isfirst)] = 1

                feature_all_tags = []
                for tag in self.T:
                    feature_all_tags.append(self.get_features(words, [tag_sen[i - 2], tag_sen[i - 1], tag], isfirst))
                self.features_dict_all_tags[(words, tag_sen[i - 2], tag_sen[i - 1], isfirst)] = feature_all_tags

    def train(self, data_path, regularization=0.1, mode='base', spelling_threshold=8,
              word_count_threshold=3, use_106_107=False, verbosity=0, log_path=None):

        self.regularization = regularization
        self.mode = mode
        self.train_or_predict_mode = 'train'
        self.use_106_107 = use_106_107
        self.verbosity = verbosity

        (self.V, self.T_with_start, self.data, self.data_tag) = data_preprocessing(data_path, 'train')

        print('Initializing internal dictionaries...')
        self.T = [x for x in self.T_with_start if (x != '/*' and x != '/STOP')]

        for i, tag in enumerate(self.T):
            self.T_dict[tag] = i
        for i, tag in enumerate(self.T_with_start):
            self.T_with_start_dict[tag] = i
        for i, tag in enumerate(self.V):
            self.V_dict[tag] = i

        self.T_size = len(self.T_with_start)
        self.V_size = len(self.V)

        self.V_count = Counter(chain(*self.data))
        self.word_count_threshold = word_count_threshold

        # for word in self.V:
        #     all_tags = [self.data_tag[i][j] for i, row in enumerate(self.data)
        #                 for j, elem in enumerate(row) if elem == word]
        #     self.V_tags_dict[word] = list(set(all_tags))

        self.spelling_threshold = spelling_threshold
        self.init_spelling_dicts()

        self.feature_size = self.get_feature_size()
        self.weights = np.zeros(self.feature_size, dtype=np.float64)
        print('Done!')

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
                    f.writelines('\nTrain data:\n')
                    f.writelines('Number of sentences trained on: {}\n'.format(len(self.data)))
                    f.writelines('T size: {}\n'.format(self.T_size - 2))
                    f.writelines('V size: {}\n'.format(self.V_size))
                    f.writelines('Use 106 and 107 features: {}\n'.format(self.use_106_107))
                    f.writelines('Training time: {}\n'.format(training_time))
                    f.writelines('Iterations number: {}\n'.format(optimal_params[2]['nit']))
                    f.writelines('Calls number: {}\n'.format(optimal_params[2]['funcalls']))

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

    def calc_all_possible_tags_probabilities(self, words, t1, t2, w, isfirst=False):
        """
        calculate probability p(ti|xi,w)
        :param words: the words[i-1,i,i+1]
        :param t1: POS tag for word[i-1]
        :param t2: POS tag for word[i-2]
        :param w: weights vector
        :return: a list for all possible ti probabilities p(ti|xi,w) as float64
        """
        feature_all_tags = []
        for tag in self.T:
            feature_all_tags.append(self.get_features(words, [t2, t1, tag], isfirst))
        denominator = np.sum(w[feature_all_tags], axis=1)

        # denominator = np.zeros(self.T_size - 2)
        # for i, tag in enumerate(self.T):
        #     denominator[i] = np.sum(w[self.get_features(words, [t2, t1, tag], isfirst)])
        return softmax(denominator, denominator)

    def calc_all_possible_tags_probabilities_pred(self, words, w, isfirst=False):
        """
        calculate probability p(ti|xi,w)
        :param words: the words[i-1,i,i+1]
        :param t1: POS tag for word[i-1]
        :param t2: POS tag for word[i-2]
        :param w: weights vector
        :return: a list for all possible ti probabilities p(ti|xi,w) as float64
        """
        feature_all_t = []
        for t in self.T:  # for each t-2 possible tag
            feature_all_u = []
            for u in self.T:  # for each t-1 possible tag:
                if (words, t, u, isfirst) in self.features_dict_all_tags:
                    feature_all_u.append(self.features_dict_all_tags[(words, t, u, isfirst)])
                else:
                    feature_all_tags = []
                    for tag in self.T:
                        feature_all_tags.append(self.get_features(words, [u, t, tag], isfirst))
                    feature_all_u.append(feature_all_tags)
            feature_all_t.append(feature_all_u)
        all_tags = np.array(np.sum(w[feature_all_t], axis=3))
        denominator = np.sum(np.exp(all_tags), axis=2)
        for i, tag in enumerate(self.T):
            val = all_tags[:, :, i]
            all_tags[:, :, i] = np.exp(val) / denominator
        return np.transpose(all_tags)

    def calc_all_possible_tags_probabilities_train(self, words, t1, t2, w, isfirst=False):
        """
        calculate probability p(ti|xi,w)
        :param words: the words[i-1,i,i+1]
        :param t1: POS tag for word[i-1]
        :param t2: POS tag for word[i-2]
        :param w: weights vector
        :return: a list for all possible ti probabilities p(ti|xi,w) as float64
        """
        # denominator = np.zeros(self.T_size - 2)
        # for i, tag in enumerate(self.T):
        #     denominator[i] = np.sum(w[self.features_dict_all_tags[(words, t2, t1, tag, isfirst)]])

        denominator = np.sum(w[self.features_dict_all_tags[(words, t2, t1, isfirst)]], axis=1)
        return softmax(denominator, denominator)

    def loss_grads(self, w):
        t0 = time.time()
        empirical_counts = np.zeros(self.feature_size, dtype=np.float64)
        expected_counts = np.zeros(self.feature_size, dtype=np.float64)

        # calculate normalization loss term
        normalization_counts = self.regularization * w * len(self.data)

        for key, features_inx in self.features_dict.items():
            (words, t2, t1, t, isfirst) = key
            count = self.features_count_dict[key]

            # calculate empirical loss term
            empirical_counts[features_inx] += count

            # calculate p(y|x,w) for words x[i-1,i,i+1] and for all possible tag[i]
            p = self.calc_all_possible_tags_probabilities_train(words, t1, t2, w, isfirst)

            # calculate expected_loss term
            tag_feat = self.features_dict_all_tags[(words, t2, t1, isfirst)]
            expected_size = expected_counts[tag_feat].shape[1]
            expected_counts[tag_feat] += np.transpose(np.tile(p, (expected_size, 1))) * count

            # calculate expected_loss term - old
            # for j, tag in enumerate(self.T):
            #     tag_feat = self.features_dict_all_tags[(words, t2, t1, tag, isfirst)]
            #     expected_counts[tag_feat] += p[j] * count

        w_grads = empirical_counts - expected_counts - normalization_counts
        if self.verbosity:
            print('Done calculate grads in {}, max abs grad is {}, max abs w is {}'.format((time.time() - t0) / 60,
                                                                                           np.max(np.abs(w_grads)),
                                                                                           np.max(np.abs(w))))
        return (-1) * w_grads

    def loss(self, w):
        t0 = time.time()
        empirical_loss = 0
        expected_loss = 0

        # calculate normalization loss term
        normalization_loss = (np.sum(np.square(w)) * self.regularization / 2) * len(self.data)

        for key, features_inx in self.features_dict.items():
            (words, t2, t1, t, isfirst) = key
            count = self.features_count_dict[key]

            # calculate empirical loss term
            empirical_loss += np.sum(w[features_inx]) * count

            # calculate expected_loss term
            exp_term = np.sum(w[self.features_dict_all_tags[(words, t2, t1, isfirst)]], axis=1)  # TODO
            # exp_term = np.zeros(self.T_size - 2)
            # for j, tag in enumerate(self.T):
            #     exp_term[j] = np.sum(w[self.features_dict_all_tags[(words, t2, t1, tag, isfirst)]])
            expected_loss += logsumexp(exp_term) * count

        loss_ = empirical_loss - expected_loss - normalization_loss
        if self.verbosity:
            print('Done calculate Loss in {} minutes, Loss is: {}'.format((time.time() - t0) / 60, (-1) * loss_))
        return (-1) * loss_

    def get_feature_size(self):
        size_dict = {}
        size_dict['F100'] = self.V_size * self.T_size  # represens word ant tag for all possible combinations
        size_dict['F103'] = self.T_size ** 3  # trigram of tags
        size_dict['F104'] = self.T_size ** 2  # bigram of tags
        if self.mode == 'complex':
            size_dict['F101_1'] = self.T_size * len(
                self.suffix_1)  # all posible tags for each word in importnat suffix list
            size_dict['F101_2'] = self.T_size * len(
                self.suffix_2)  # all posible tags for each word in importnat suffix list
            size_dict['F101_3'] = self.T_size * len(
                self.suffix_3)  # all posible tags for each word in importnat suffix list
            size_dict['F101_4'] = self.T_size * len(
                self.suffix_4)  # all posible tags for each word in importnat suffix list
            size_dict['F102_1'] = self.T_size * len(
                self.prefix_1)  # all posible tags for each word in importnat prefix list
            size_dict['F102_2'] = self.T_size * len(
                self.prefix_2)  # all posible tags for each word in importnat prefix list
            size_dict['F102_3'] = self.T_size * len(
                self.prefix_3)  # all posible tags for each word in importnat prefix list
            size_dict['F102_4'] = self.T_size * len(
                self.prefix_4)  # all posible tags for each word in importnat prefix list
            size_dict['F105'] = self.T_size  # unigram of tag
            if self.use_106_107:
                size_dict['F106'] = self.V_size * self.T_size  # representes last word and current tag
                size_dict['F107'] = self.V_size * self.T_size  # representes next word and current tag
            size_dict['G1'] = self.T_size  # is current word of the form <number-word> and tagged as t(i)
            size_dict['G2'] = self.T_size  # is current word starts with Upper case + the current tag
            size_dict['G3'] = self.T_size  # is Upper case and first word in sentance, with tag t(i)
            size_dict['G4'] = self.T_size  # is Upper case and *not* first word in sentance, with tag t(i)
            size_dict['G5'] = self.T_size  # is the all word in uppercase, with tag t(i)
            size_dict['G6'] = self.T_size  # is the all word in number, with tag t(i)
            size_dict['G7_1'] = self.T_size  # is the word unknown (train: less then threshold), not a number and t(i)
            size_dict[
                'G7_2'] = self.T_size ** 2  # is the word unknown (train: less then threshold), not a number and t(i-1,i)
            size_dict[
                'G7_3'] = self.T_size ** 3  # is the word unknown (train: less then threshold), not a number and t(i-2,i-1,i)

        return sum(size_dict.values())

    def get_features(self, words, tags, is_first=False):
        """
        :param words: list of <w(i-1),w(i),w(i+1)>
        :param tags: POS tags of the trigram as as a list <t(i-2), t(i-1), t(i)>
        :return: features - list of the features vector's indexes which are "true"
        """
        features = []
        word_len = len(words[1])

        # base features:
        # 1 if xi = x and ti = t
        try:
            F100 = self.V_dict[words[1]] * self.T_size + self.T_with_start_dict[tags[2]]
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
            if word_len > 2 and words[1][-2:] in self.suffix_2.keys():
                F101_2 = self.suffix_2[words[1][-2:]] * self.T_size + self.T_with_start_dict[tags[2]]
                features.append(F101_2 + F104_len)
            F101_2_len = F104_len + self.T_size * len(self.suffix_2)
            if word_len > 3 and words[1][-3:] in self.suffix_3.keys():
                F101_3 = self.suffix_3[words[1][-3:]] * self.T_size + self.T_with_start_dict[tags[2]]
                features.append(F101_3 + F101_2_len)
            F101_3_len = F101_2_len + self.T_size * len(self.suffix_3)
            if word_len > 4 and words[1][-4:] in self.suffix_4.keys():
                F101_4 = self.suffix_4[words[1][-4:]] * self.T_size + self.T_with_start_dict[tags[2]]
                features.append(F101_4 + F101_3_len)
            F101_4_len = F101_3_len + self.T_size * len(self.suffix_4)
            F101_len = F101_4_len

            # F102: prefix of length 2/3/4 letters which is in prefix list && tag <t(i)>
            if word_len > 2 and words[1][:2] in self.prefix_2.keys():
                F102_2 = self.prefix_2[words[1][:2]] * self.T_size + self.T_with_start_dict[tags[2]]
                features.append(F102_2 + F101_len)
            F102_2_len = F101_len + self.T_size * len(self.prefix_2)

            if word_len > 3 and words[1][:3] in self.prefix_3.keys():
                F102_3 = self.prefix_3[words[1][:3]] * self.T_size + self.T_with_start_dict[tags[2]]
                features.append(F102_3 + F102_2_len)
            F102_3_len = F102_2_len + self.T_size * len(self.prefix_3)

            if word_len > 4 and words[1][:4] in self.prefix_4.keys():
                F102_4 = self.prefix_4[words[1][:4]] * self.T_size + self.T_with_start_dict[tags[2]]
                features.append(F102_4 + F102_3_len)
            F102_4_len = F102_3_len + self.T_size * len(self.prefix_4)
            F102_len = F102_4_len

            # F105: tag is <t(i)>
            F105 = self.T_with_start_dict[tags[2]]
            features.append(F105 + F102_len)
            F105_len = F102_len + self.T_size

            if self.use_106_107:
                # F106: is last word w[i-1] and tag t[i]
                try:
                    F106 = self.V_dict[words[0]] * self.T_size + self.T_with_start_dict[tags[2]]
                    features.append(F106 + F105_len)
                except:
                    tmp = 0
                F106_len = F105_len + self.V_size * self.T_size

                # F107: is next word w[i+1] and tag t[i]
                try:
                    F107 = self.V_dict[words[2]] * self.T_size + self.T_with_start_dict[tags[2]]
                    features.append(F107 + F106_len)
                except:
                    tmp = 0
                F107_len = F106_len + self.V_size * self.T_size

            else:
                F107_len = 0

            # G1 : is the current word in the form of <number-noun> (e.g 12-inch) and tagged as t(i)?
            number_check = words[1].partition('-')
            if number_check[0].isdigit() and number_check[1] == '-' and not number_check[2].isdigit():
                G1 = self.T_with_start_dict[tags[2]]
                features.append(G1 + F107_len)
            G1_len = F107_len + self.T_size

            # G2 : is the current word starts in Upper case and tag is t_i?
            if words[1][0].isupper() and words[1][0].isalpha():
                G2 = self.T_with_start_dict[tags[2]]
                features.append(G2 + G1_len)
            G2_len = G1_len + self.T_size

            # G3 : is word tagged as t(i), is Capital letter and first word in sentance?
            if words[1][0].isupper() and is_first:
                G3 = self.T_with_start_dict[tags[2]]
                features.append(G3 + G2_len)
            G3_len = G2_len + self.T_size

            # G4 : is  word tagged as t(i), it Capital letter and first word in sentance?
            if words[1][0].isupper() and not is_first:
                G4 = self.T_with_start_dict[tags[2]]
                features.append(G4 + G3_len)
            G4_len = G3_len + self.T_size

            # G5 : is the all word in uppercase and tagged as t(i)?
            if words[1].isupper():
                G5 = self.T_with_start_dict[tags[2]]
                features.append(G5 + G4_len)
            G5_len = G4_len + self.T_size

            # G6 : is all the word is digits (even if seperated by '.' or ',' or ':'), and taged as t(i)?
            is_number = False
            # check for digit or for . partition
            number_check = words[1].partition('.')
            if number_check[0].isdigit():
                if number_check[2] == '' or number_check[2].isdigit():
                    is_number = True
            # check for : partition
            number_check = words[1].partition(':')
            if number_check[0].isdigit() and number_check[2].isdigit():
                is_number = True
            # check for , partition
            number_check = words[1].partition(',')
            if number_check[0].isdigit() and number_check[2].isdigit():
                is_number = True

            if is_number:
                G6 = self.T_with_start_dict[tags[2]]
                features.append(G6 + G5_len)
            G6_len = G5_len + self.T_size

            # G7 : is word in unknown (in train: less then some value) not a number and tags history
            G7_1_len = self.T_size + G6_len
            G7_2_len = self.T_size ** 2 + G7_1_len
            G7_3_len = self.T_size ** 3 + G7_2_len
            G7_len = G7_3_len

            if (self.train_or_predict_mode == 'predict' and words[1] not in self.V and not is_number) or \
                    (self.train_or_predict_mode == 'train' and self.V_count[
                        words[1]] <= self.word_count_threshold and not is_number):
                # G7_1 : tag unigram
                G7_1 = self.T_with_start_dict[tags[2]]
                features.append(G7_1 + G6_len)

                # G7_2 : tag bigram
                G7_2 = self.T_with_start_dict[tags[2]] * self.T_size + self.T_with_start_dict[tags[1]]
                features.append(G7_2 + G7_1_len)

                # G7_3 : tag trigram
                G7_3 = self.T_with_start_dict[tags[2]] * (self.T_size ** 2) + self.T_with_start_dict[
                                                                                  tags[1]] * self.T_size + \
                       self.T_with_start_dict[tags[0]]
                features.append(G7_3 + G7_2_len)

        return features

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
        # V_COMP_size = len(V_COMP)
        if self.use_106_107:
            V_COMP_dict = {}
            j = 0
            for sen in corpus:
                sentence_len = len(sen)
                for i, word in enumerate(sen):
                    if i == 0:
                        if ('/*', sen[i], sen[i + 1]) not in V_COMP_dict.keys():
                            V_COMP_dict[('/*', sen[i], sen[i + 1])] = j
                            j += 1
                    elif i == sentence_len - 1:
                        if (sen[i - 1], sen[i], '/STOP') not in V_COMP_dict.keys():
                            V_COMP_dict[(sen[i - 1], sen[i], '/STOP')] = j
                            j += 1
                    elif tuple(sen[i - 1:i + 2]) not in V_COMP_dict.keys():
                        V_COMP_dict[tuple(sen[i - 1:i + 2])] = j
                        j += 1
            V_COMP_dict_size = len(V_COMP_dict)
        else:
            V_COMP_dict = {}
            for i, v in enumerate(V_COMP):
                V_COMP_dict[('/106_107', v, '/106_107')] = i
            V_COMP_dict_size = len(V_COMP_dict)

        # init probability matrix:
        # holds all p(word,t(i),t(i-1),t(i-2))
        prob_mat = np.zeros((V_COMP_dict_size, self.T_size - 2, self.T_size - 2, self.T_size - 2))

        all_sentence_tags = []
        all_tagged_sentence = []

        print('Start predicting...')
        t0 = time.time()
        for sen_num, sentence in enumerate(corpus):
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
                words = ('/106_107', sentence[k], '/106_107')

                # if havn't seen the word before - update the probebility matrix for all possible tagsL
                if k > 1:
                    if self.use_106_107:
                        if k == sentence_len - 1:
                            words = (sentence[k - 1], sentence[k], '/STOP')
                        else:
                            words = tuple(sentence[k - 1:k + 2])

                    if not prob_mat[V_COMP_dict[words], 0, 0, 0].any():
                        # for u in self.T:  # for each t-1 possible tag
                        #     for t in self.T:  # for each t-2 possible tag:
                        #         # if this is the last word - send the next word as "STOP"
                        #         prob_mat[V_COMP_dict[words], :, self.T_dict[u],
                        #         self.T_dict[t]] = self.calc_all_possible_tags_probabilities(words, u, t, self.weights)

                        prob_mat[V_COMP_dict[words], :, :, :] = self.calc_all_possible_tags_probabilities_pred(words, self.weights)
                        # if self.verbosity:
                        #     print('Finished calculate prob matrix for: ', words)

                for current_tag in self.T:  # for each t possible tag

                    if k == 0:
                        # at the first two words there is no meaning to the k-1 tag index. pi[k-1]
                        if self.use_106_107:
                            words = ('/*', sentence[k], sentence[k + 1])
                        pi_matrix[k, 0, :] = 1 * self.calc_all_possible_tags_probabilities(words, '/*', '/*',
                                                                                           self.weights, True)
                        break
                    elif k == 1:
                        if self.use_106_107:
                            words = tuple(sentence[k - 1:k + 2])
                        for u in self.T:  # for each t-1 possible tag
                            pi_matrix[k, self.T_dict[u], :] = pi_matrix[k - 1, 0, self.T_dict[
                                u]] * self.calc_all_possible_tags_probabilities(words, u, '/*', self.weights)
                        break
                    else:
                        for u in self.T:  # for each t-1 possible tag
                            # calculate pi value, and check if it exeeds the current max:
                            pi_values = pi_matrix[k - 1, :, self.T_dict[u]] * prob_mat[V_COMP_dict[words],
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

            if sen_num % 50 == 0 and sen_num:
                print('Finished predicting sentence {} in {} minutes'.format(sen_num, (time.time() - t0) / 60))

        prediction_time = (time.time() - t0) / 60
        if log_path is not None:
            with open(log_path, 'a') as f:
                f.writelines('\nPrediction data:\n')
                f.writelines('Number of sentences predicted: {}\n'.format(len(corpus)))
                f.writelines('Prediction time: {}\n'.format(prediction_time))

        print('Done predicting in {} minutes'.format(prediction_time))
        return all_tagged_sentence, all_sentence_tags

    def init_spelling_dicts(self):
        """
        :param threshold: min count for suffix in the train set, in order to be taken for feature
        :return: save 3 lists for suffix at length 2-4 and 3 for prefix as "self" items
        """
        # declare histograms that will countain count in the corpus for each prefix and suffix
        histogram_suffix1 = {}
        histogram_suffix2 = {}
        histogram_suffix3 = {}
        histogram_suffix4 = {}

        histogram_prefix1 = {}
        histogram_prefix2 = {}
        histogram_prefix3 = {}
        histogram_prefix4 = {}

        # init the histograms to 0 for all relevant prefixes and suffixes
        for word in self.V:
            word_len = len(word)
            if word[-1].isdigit() or word[0].isdigit(): continue
            if word_len > 1:
                histogram_suffix1[word[-1:]] = 0
                histogram_prefix1[word[:1]] = 0
            if word_len >= 2:
                histogram_suffix2[word[-2:]] = 0
                histogram_prefix2[word[:2]] = 0
            if word_len >= 3:
                histogram_suffix3[word[-3:]] = 0
                histogram_prefix3[word[:3]] = 0
            if word_len >= 4:
                histogram_suffix4[word[-4:]] = 0
                histogram_prefix4[word[:4]] = 0

        # fill the histogram with count of each relevant prefix / suffix in the corpus
        for word in self.V:
            word_len = len(word)
            if word[-1].isdigit() or word[0].isdigit():
                continue
            if word_len > 1:
                histogram_suffix1[word[-1:]] += 1
                histogram_prefix1[word[:1]] += 1
            if word_len >= 2:
                histogram_suffix2[word[-2:]] += 1
                histogram_prefix2[word[:2]] += 1
            if word_len >= 3:
                histogram_suffix3[word[-3:]] += 1
                histogram_prefix3[word[:3]] += 1
            if word_len >= 4:
                histogram_suffix4[word[-4:]] += 1
                histogram_prefix4[word[:4]] += 1

        # suffix 1
        i = 0
        for key in histogram_suffix1.keys():
            if histogram_suffix1[key] > self.spelling_threshold:
                self.suffix_1[key] = i;
                i += 1

        # suffix 2
        i = 0
        for key in histogram_suffix2.keys():
            if histogram_suffix2[key] > self.spelling_threshold:
                self.suffix_2[key] = i;
                i += 1

        # suffix 3
        i = 0
        for key in histogram_suffix3.keys():
            if histogram_suffix3[key] > self.spelling_threshold:
                self.suffix_3[key] = i
                i += 1

        # suffix 4
        i = 0
        for key in histogram_suffix4.keys():
            if histogram_suffix4[key] > self.spelling_threshold:
                self.suffix_4[key] = i;
                i += 1

        # prefix 1
        i = 0
        for key in histogram_prefix1.keys():
            if histogram_prefix1[key] > self.spelling_threshold:
                self.prefix_1[key] = i;
                i += 1

        # prefix 2
        i = 0
        for key in histogram_prefix2.keys():
            if histogram_prefix2[key] > self.spelling_threshold:
                self.prefix_2[key] = i;
                i += 1

        # prefix 3
        i = 0
        for key in histogram_prefix3.keys():
            if histogram_prefix3[key] > self.spelling_threshold:
                self.prefix_3[key] = i;
                i += 1

        # prefix 4
        i = 0
        for key in histogram_prefix4.keys():
            if histogram_prefix4[key] > self.spelling_threshold:
                self.prefix_4[key] = i;
                i += 1

    def test(self, test_data_path, end=0, start=0, parallel = False, verbosity=0, save_results_to_file=None, log_path=None):
        self.verbosity = verbosity
        (_, _, test, test_tag) = data_preprocessing(test_data_path, 'test')

        if end:
            corpus = test[start:end]
            corpus_tag = test_tag[start:end]
        else:
            corpus = test[start:]
            corpus_tag = test_tag[start:]

        # run Viterbi algorithm
        if parallel:
            (all_tagged_sentence, all_sentence_tags) = self.predict_parallel(corpus, verbosity=verbosity, log_path=log_path)
        else:
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
                f.writelines('\nTest data:\n')
                f.writelines('Number of sentences tested on: {}\n'.format(len(corpus)))
                f.writelines('Total accuracy: {}\n'.format(tot_accuracy))

    def save_model(self, resultsfn):
        print('Saving model to {}'.format(resultsfn))
        # creating directory
        if not os.path.exists(resultsfn):
            os.makedirs(resultsfn)

        # dump all results:
        with open(resultsfn + '\\model.pkl', 'wb') as f:
            pickle.dump(self, f)

    def predict_parallel(self, corpus, verbosity=0, log_path=None):

        print('Start predicting...')
        t0 = time.time()
        self.verbosity = verbosity

        self.build_prob_matrix_parallel(corpus)

        pool = ThreadPool() #Pool(processes=5, maxtasksperchild=3)
        res = pool.map(self.predict_sentence, corpus)
        pool.close()
        pool.join()

        all_sentence_tags = [x[1] for x in res]
        all_tagged_sentence = [x[0] for x in res]

        prediction_time = (time.time() - t0) / 60

        del self.prob_mat, self.V_COMP_dict

        self.verbosity = 0

        if log_path is not None:
            with open(log_path, 'a') as f:
                f.writelines('Prediction time: {}\n'.format(prediction_time))

        print('Done parallel predicting in {} minutes'.format(prediction_time))
        return all_tagged_sentence, all_sentence_tags

    def build_word_prob_matrix(self, job):
        words = job[0]
        prob_mat = np.zeros((self.T_size - 2, self.T_size - 2, self.T_size - 2))

        for u in self.T:  # for each t-1 possible tag
            for t in self.T:  # for each t-2 possible tag:
                prob_mat[:, self.T_dict[u],
                self.T_dict[t]] = self.calc_all_possible_tags_probabilities(words, u, t, self.weights)

        self.prob_mat[self.V_COMP_dict[words], :, :, :] = prob_mat
        if self.verbosity:
            V_idx = job[1]
            print('Finished calculate prob matrix for: ', words, ' with index:', V_idx)

    def build_prob_matrix_parallel(self, corpus):
        print('Building prob matrix...')
        t0 = time.time()
        # case corpus is only 1 sentence:
        if len(corpus) == 1:
            corpus = [corpus]
        # init a list of singular words in the target corpus:
        V_COMP = sorted(list(set(chain(*corpus))))
        # V_COMP_size = len(V_COMP)
        if self.use_106_107:
            V_COMP_dict = {}
            j = 0
            for sen in corpus:
                sentence_len = len(sen)
                for i, word in enumerate(sen):
                    if i == 0:
                        if ('/*', sen[i], sen[i + 1]) not in V_COMP_dict.keys():
                            V_COMP_dict[('/*', sen[i], sen[i + 1])] = j
                            j += 1
                    elif i == sentence_len - 1:
                        if (sen[i - 1], sen[i], '/STOP') not in V_COMP_dict.keys():
                            V_COMP_dict[(sen[i - 1], sen[i], '/STOP')] = j
                            j += 1
                    elif tuple(sen[i - 1:i + 2]) not in V_COMP_dict.keys():
                        V_COMP_dict[tuple(sen[i - 1:i + 2])] = j
                        j += 1
        else:
            V_COMP_dict = {}
            for i, v in enumerate(V_COMP):
                V_COMP_dict[('/106_107', v, '/106_107')] = i

        self.V_COMP_dict = V_COMP_dict
        self.V_comp_results = {}

        V_COMP_dict_size = len(V_COMP_dict)

        # init probability matrix:
        # holds all p(word,t(i),t(i-1),t(i-2))
        self.prob_mat = np.zeros((V_COMP_dict_size, self.T_size - 2, self.T_size - 2, self.T_size - 2))

        pool = ThreadPool() # Pool(processes=4, maxtasksperchild=1)
        # res = pool.map(self.build_word_prob_matrix, zip(V_COMP_dict.keys(), V_COMP_dict.values()))
        pool.map(self.build_word_prob_matrix, zip(V_COMP_dict.keys(), V_COMP_dict.values()))
        pool.close()
        pool.join()

        # for tuple in res:
        #     self.prob_mat[V_COMP_dict[tuple[0]]] = tuple[1]

        print('Build prob matrix in time: {} min'.format((time.time() - t0) / 60))

    def predict_sentence(self, sentence):
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
            words = ('/106_107', sentence[k], '/106_107')

            if k > 1:
                if self.use_106_107:
                    if k == sentence_len - 1:
                        words = (sentence[k - 1], sentence[k], '/STOP')
                    else:
                        words = tuple(sentence[k - 1:k + 2])

            for current_tag in self.T:  # for each t possible tag

                if k == 0:
                    # at the first two words there is no meaning to the k-1 tag index. pi[k-1]
                    if self.use_106_107:
                        words = ('/*', sentence[k], sentence[k + 1])
                    pi_matrix[k, 0, :] = 1 * self.calc_all_possible_tags_probabilities(words, '/*', '/*',
                                                                                       self.weights)
                    break
                elif k == 1:
                    if self.use_106_107:
                        words = tuple(sentence[k - 1:k + 2])
                    for u in self.T:  # for each t-1 possible tag
                        pi_matrix[k, self.T_dict[u], :] = pi_matrix[k - 1, 0, self.T_dict[
                            u]] * self.calc_all_possible_tags_probabilities(words, u, '/*', self.weights)
                    break
                else:
                    for u in self.T:  # for each t-1 possible tag
                        # calculate pi value, and check if it exeeds the current max:
                        pi_values = pi_matrix[k - 1, :, self.T_dict[u]] * self.prob_mat[self.V_COMP_dict[words],
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

        if self.verbosity:
            print(tagged_sentence)

        return tagged_sentence, sentence_tags
