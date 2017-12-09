# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 01:02:59 2017

@author: carmelr
"""
import time

try: import cPickle as pickle
except: import pickle

import pandas as pd
import numpy as np
from itertools import chain
from scipy.misc import logsumexp
from scipy.optimize import fmin_l_bfgs_b
from collections import Counter


def softmax(numerator, denominator):
    denominator_max = np.max(denominator)
    denominator -= denominator_max
    numerator -= denominator_max
    return np.exp(numerator) / np.sum(np.exp(denominator))


def load_model(Fn):
    with open(Fn+'.pkl', 'rb') as f:
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

    def train(self, data_path, regularization=0.1, mode='base', spelling_threshold=10, verbosity=0):
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

        print('Start training...')
        t0 = time.time()
        optimal_params = fmin_l_bfgs_b(func=self.loss, x0=self.weights, fprime=self.loss_grads)
        print('Finished training with code: ',optimal_params[2]['warnflag'])
        print('Training time: ',(time.time() - t0)/60)
        print('Iterations number: ', optimal_params[2]['nit'])
        print('Calls number: ', optimal_params[2]['funcalls'])

        if optimal_params[2]['warnflag']:
            print('Error in training:\n{}\\n'.format(optimal_params[2]['task']))
        else:
            self.weights = optimal_params[0]

        del self.data, self.data_tag

    def create_confusion_matrix(self, all_sentence_tags, results_path):
        # init confusion matrix - [a,b] if we tagged word as "a" but the real tag is "b"
        confusion_matrix = np.zeros((self.T_size -2, self.T_size -2),dtype=np.int)
        max_failure_matrix = np.zeros((10, self.T_size -2),dtype=np.int)

        res = 0
        failure_dict = {}
        for tag in self.T:
            failure_dict[tag] = 0
        for sen_idx, sen_tags in enumerate(all_sentence_tags):
            for pred_tag,real_tag in zip(sen_tags, self.test_tag[sen_idx]):
                if pred_tag == real_tag:
                    res += 1
                else:
                    failure_dict[pred_tag] += 1
                confusion_matrix[self.T_dict[pred_tag],self.T_dict[real_tag]] += 1
        common_failure_tags = dict(Counter(failure_dict).most_common(10))

        i = 0
        for key in common_failure_tags.keys():
            common_failure_tags[key] = i
            i += 1

        for tag in common_failure_tags.keys():
            max_failure_matrix[common_failure_tags[tag]] = confusion_matrix[self.T_dict[tag]]

        confusion_matrix_fn = results_path +'\\confusion_matrix.csv'
        df = pd.DataFrame(confusion_matrix, index = self.T, columns = self.T)
        df.to_csv(confusion_matrix_fn, index=True, header=True, sep=',')

        max_failure_matrix_fn = results_path +'\\max_failure_matrix.csv'
        df = pd.DataFrame(max_failure_matrix, index = list(common_failure_tags.keys()), columns = self.T)
        df.to_csv(max_failure_matrix_fn, index=True, header=True, sep=',')

    def calc_all_possible_tags_probabilities(self, x, t1, t2, w):
        """
        calculate probability p(ti|xi,w)
        :param x: list of words <w(i-1),w(i),w(i+1)>
        :param t1: POS tag for word[i-1]
        :param t2: POS tag for word[i-2]
        :param w: weights vector
        :return: a list for all possible ti probabilities p(ti|xi,w) as float64
        """
        denominator = np.zeros(self.T_size - 2)
        for i, tag in enumerate(self.T):
            denominator[i] = np.sum(w[self.get_features(x, [t2, t1, tag])])
        return softmax(denominator,denominator)

    def loss_grads(self, w):

        # TODO: remove prints
        # t0 = time.time()
        w_grads = np.zeros(self.feature_size, dtype=np.float64)
        for h, sentence in enumerate(self.data):
            tag_sentence = self.data_tag[h]

        # calculate weights normalization term
            normalization_counts = self.regularization * w

        # calculate empirical counts term
            empirical_counts = np.zeros(self.feature_size, dtype=np.float64)
            for i, word in enumerate(sentence[:-1]):
                if i == 0 or i == 1:
                    continue
                empirical_counts[self.get_features(word, tag_sentence[i-2:i+1])] += 1

        # calculate expected counts term
            expected_counts = np.zeros(self.feature_size, dtype=np.float64)

            # go over all words in sentence
            for i, word in enumerate(sentence[:-1]):
                # 2 first words are /* /* start symbols
                if i == 0 or i == 1:
                    continue

                # calculate p(y|x,w) for word x and for all possible tag[i]
                p = self.calc_all_possible_tags_probabilities(word, tag_sentence[i - 1], tag_sentence[i - 2], w)

                for j, tag in enumerate(self.T):
                    # take features indexes for tag[i] = j
                    tag_feat = self.get_features(word, [tag_sentence[i - 2], tag_sentence[i - 1], tag])

                    # add p[j] to all features indexes that are equal to 1 (f_array[i - 2, j, :] is a list of indexes)
                    expected_counts[tag_feat] += p[j]
                # TODO: need to insert something that checks for inf or nan like:  np.isinf(a).any()

            # update grads for the sentence
            w_grads += empirical_counts - expected_counts - normalization_counts
        # TODO: remove prints
        # print('Done calculate grads in {}, max abs grad is {}, max abs w is {}'.format((time.time()-t0)/60, np.max(np.abs(w_grads)), np.max(np.abs(w))))
        return (-1)*w_grads

    def loss(self, w):
        loss_ = 0
        for h, sentence in enumerate(self.data):
            tag_sentence = self.data_tag[h]
            empirical_loss = 0
            expected_loss = 0

            # calculate normalization loss term
            normalization_loss = np.sum(np.square(w)) * self.regularization/2

            for i, word in enumerate(sentence[:-1]):
                if i == 0 or i == 1:
                    continue
                # calculate empirical loss term
                features_inx = self.get_features(word, tag_sentence[i-2:i+1])
                empirical_loss += np.sum(w[features_inx])

                # calculate expected_loss term
                exp_term = np.zeros(self.T_size - 2)
                for j, tag in enumerate(self.T):
                    exp_term[j] = np.sum(w[self.get_features(word, [tag_sentence[i-2], tag_sentence[i-1], tag])])
                expected_loss += logsumexp(exp_term)

            loss_ += empirical_loss - expected_loss - normalization_loss
        if self.verbosity:
            print('Loss is: {}'.format((-1)*loss_))
        return (-1)*loss_

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
            size_dict['G1'] = self.T_size  # is current word a number + the current tag
            size_dict['G2'] = self.T_size  # is current word starts with Upper case + the current tag

        return sum(size_dict.values())

    def predict(self, corpus, verbosity=0, save_results_to_file=None):
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
        for i,v in enumerate(V_COMP):
            V_COMP_dict[v] = i

        # init probability matrix:
        # holds all p(word,t(i),t(i-1),t(i-2))
        prob_mat = np.zeros((V_COMP_size, self.T_size - 2,self.T_size - 2,self.T_size - 2))

        all_sentence_tags = []
        all_tagged_sentence = []

        print('Start predicting...')
        t0 = time.time()
        for sentence in corpus:
            # init empty array of strings to save the tag for each word in the sentance
            sentence_len = len(sentence)
            sentence_tags = [''  for x in range(sentence_len)]

            # init dynamic matrix with size: 
            # pi_matrix[k,t(i-1),t(i)] is the value of word number k, preciding tag u and t accordingly
            pi_matrix = np.zeros((sentence_len,self.T_size-2,self.T_size-2))

            # init back pointers matrix:
            # bp[k,t,u] is the tag index of word number k-2, following tag t and u accordingly
            bp = np.zeros((sentence_len,self.T_size-2,self.T_size-2),dtype=np.int)

            for k in range (0,sentence_len): # for each word in the sentence

                # if havn't seen the word before - update the probebility matrix for all possible tagsL
                if k > 1 and not prob_mat[V_COMP_dict[sentence[k]],0,0,0].any():
                    for u in self.T: # for each t-1 possible tag
                        for t in self.T: # for each t-2 possible tag:
                            # if this is the last word - send the next word as "STOP"
                            if k == sentence_len-1:
                                prob_mat[V_COMP_dict[sentence[k]],:, self.T_dict[u], self.T_dict[t]] = self.calc_all_possible_tags_probabilities([sentence[k-1],sentence[k],'/STOP'], u, t, self.weights)
                            else:
                                prob_mat[V_COMP_dict[sentence[k]],:, self.T_dict[u], self.T_dict[t]] = self.calc_all_possible_tags_probabilities(sentence[k-1:k+2], u, t, self.weights)

                for current_tag in self.T: # for each t possible tag

                    if k == 0:
                        # at the first two words there is no meaning to the k-1 tag index. pi[k-1]
                        pi_matrix[k, 0, :] = 1 * self.calc_all_possible_tags_probabilities(['/*', '/*', sentence[k]], '/*', '/*', self.weights)
                        break
                    elif k == 1:
                        for u in self.T: # for each t-1 possible tag
                            pi_matrix[k, self.T_dict[u], :] = pi_matrix[k - 1, 0, self.T_dict[u]] * self.calc_all_possible_tags_probabilities(['/*', sentence[k-1], sentence[k]], u, '/*', self.weights)
                        break
                    else:
                        for u in self.T: # for each t-1 possible tag
                            # calculate pi value, and check if it exeeds the current max:
                            pi_values = pi_matrix[k-1, :, self.T_dict[u]] * prob_mat[V_COMP_dict[sentence[k]], self.T_dict[current_tag], self.T_dict[u], :]
                            ind = np.argmax(pi_values)
                            if pi_values[ind] > pi_matrix[k, self.T_dict[u], self.T_dict[current_tag]]:

                                # update max:
                                pi_matrix[k, self.T_dict[u], self.T_dict[current_tag]] = pi_values[ind]

                                # update back pointers:
                                bp[k, self.T_dict[u], self.T_dict[current_tag]] = ind

            u_ind, curr_ind = np.unravel_index(pi_matrix[sentence_len-1,:,:].argmax(), pi_matrix[sentence_len-1,:,:].shape)
            sentence_tags[-2:] = [self.T[u_ind], self.T[curr_ind]]

            # extracting MEMM tags path from back pointers matrix:
            for i in range(sentence_len-3,-1,-1):
                # calculate the idx of tag i in T db:
                # reminder - bp[k,t,u] is the tag of word k-2, following tag t and u accordingly
                k_tag_idx = bp[i + 2, self.T_dict[sentence_tags[i + 1]], self.T_dict[sentence_tags[i + 2]]]

                # update the i-th tag to the list of tags
                sentence_tags[i] = self.T[k_tag_idx]

            # build tagged sentence:
            tagged_sentence = ''
            for i in range(sentence_len):
                tagged_sentence += (sentence[i] +'_')
                tagged_sentence += sentence_tags[i] + (' ')
            all_sentence_tags.append(sentence_tags)
            all_tagged_sentence.append(tagged_sentence)
            if self.verbosity:
                print(tagged_sentence)
        print('Done predicting in {} minutes'.format((time.time() - t0)/60))
        # if save_results_to_file is not None:
        #     print('Saving results to predicting in {} minutes'.format((time.time() - t0) / 60))
        return all_tagged_sentence, all_sentence_tags

    def get_features(self, words, tags):
        """
        :param word: list of words - <w(i-1), w(i), w(i+1)>
        :param tags: POS tags of the trigram as as a list <t(i-2), t(i-1), t(i)>
        :return: features - list of the features vector's indexes which are "true" 
        """
        features = []
        word_len = len(words[1])

        #base features:
        # 1 if xi = x and ti = t
        try:
            F100 = self.V_dict[words[1]] * self.T_size + self.T_with_start_dict[tags[2]]
            features.append(F100)
        except:
            tmp = 0 # must do something in except
        F100_len = self.V_size * self.T_size

        # trigram feature - 1 if <t(i-2),t(is),t(i)> = <t1,t2,t3>
        F103 = self.T_with_start_dict[tags[2]] * (self.T_size ** 2) + self.T_with_start_dict[tags[1]] * self.T_size + \
               self.T_with_start_dict[tags[0]]
        features.append(F103 + F100_len)
        F103_len = F100_len + self.T_size**3

        # bigram feature - 1 if <t(i-1),t(i)> = <t1,t2>
        F104 = self.T_with_start_dict[tags[2]] * self.T_size + self.T_with_start_dict[tags[1]]
        features.append(F104 + F103_len)
        F104_len = F103_len + self.T_size**2

        # complex features:
        if self.mode == 'complex':

            # F101: suffix of length  2/3/4 which is in suffix lists && tag <t(i)>
            if word_len > 2 and word[1][-2:] in self.suffix_2.keys():
                F101_2 = self.suffix_2[words[1][-2:]]*self.T_size + self.T_with_start_dict[tags[2]]
                features.append(F101_2 + F104_len)
            F101_2_len = F104_len + self.T_size*len(self.suffix_2)
            if word_len > 3 and words[1][-3:] in self.suffix_3.keys():
                F101_3 = self.suffix_3[words[1][-3:]]*self.T_size + self.T_with_start_dict[tags[2]]
                features.append(F101_3 + F101_2_len)
            F101_3_len = F101_2_len + self.T_size*len(self.suffix_3)
            if word_len > 4 and words[1][-4:] in self.suffix_4.keys():
                F101_4 = self.suffix_4[words[1][-4:]]*self.T_size + self.T_with_start_dict[tags[2]]
                features.append(F101_4 + F101_3_len)
            F101_4_len = F101_3_len + self.T_size*len(self.suffix_4)
            F101_len = F101_4_len

            # F102: prefix of length 2/3/4 letters which is in prefix list && tag <t(i)>
            if word_len > 2 and words[1][:2] in self.prefix_2.keys():
                F102_2 = self.prefix_2[words[1][:2]]*self.T_size + self.T_with_start_dict[tags[2]]
                features.append(F102_2 + F101_len)
            F102_2_len = F101_len + self.T_size*len(self.prefix_2)

            if word_len > 3 and words[1][:3] in self.prefix_3.keys():
                F102_3 = self.prefix_3[words[1][:3]]*self.T_size + self.T_with_start_dict[tags[2]]
                features.append(F102_3 + F102_2_len)
            F102_3_len = F102_2_len + self.T_size*len(self.prefix_3)

            if word_len > 4 and words[1][:4] in self.prefix_4.keys():
                F102_4 = self.prefix_4[words[1][:4]]*self.T_size + self.T_with_start_dict[tags[2]]
                features.append(F102_4 + F102_3_len)
            F102_4_len = F102_3_len + self.T_size*len(self.prefix_4)
            F102_len = F102_4_len

            # F105: tag is <t(i)>
            F105 = self.T_with_start_dict[tags[2]]
            features.append(F105 + F102_len)
            F105_len = F102_len + self.T_size

            # F106: last word is w(i-1) and current tag is t(i):
            F106 = self.V_dict(words[0])*self.T_size + self.T_with_start_dict[tags[2]]
            features.append(F106 + F105_len)
            F106_len = F105_len + self.V_size * self.T_size
            
            # F107: next word is w(i+1) and current tag is t(i):
            F107 = self.V_dict(words[2])*self.T_size + self.T_with_start_dict[tags[2]]
            features.append(F107 + F106_len)
            F107_len = F106_len + self.V_size * self.T_size

            # G1 : is the cuurent word a number and tag is t(i)?
            if words[1][0].isdigit():
                G1 = self.T_with_start_dict[tags[2]]
                features.append(G1 + F107_len)
            G1_len = F107_len + self.T_size

            # G2 : is the cuurent word starts in Upper case and tag is t(i)?
            if words[1][0].isupper() and words[1][0].isalpha():
                G2 = self.T_with_start_dict[tags[2]]
                features.append(G2 + G1_len)
            G2_len = G1_len + self.T_size

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

    def test(self, test_data_path, end=-1, start=0, verbosity=0):
        self.verbosity = verbosity
        (_, _, test, test_tag) = data_preprocessing(test_data_path, 'test')

        corpus = test[start:end]
        corpus_tag = test_tag[start:end]

        # run Viterbi algorithm
        (all_tagged_sentence, all_sentence_tags) = self.predict(corpus, verbosity=verbosity)

        tot_length = 0
        tot_correct = 0
        for i, tag_line in enumerate(all_sentence_tags):
            res = np.sum([x == y for x, y in zip(tag_line, corpus_tag[i])])
            tot_length += len(tag_line)
            tot_correct += res

        tot_accuracy = tot_correct/tot_length
        print("Total accuracy is: ", tot_accuracy)

        return tot_accuracy

    def save_model(self, resultsFn):
        # dump all results:
        with open(resultsFn+'.pkl', 'wb') as f:
            pickle.dump(self, f)



