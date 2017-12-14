# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:39:38 2017

@author: carmelr
"""

from pos_memm.pos_memm import POS_MEMM, load_model, data_preprocessing, analyze_results
import os
import argparse
import sys
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('results_dir', help='output results')
    parser.add_argument('-regularization', type=np.float64, default=0.0005)
    parser.add_argument('-toy', action='store_true')
    parser.add_argument('-baba', action='store_true')
    parser.add_argument('-spelling_threshold', type=int, default=8)
    parser.add_argument('-word_threshold', type=int, default=3)
    parser.add_argument('-parallel', action='store_true')
    parser.add_argument('-verbosity', type=int, default=0)
    parser.add_argument('-mode', type=str, default='complex')
    parser.parse_args(namespace=sys.modules['__main__'])

    project_dir = os.path.dirname(os.path.realpath('__file__'))
    results_path = project_dir + '\\results\\' + results_dir
    log_path = results_path + '\\logs.txt'
    test_path = project_dir + '\\data\\test.wtag'
    comp_path = project_dir + '\\data\\comp.words'
    data_path = project_dir + '\\data\\train.wtag'
    # test_path = project_dir + '\\data\\test_half.wtag'
    # data_path = project_dir + '\\data\\train_and_half_test.wtag'
    pred_path = results_path + '\\predictions.txt'
    analysis_path = results_path + '\\analysis.csv'

    if toy:
        test_path = project_dir + '\\data\\test.wtag'
        comp_path = project_dir + '\\data\\comp.words'
        data_path = project_dir + '\\data\\train.wtag'

        results_dir = 'tmp'
        project_dir = 'D:\\TECHNION\\NLP\\part_of_speech_taging_MEMM'
        results_path = project_dir + '\\results\\' + results_dir
        # test_path = project_dir + '\\data\\test_toy.txt'
        # data_path = project_dir + '\\data\\train_toy.txt'
        # test_path = project_dir + '\\data\\test_half.wtag'
        # data_path = project_dir + '\\data\\train_and_half_test.wtag'
        test_path = project_dir + '\\data\\carmel_test4.txt'
        data_path = project_dir + '\\data\\carmel_test2.txt'
        log_path = results_path + '\\logs.txt'
        analysis_path = results_path + '\\analysis.csv'
        pred_path = results_path + '\\predictions.txt'

        regularization = 0.0005
        verbosity = 1
        word_threshold = 3
        spelling_threshold = 8
        parallel = True
        # log_path = project_dir + '\\results\\' + results_dir + '\\logs_parallel.txt'

    # if baba:
    #     from pos_memm.pos_memm import POS_MEMM
    #     results_dir = 'baba'
    #     project_dir = 'C:\\Users\\amirli\\Desktop\\amir\\part_of_speech_taging_MEMM-carmel\\POS_MEMM'
    #     results_path = project_dir + '\\results\\' + results_dir
    #     test_path = project_dir + '\\data\\debug.wtag'
    #     data_path = project_dir + '\\data\\debug.wtag'
    #     regularization = 0.05
    #     mode = 'complex'
    #     verbosity = 1
    #     use_106_107 = False
    #     log_path = project_dir + '\\results\\' + results_dir + '\\logs.log'

    # save logs
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    with open(log_path, 'w') as f:
        f.writelines('Data path: {}\n'.format(data_path))
        f.writelines('Test path: {}\n'.format(test_path))
        f.writelines('regularization: {}\n'.format(regularization))
        f.writelines('Spelling threshold: {}\n'.format(spelling_threshold))
        f.writelines('Unknown words threshold: {}\n'.format(word_threshold))
        f.writelines('Mode: {}\n'.format(mode))

    model = POS_MEMM()
    model.train(data_path,
                verbosity=verbosity,
                mode=mode,
                regularization=regularization,
                log_path=log_path,
                spelling_threshold=spelling_threshold,
                word_count_threshold=word_threshold,
                use_106_107=True)

    model.save_model(results_path)
    # model = load_model(results_path)
    model.test(test_path,
               verbosity=verbosity,
               parallel=parallel,
               save_results_to_file=results_path,
               log_path=log_path)
    analyze_results(pred_path, test_path, data_path, analysis_path)
