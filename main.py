# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:39:38 2017

@author: Carmel Rabinovitz, Amir Livne
"""

from pos_memm.pos_memm import POS_MEMM, load_model, data_preprocessing, analyze_results, save_comp_results
import os
import argparse
import sys
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('results_dir', help='output results')
    parser.add_argument('-oparation_mode', type=str, default='comp')  # option are comp or test
    parser.add_argument('-trained_model', action='store_true')
    parser.add_argument('-data_path', type=str, default=None)
    parser.add_argument('-comp_path', type=str, default=None)
    parser.add_argument('-test_path', type=str, default=None)
    parser.add_argument('-regularization', type=np.float64, default=5e-05)
    parser.add_argument('-spelling_threshold', type=int, default=5)
    parser.add_argument('-word_threshold', type=int, default=3)
    parser.add_argument('-end', type=int, default=0)
    parser.add_argument('-parallel', action='store_true')
    parser.add_argument('-verbosity', type=int, default=0)
    parser.add_argument('-mode', type=str, default='complex')
    parser.parse_args(namespace=sys.modules['__main__'])

    project_dir = os.path.dirname(os.path.realpath('__file__'))
    if comp_path is None:
        comp_path = project_dir + '\\data\\comp.words'
    if test_path is None:
        test_path = project_dir + '\\data\\test.wtag'
        # test_path = project_dir + '\\data\\debug2.txt'
    if data_path is None:
        # data_path = project_dir + '\\data\\train.wtag'
        # data_path = project_dir + '\\data\\test.wtag'
        # data_path = project_dir + '\\data\\debug.txt'
        data_path = project_dir + '\\data\\train_and_test.wtag'
    results_path = project_dir + '\\results\\' + results_dir
    log_path = results_path + '\\logs.txt'
    pred_path = results_path + '\\predictions.txt'
    analysis_path = results_path + '\\analysis.csv'

    # save logs
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    with open(log_path, 'w') as f:
        f.writelines('Data path: {}\n'.format(data_path))
        f.writelines('Test path: {}\n'.format(test_path))
        f.writelines('Comp path: {}\n'.format(comp_path))

    if not trained_model:
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

    if oparation_mode == 'test':
        if trained_model:
            model = load_model(results_path)
        model.test(test_path,
                   verbosity=verbosity,
                   parallel=parallel,
                   save_results_to_file=results_path,
                   log_path=log_path,
                   end=end)
        analyze_results(pred_path, test_path, data_path, analysis_path)

    elif oparation_mode == 'comp':
        if trained_model:
            model = load_model(results_path)
        (_, _, comp, _) = data_preprocessing(comp_path, 'comp')
        (all_tagged_sentence, all_sentence_tags) = model.predict_parallel(corpus=comp,
                                                                          verbosity=verbosity,
                                                                          log_path=log_path)
        save_comp_results(all_tagged_sentence, results_path, comp_path)



