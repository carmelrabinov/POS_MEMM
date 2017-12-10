# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:39:38 2017

@author: carmelr
"""

from pos_memm.pos_memm import POS_MEMM, load_model, data_preprocessing
import os
import argparse
import sys
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('results_dir', help='output results')
    parser.add_argument('-regularization', type=float, default=0.02)
    parser.add_argument('-toy', action='store_true')
    parser.add_argument('-baba', action='store_true')
    parser.add_argument('-threshold', type=int, default=10)
    parser.add_argument('-verbosity', type=int, default=0)
    parser.add_argument('-mode', type=str, default='complex')
    parser.parse_args(namespace=sys.modules['__main__'])

    project_dir = os.path.dirname(os.path.realpath('__file__'))
#    results_path = project_dir+ '\\results\\' + results_dir
    test_path = project_dir + '\\data\\test.wtag'
    comp_path = project_dir + '\\data\\comp.words'
    data_path = project_dir + '\\data\\train.wtag'

    if toy:
        results_dir = 'on_all_train_complex'
        project_dir = 'D:\\TECHNION\\NLP\\part_of_speech_taging_MEMM'
        results_path = project_dir + '\\results\\' + results_dir
        test_path = project_dir + '\\data\\carmel_test3.txt'
        data_path = project_dir + '\\data\\carmel_test3.txt'
        regularization = 0.05
        mode = 'complex'
        verbosity = 1

    if baba:
        from pos_memm.pos_memm_106_107 import POS_MEMM as POS_MEMM
#        results_dir = 'baba'
#        project_dir = 'C:\\Users\\amirli\\Desktop\\amir\\part_of_speech_taging_MEMM-carmel\\POS_MEMM'
#        results_path = project_dir + '\\results\\' + results_dir
#        test_path = project_dir + '\\data\\debug.wtag'
#        data_path = project_dir + '\\data\\debug.wtag'
#        regularization = 0.05
#        mode = 'complex'
#        verbosity = 1
        
    model = POS_MEMM()
    t0 = time.time()
    print("Start traininig in {} mode, verbosity: {}".format(mode, verbosity))
    model.train(data_path, verbosity=verbosity, mode=mode, regularization=regularization)
    t1 = time.time()
    train_time = (t1-t0)/60
    res = model.test(test_path, verbosity=verbosity, save_results_to_file=results_path)
    t2 = time.time()
    test_time = (t2-t1)/60
    print("\n.\n.\n.\n.\n")
    print("Done Training in: {} minutes".format(train_time))
    print("Done Testing in: {} minutes".format(test_time))
    model.save_model(results_path)

#    with open(results_path + '\\logs.txt', 'w') as f:
#        f.writelines('Accuracy: {}\n'.format(res[0]))
#        f.writelines('Data path: {}\n'.format(data_path))
#        f.writelines('Test path: {}\n'.format(test_path))
#        f.writelines('regularization: {}\n'.format(regularization))
#        f.writelines('Spelling threshold: {}\n'.format(threshold))
#        f.writelines('Mode: {}\n'.format(mode))