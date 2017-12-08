# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:39:38 2017

@author: carmelr
"""

from pos_memm.pos_memm import POS_MEMM, load_model
import argparse
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('resultsFn', help='output results')
    parser.add_argument('-lambda_rate', type=np.float64, default=0.1)
    parser.add_argument('-toy', action='store_true')
    parser.add_argument('-input_path', type=str, default=None)
    parser.add_argument('-threshold', type=int, default=10)
    parser.add_argument('-mode', type=str, default='base')
    parser.parse_args(namespace=sys.modules['__main__'])

    project_dir = 'D:\\TECHNION\\NLP\\part_of_speech_taging_MEMM'
#    project_dir = 'C:\\Users\\amirli\\Desktop\\amir\\part_of_speech_taging_MEMM-carmel\\POS_MEMM'
#    project_dir = os.path.dirname(os.path.realpath('__file__'))
    data_path = project_dir + '\\data\\carmel_test3'
    test_path = project_dir + '\\data\\carmel_test3'
    resultsFn = 'test1'
    mode = 'base'

    model = POS_MEMM()
    model.train(data_path)
    
    
    
    
    