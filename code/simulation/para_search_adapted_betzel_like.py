import sys
sys.path.append('../utils_function/')
sys.path.append(r'/data/zyjin/common_func/')
import utils_function as uf
import common_functions as cf
import pandas as pd
import numpy as np
import os
from itertools import product


def get_Lambda_range(animal, dataset_key):
    optimal_params = cf.load_dict('../../results/optimal_params_area_EDR_3d/optimal_params_area_EDR_3d')
    return [optimal_params[cf.cat(animal, dataset_key)]['Lambda']]


def get_gamma_range():
    return np.linspace(-2., 2.0, 21)


def main(params):
    basedir = '../../results/'
    basedir = cf.pj(basedir, os.path.splitext(os.path.basename(cf.current_file()))[0])

    model = uf.AdaptedBetzelLikeModel()
    model.set_up(params=params, basedir=basedir, code_file_list=[cf.current_file(), '../utils_function/utils_function.py'])
    model.run()
    model.SC_model_dict


# animal_list = ['mouse', 'marmoset', 'macaque', 'human', 'marmoset']
# dataset_key_list = ['gamanut', 'majka', 'markov40', 'lariviere', 'V4']
animal_list = ['mouse', 'marmoset', 'macaque', 'human']
dataset_key_list = ['gamanut', 'majka', 'markov40', 'lariviere']
params_list = []
for animal, dataset_key in zip(animal_list, dataset_key_list):
    print(f'Running for {animal} and {dataset_key}')

    Lambda_range = get_Lambda_range(animal, dataset_key)
    gamma_range = get_gamma_range()
    for Lambda, gamma in product(Lambda_range, gamma_range):
        params = {'animal': animal, 'dataset_key': dataset_key, 'animal_and_dataset_key': cf.cat(animal, dataset_key), 'Lambda': Lambda, 'gamma': gamma, 'binary_mode': 'intersection'}
        params_list.append({'params': params})

cf.multi_process(process_num=10, func=main, kwargs_list=params_list)