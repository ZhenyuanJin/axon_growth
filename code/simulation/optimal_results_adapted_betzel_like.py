





import os

import numpy as np
import pandas as pd
from utils_function import cf
from utils_function import utils_function as uf

optimal_params = cf.load_dict('../../results/optimal_params_adapted_betzel_like/optimal_params_adapted_betzel_like')
animal_list = ['mouse', 'marmoset', 'macaque', 'human', 'marmoset']
dataset_key_list = ['gamanut', 'majka', 'markov40', 'lariviere', 'V4']
model_timedir_list = []


optimal_params['marmoset_V4'] = optimal_params['marmoset_majka']

for animal, dataset_key in zip(animal_list, dataset_key_list):
    print(f'Running for {animal} and {dataset_key}')

    basedir = '../../results/'
    basedir = cf.pj(basedir, os.path.splitext(os.path.basename(cf.current_file()))[0])
    Lambda = optimal_params[cf.cat(animal, '_', dataset_key)]['Lambda']
    gamma = optimal_params[cf.cat(animal, '_', dataset_key)]['gamma']

    model = uf.AdaptedBetzelLikeModel()
    params = {'animal': animal, 'dataset_key': dataset_key, 'animal_and_dataset_key': cf.cat(animal, dataset_key), 'Lambda': Lambda, 'gamma': gamma, 'binary_mode': 'intersection'}
    model.set_up(params=params, basedir=basedir, code_file_list=[cf.current_file(), '../utils_function/utils_function.py'])
    model.run()
    model.SC_model_dict
    model_timedir_list.append(model.timedir)

    print(model.get_data_model_correlation())
print('Model directories:', model_timedir_list)
