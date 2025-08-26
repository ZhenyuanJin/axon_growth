import sys
sys.path.append('../utils_function/')
sys.path.append(r'/data/zyjin/common_func/')
import utils_function as uf
import common_functions as cf
import pandas as pd
import numpy as np
import os


def get_Lambda_range(animal, dataset_key):
    if animal == 'mouse' and dataset_key == 'gamanut':
        Lambda_range = np.arange(0.7, 0.9, 0.25)
        return Lambda_range
    elif animal == 'marmoset' and dataset_key == 'majka':
        return np.arange(10, 31, 2)
    elif animal == 'macaque' and dataset_key == 'markov40':
        return np.arange(20, 51, 3)
    elif animal == 'human' and dataset_key == 'lariviere':
        return np.arange(30, 81, 5)
    elif animal == 'marmoset' and dataset_key == 'V4':
        return np.arange(10, 31, 2)
    else:
        raise ValueError('Unknown animal and dataset_key combination')


animal_list = ['mouse', 'marmoset', 'macaque', 'human', 'marmoset']
dataset_key_list = ['gamanut', 'majka', 'markov40', 'lariviere', 'V4']

for animal, dataset_key in zip(animal_list, dataset_key_list):
    print(f'Running for {animal} and {dataset_key}')

    Lambda_range = get_Lambda_range(animal, dataset_key)
    for i in range(len(Lambda_range)):
        basedir = '../../results/'
        basedir = cf.pj(basedir, os.path.splitext(os.path.basename(cf.current_file()))[0])
        direction_method = 'random'
        length_method = 'exponential'
        direction_params = {}
        length_params = {'Lambda': Lambda_range[i]}
        na = int(8e6)

        axon_growth_model = uf.AxonGrowthModel()
        params = {'animal': animal, 'dataset_key': dataset_key, 'animal_and_dataset_key': cf.cat(animal, dataset_key), 'direction_method': direction_method, 'length_method': length_method, 'direction_params': direction_params, 'length_params': length_params, 'na': na, 'process_num': 10}
        axon_growth_model.set_up(params=params, basedir=basedir, code_file_list=[cf.current_file(), '../utils_function/utils_function.py'])
        axon_growth_model.run()
        axon_growth_model.SC_model_dict
        model_timedir_list.append(axon_growth_model.timedir)