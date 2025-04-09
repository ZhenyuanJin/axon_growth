import sys
sys.path.append('../utils_function/')
sys.path.append(r'/data/zyjin/common_func/')
import utils_function as uf
import common_functions as cf
import pandas as pd
import numpy as np
import os

optimal_params = cf.load_dict('../../results/optimal_params/optimal_params')
animal_list = ['mouse', 'marmoset', 'macaque', 'human']
dataset_key_list = ['gamanut', 'majka', 'markov40', 'lariviere']
model_timedir_list = []

for animal, dataset_key in zip(animal_list, dataset_key_list):
    print(f'Running for {animal} and {dataset_key}')

    basedir = '../../results/'
    basedir = cf.pj(basedir, os.path.splitext(os.path.basename(cf.current_file()))[0])
    direction_method = 'random'
    length_method = 'exponential'
    direction_params = {}
    length_params = {'Lambda': optimal_params[cf.cat(animal, dataset_key)]['Lambda']}
    na = int(8e6)

    axon_growth_model = uf.AxonGrowthModel()
    params = {'animal': animal, 'dataset_key': dataset_key, 'animal_and_dataset_key': cf.cat(animal, dataset_key), 'direction_method': direction_method, 'length_method': length_method, 'direction_params': direction_params, 'length_params': length_params, 'na': na, 'process_num': 10}
    axon_growth_model.set_up(params=params, basedir=basedir, code_file_list=[cf.current_file(), '../utils_function/utils_function.py'])
    axon_growth_model.run()
    axon_growth_model.SC_model_dict
    model_timedir_list.append(axon_growth_model.timedir)
print('Model directories:', model_timedir_list)