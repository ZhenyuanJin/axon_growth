import os

import numpy as np
import pandas as pd
from utils_function import cf
from utils_function import utils_function as uf


def get_Lambda_range(animal, dataset_key):
    animal_info = uf.AnimalInfo(animal=animal, dataset_key=dataset_key)
    distance_3D = animal_info.distance_3D
    max_distance = np.max(distance_3D.values)
    if animal == 'mouse' and dataset_key == 'gamanut':
        Lambda_range = [np.linspace(0.1, max_distance/2, 30)]
    elif animal == 'marmoset' and dataset_key == 'majka':
        Lambda_range = [np.linspace(0.5, max_distance/2, 30)]
    elif animal == 'macaque' and dataset_key == 'markov40':
        Lambda_range = [np.linspace(0.5, max_distance/2, 30)]
    elif animal == 'human' and dataset_key == 'lariviere':
        Lambda_range = [np.linspace(5., max_distance/2, 30)]
    elif animal == 'marmoset' and dataset_key == 'V4':
        Lambda_range = [np.linspace(0.5, max_distance/2, 30)]
    else:
        raise ValueError('Unknown animal and dataset_key combination')
    Lambda_range = np.concatenate(Lambda_range)
    Lambda_range = np.unique(Lambda_range)
    return Lambda_range




animal_list = ['mouse', 'marmoset', 'macaque', 'human']
dataset_key_list = ['gamanut', 'majka', 'markov40', 'lariviere']

for animal, dataset_key in zip(animal_list, dataset_key_list):
    print(f'Running for {animal} and {dataset_key}')

    Lambda_range = get_Lambda_range(animal, dataset_key)
    for i in range(len(Lambda_range)):
        basedir = '../../results/'
        basedir = cf.pj(basedir, os.path.splitext(os.path.basename(cf.current_file()))[0])
        length_params = {'Lambda': Lambda_range[i]}

        model = uf.AreaEDR3DModel()
        params = {'animal': animal, 'dataset_key': dataset_key, 'animal_and_dataset_key': cf.cat(animal, dataset_key), 'length_params': length_params}
        model.set_up(params=params, basedir=basedir, code_file_list=[cf.current_file(), '../utils_function/utils_function.py'])
        model.run()
        model.SC_model_dict
