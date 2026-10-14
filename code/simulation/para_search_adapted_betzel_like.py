from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from utils_function import cf
from utils_function import utils_function as uf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / 'results'
UTILS_FILE = PROJECT_ROOT / 'code' / 'utils_function' / 'utils_function.py'


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


def get_gamma_range():
    return np.linspace(-2., 2.0, 21)


def main(params):
    basedir = RESULTS_DIR / Path(__file__).stem

    model = uf.AdaptedBetzelLikeModel()
    model.set_up(params=params, basedir=basedir, code_file_list=[str(Path(__file__).resolve()), str(UTILS_FILE)])
    model.run()
    model.SC_model_dict




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
