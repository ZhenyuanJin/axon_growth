





from pathlib import Path

import numpy as np
import pandas as pd
from utils_function import cf
from utils_function import utils_function as uf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / 'results'
UTILS_FILE = PROJECT_ROOT / 'code' / 'utils_function' / 'utils_function.py'

optimal_params = cf.load_dict(str(RESULTS_DIR / 'optimal_params_adapted_betzel_like' / 'optimal_params_adapted_betzel_like'))
animal_list = ['mouse', 'marmoset', 'macaque', 'human', 'marmoset']
dataset_key_list = ['gamanut', 'majka', 'markov40', 'lariviere', 'V4']
model_timedir_list = []


optimal_params['marmoset_V4'] = optimal_params['marmoset_majka']

for animal, dataset_key in zip(animal_list, dataset_key_list):
    print(f'Running for {animal} and {dataset_key}')

    basedir = RESULTS_DIR / Path(__file__).stem
    Lambda = optimal_params[cf.cat(animal, '_', dataset_key)]['Lambda']
    gamma = optimal_params[cf.cat(animal, '_', dataset_key)]['gamma']

    model = uf.AdaptedBetzelLikeModel()
    params = {'animal': animal, 'dataset_key': dataset_key, 'animal_and_dataset_key': cf.cat(animal, dataset_key), 'Lambda': Lambda, 'gamma': gamma, 'binary_mode': 'intersection'}
    model.set_up(params=params, basedir=basedir, code_file_list=[str(Path(__file__).resolve()), str(UTILS_FILE)])
    model.run()
    model.SC_model_dict
    model_timedir_list.append(model.timedir)

    print(model.get_data_model_correlation())
print('Model directories:', model_timedir_list)
