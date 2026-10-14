from pathlib import Path

import numpy as np
import pandas as pd
from utils_function import cf
from utils_function import utils_function as uf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / 'results'
UTILS_FILE = PROJECT_ROOT / 'code' / 'utils_function' / 'utils_function.py'


def get_Lambda_range(animal, dataset_key):
    if animal == 'mouse' and dataset_key == 'gamanut':
        Lambda_range = [np.arange(0.7, 0.9, 0.025), np.arange(0.5, 1.5, 0.1)]
    elif animal == 'marmoset' and dataset_key == 'majka':
        Lambda_range = [np.arange(2., 2.2, 0.025), np.arange(1., 3., 0.1)]
    elif animal == 'macaque' and dataset_key == 'markov40':
        Lambda_range = [np.arange(9., 10., 0.1), np.arange(7., 14., 0.5)]
    elif animal == 'human' and dataset_key == 'lariviere':
        Lambda_range = [np.arange(22., 23., 0.1), np.arange(10., 30., 0.5)]
    elif animal == 'marmoset' and dataset_key == 'V4':
        Lambda_range = [np.arange(1., 3., 0.1)]
    else:
        raise ValueError('Unknown animal and dataset_key combination')
    Lambda_range = np.concatenate(Lambda_range)
    Lambda_range = np.unique(Lambda_range)
    return Lambda_range


def rm_large_file(timedir):
    simulation_dir = Path(timedir) / 'simulation'
    cf.delete_file(simulation_dir / 'source_points.npy')
    cf.delete_file(simulation_dir / 'target_points.npy')
    cf.delete_file(simulation_dir / 'source_areas.npy')
    cf.delete_file(simulation_dir / 'target_areas.npy')


animal_list = ['mouse', 'marmoset', 'macaque', 'human', 'marmoset']
dataset_key_list = ['gamanut', 'majka', 'markov40', 'lariviere', 'V4']

for animal, dataset_key in zip(animal_list, dataset_key_list):
    print(f'Running for {animal} and {dataset_key}')

    Lambda_range = get_Lambda_range(animal, dataset_key)
    for i in range(len(Lambda_range)):
        for the_seed in range(5):
            basedir = RESULTS_DIR / Path(__file__).stem
            direction_method = 'random'
            length_method = 'exponential'
            direction_params = {}
            length_params = {'Lambda': Lambda_range[i]}
            na = int(8e6)

            connection_model = uf.ConnectionGenerationModel()
            params = {'animal': animal, 'dataset_key': dataset_key, 'animal_and_dataset_key': cf.cat(animal, dataset_key), 'direction_method': direction_method, 'length_method': length_method, 'direction_params': direction_params, 'length_params': length_params, 'na': na, 'process_num': 10, 'the_seed': the_seed}
            connection_model.set_up(params=params, basedir=basedir, code_file_list=[str(Path(__file__).resolve()), str(UTILS_FILE)])
            connection_model.run()
            connection_model.SC_model_dict

            rm_large_file(connection_model.timedir)
