





from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from utils_function import cf
from utils_function import utils_function as uf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / 'results'
OPTIMAL_PARAMS_PATH = str(RESULTS_DIR / 'optimal_params_area_EDR_3d' / 'optimal_params_area_EDR_3d')


def get_optimal_params(performance_dict):
    best_lambda = None
    best_avg_corr = -float('inf')


    for lambda_val, corr_list in performance_dict.items():

        avg_corr = sum(corr_list) / len(corr_list)


        if avg_corr > best_avg_corr:
            best_avg_corr = avg_corr
            best_lambda = lambda_val

    return best_lambda







animal_list = ['mouse', 'marmoset', 'macaque', 'human']
dataset_key_list = ['gamanut', 'majka', 'markov40', 'lariviere']
performance_dict = {}
optimal_params_dict = {}
for animal, dataset_key in zip(animal_list, dataset_key_list):
    print(f'Running for {animal} and {dataset_key}')

    performance_dict_this_animal = defaultdict(list)

    model_container = uf.load_model_runs(uf.AreaEDR3DModel, RESULTS_DIR / 'para_search_area_EDR_3d' / f'{animal}_{dataset_key}')

    for model in model_container:
        performance_dict_this_animal[model.get_unique_Lambda()].append(model.get_data_model_correlation())

    performance_dict[f'{animal}_{dataset_key}'] = performance_dict_this_animal
    optimal_params_dict[f'{animal}_{dataset_key}'] = {'Lambda': get_optimal_params(performance_dict_this_animal)}
    print(performance_dict_this_animal)
    print(f'Optimal lambda for {animal}_{dataset_key}: {optimal_params_dict[f"{animal}_{dataset_key}"]}')
    print(f'Optimal average correlation: {max([sum(v)/len(v) for v in performance_dict_this_animal.values()])}')

if cf.check_all_file_exist_with_any_extension(OPTIMAL_PARAMS_PATH):
    print('already saved, please check whether you want to overwrite it')
else:
    cf.save_dict(optimal_params_dict, OPTIMAL_PARAMS_PATH)
