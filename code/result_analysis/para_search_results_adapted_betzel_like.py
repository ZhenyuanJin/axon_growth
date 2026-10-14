





from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from utils_function import cf
from utils_function import utils_function as uf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / 'results'
OPTIMAL_PARAMS_PATH = str(RESULTS_DIR / 'optimal_params_adapted_betzel_like' / 'optimal_params_adapted_betzel_like')


def get_optimal_params(performance_dict):
    best_avg_corr = -float('inf')

    for param, corr_list in performance_dict.items():
        avg_corr = sum(corr_list) / len(corr_list)
        std_corr = np.std(corr_list)



        if avg_corr > best_avg_corr:
            best_avg_corr = avg_corr
            best_param = param

    return best_param







animal_list = ['mouse', 'marmoset', 'macaque', 'human']
dataset_key_list = ['gamanut', 'majka', 'markov40', 'lariviere']
performance_dict = {}
optimal_params_dict = {}
for animal, dataset_key in zip(animal_list, dataset_key_list):
    print(f'Running for {animal} and {dataset_key}')

    performance_dict_this_animal = defaultdict(list)

    model_container = uf.load_model_runs(uf.AdaptedBetzelLikeModel, RESULTS_DIR / 'para_search_adapted_betzel_like' / f'{animal}_{dataset_key}')

    for model in model_container:
        performance_dict_this_animal[(model.Lambda, model.gamma)].append(model.get_data_model_correlation())
    print(performance_dict_this_animal)
    performance_dict[f'{animal}_{dataset_key}'] = performance_dict_this_animal
    best_Lambda, best_gamma = get_optimal_params(performance_dict_this_animal)
    optimal_params_dict[f'{animal}_{dataset_key}'] = {'Lambda': best_Lambda, 'gamma': best_gamma}
    print(performance_dict_this_animal)
    print(f'Optimal lambda for {animal}_{dataset_key}: {optimal_params_dict[f"{animal}_{dataset_key}"]}')
    print(f'Optimal average correlation: {max([sum(v)/len(v) for v in performance_dict_this_animal.values()])}')





if cf.check_all_file_exist_with_any_extension(OPTIMAL_PARAMS_PATH):
    print('already saved, please check whether you want to overwrite it')
else:
    cf.save_dict(optimal_params_dict, OPTIMAL_PARAMS_PATH)
