





import os
from collections import defaultdict

import numpy as np
import pandas as pd
from utils_function import cf
from utils_function import utils_function as uf


def get_optimal_params(performance_dict):
    best_lambda = None
    best_avg_corr = -float('inf')


    for lambda_val, corr_list in performance_dict.items():

        avg_corr = sum(corr_list) / len(corr_list)


        if avg_corr > best_avg_corr:
            best_avg_corr = avg_corr
            best_lambda = lambda_val

    return best_lambda





animal_list = ['mouse', 'marmoset', 'macaque', 'human', 'marmoset']
dataset_key_list = ['gamanut', 'majka', 'markov40', 'lariviere', 'V4']
performance_dict = {}
optimal_params_dict = {}
for animal, dataset_key in zip(animal_list, dataset_key_list):
    print(f'Running for {animal} and {dataset_key}')

    performance_dict_this_animal = defaultdict(list)

    model_container = uf.load_model_runs(uf.ConnectionGenerationModel, cf.pj('../../results/para_search', f'{animal}_{dataset_key}'))

    for model in model_container:
        performance_dict_this_animal[model.get_unique_Lambda()].append(model.get_data_model_correlation())

    performance_dict[f'{animal}_{dataset_key}'] = performance_dict_this_animal
    optimal_params_dict[f'{animal}_{dataset_key}'] = {'Lambda': get_optimal_params(performance_dict_this_animal)}
    print(performance_dict_this_animal)
    print(f'Optimal lambda for {animal}_{dataset_key}: {optimal_params_dict[f"{animal}_{dataset_key}"]}')
    print(f'Optimal average correlation: {max([sum(v)/len(v) for v in performance_dict_this_animal.values()])}')
