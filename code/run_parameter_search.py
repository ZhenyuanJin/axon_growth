from itertools import product
from pathlib import Path

import numpy as np
from utils_function import cf
from utils_function import utils_function as uf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "parameter_search"
WORKER_COUNT = 10
CONNECTION_COUNT = int(8e6)
SEEDS = range(5)
DATASETS = [
    ("mouse", "gamanut", np.arange(0.7, 0.9, 0.025)),
    ("marmoset", "majka", np.arange(2.0, 2.2, 0.025)),
    ("macaque", "markov40", np.arange(9.0, 10.0, 0.1)),
    ("human", "lariviere", np.arange(22.0, 23.0, 0.1)),
]


def get_task_table():
    tasks = []
    for animal, dataset_key, lambda_values in DATASETS:
        for task_id, (lambda_value, seed) in enumerate(product(lambda_values, SEEDS)):
            dataset_name = f"{animal}_{dataset_key}"
            tasks.append({
                "task_id": f"{dataset_name}_{task_id:04d}",
                "config": {
                    "animal": animal,
                    "dataset_key": dataset_key,
                    "animal_and_dataset_key": dataset_name,
                    "direction_method": "random",
                    "length_method": "exponential",
                    "direction_params": {},
                    "length_params": {"Lambda": float(lambda_value)},
                    "na": CONNECTION_COUNT,
                    "process_num": 10,
                    "worker_count": 1,
                    "the_seed": seed,
                },
            })
    task_ids = [task["task_id"] for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Task identifiers must be unique.")
    return tasks


def run_task(task):
    model = uf.get_connection_generation_results(task["config"], OUTPUT_DIR)
    return {"task_id": task["task_id"], "run_dir": model.run_dir}


def main():
    task_table = get_task_table()
    results = cf.multi_process(
        process_num=WORKER_COUNT,
        func=run_task,
        kwargs_list=[{"task": task} for task in task_table],
        func_name="connection parameter search",
    )
    results = sorted(results, key=lambda result: result["task_id"])
    print(f"Completed {len(results)} tasks")


if __name__ == "__main__":
    main()
