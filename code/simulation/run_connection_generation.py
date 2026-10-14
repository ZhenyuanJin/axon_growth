import os
from pathlib import Path

from utils_function import utils_function as uf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "models"
CONNECTION_COUNT = int(8e6)
LOGICAL_TASK_COUNT = 10
WORKER_COUNT = 10
BASE_SEED = 0
DATASETS = [
    ("mouse", "gamanut"),
    ("marmoset", "majka"),
    ("macaque", "markov40"),
    ("human", "lariviere"),
    ("marmoset", "V4"),
]


def get_run_config(animal, dataset_key):
    dataset_name = f"{animal}_{dataset_key}"
    return {
        "animal": animal,
        "dataset_key": dataset_key,
        "animal_and_dataset_key": dataset_name,
        "direction_method": "random",
        "length_method": "exponential",
        "direction_params": {},
        "length_params": {"Lambda": uf.OPTIMAL_PARAMS[dataset_name]["Lambda"]},
        "na": CONNECTION_COUNT,
        "process_num": LOGICAL_TASK_COUNT,
        "worker_count": WORKER_COUNT,
        "the_seed": BASE_SEED,
    }


def main():
    run_dirs = []
    for animal, dataset_key in DATASETS:
        config = get_run_config(animal, dataset_key)
        print(f"Running connection generation for {animal}_{dataset_key}")
        model = uf.get_connection_generation_results(config, OUTPUT_DIR)
        print(f"Correlation: {model.get_data_model_correlation():.6f}")
        print(f"Output: {model.run_dir}")
        run_dirs.append(model.run_dir)
    print("Run directories:")
    print(os.linesep.join(run_dirs))


if __name__ == "__main__":
    main()
