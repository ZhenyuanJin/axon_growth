from pathlib import Path

import numpy as np

from utils_function import utils_function as uf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "experiments" / "target_conditioned_angle_edr"
DATASETS = [("mouse", "gamanut"), ("marmoset", "majka"), ("macaque", "markov40"),
            ("human", "lariviere"), ("marmoset", "V4")]
BASE_SEED = 20261029
REPEAT_COUNT = 5
NA_PER_TARGET = 20_000
ANGLE_KDE_BANDWIDTH_DEGREES = 15.0
MAX_ITERATION = 10
WORKER_COUNT = 32
DIRECTION_METHODS = ["random", "SC_angle_kde"]
INITIAL_GRID_INDICES = list(range(6, 29))
GRID_EXTENSION_COUNT = 6
MAX_GRID_EXTENSIONS = 6
METRIC_OBJECTIVES = {
    "correlation": "max", "log_correlation": "max", "rmae_both": "min",
    "median_target_correlation": "max", "mean_target_correlation": "max",
}


def get_lambda(animal, dataset_key, grid_index):
    """Return a deterministic positive float Lambda on a species-scaled 0.125 grid."""
    reference = uf.OPTIMAL_PARAMS[f"{animal}_{dataset_key}"]["Lambda"]
    return float(np.round(grid_index * 0.125 * reference / uf.OPTIMAL_PARAMS["marmoset_majka"]["Lambda"], 12))
