import hashlib
import json
import os
import pathlib
import platform

import matplotlib
import numpy as np
import pandas as pd
import scipy

ARRAY_NAME_MAP = {
    "ss": "source_points",
    "tt": "target_points",
    "indices_ss": "source_areas",
    "indices_tt": "target_areas",
}
LEGACY_ARRAY_NAME_MAP = {value: key for key, value in ARRAY_NAME_MAP.items()}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def get_config_hash(config):
    payload = json.dumps(config, sort_keys=True, default=_json_default, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def save_json(data, filename):
    filename = pathlib.Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True, default=_json_default)


def load_json(filename):
    with pathlib.Path(filename).open(encoding="utf-8") as file:
        return json.load(file)


def save_simulation_results(results, run_dir):
    simulation_dir = pathlib.Path(run_dir) / "simulation"
    simulation_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for key, value in results.items():
        output_name = ARRAY_NAME_MAP.get(key, key)
        if isinstance(value, pd.DataFrame):
            value.to_csv(simulation_dir / f"{output_name}.csv")
            metadata[key] = {"format": "csv", "name": output_name}
        else:
            array = np.asarray(value)
            np.save(simulation_dir / f"{output_name}.npy", array)
            metadata[key] = {
                "format": "npy",
                "name": output_name,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
    save_json(metadata, simulation_dir / "manifest.json")


def load_simulation_results(run_dir):
    simulation_dir = pathlib.Path(run_dir) / "simulation"
    metadata = load_json(simulation_dir / "manifest.json")
    results = {}
    for key, item in metadata.items():
        path = simulation_dir / f"{item['name']}.{item['format']}"
        if item["format"] == "csv":
            value = pd.read_csv(path, index_col=0)
            value.index = value.index.astype(str)
            value.columns = value.columns.astype(str)
            results[key] = value
        else:
            results[key] = np.load(path, allow_pickle=False)
    return results


def save_connectivity(connectivity, run_dir):
    connectivity_dir = pathlib.Path(run_dir) / "connectivity"
    connectivity_dir.mkdir(parents=True, exist_ok=True)
    for name, value in connectivity.items():
        value.to_csv(connectivity_dir / f"{name}.csv")


def load_connectivity(run_dir):
    connectivity_dir = pathlib.Path(run_dir) / "connectivity"
    if not connectivity_dir.exists():
        raise FileNotFoundError(f"Missing connectivity directory: {connectivity_dir}")
    result = {}
    for path in sorted(connectivity_dir.glob("*.csv")):
        value = pd.read_csv(path, index_col=0)
        value.index = value.index.astype(str)
        value.columns = value.columns.astype(str)
        result[path.stem] = value
    return result


def get_runtime_versions():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }


class ConnectionModel:
    model_name = "connection_model"

    def __init__(self):
        self.simulation_results = {}
        self.params = {}
        self.force_run = False

    def set_default_params(self):
        return None

    def set_params(self, params):
        self.params = dict(params)
        for key, value in self.params.items():
            setattr(self, key, value)

    def set_up(self, params, basedir, code_file_list=None, force_run=False, **kwargs):
        self.set_default_params()
        self.set_params(params)
        self.basedir = os.path.abspath(basedir)
        self.force_run = force_run
        dataset_name = f"{self.animal}_{self.dataset_key}"
        self.run_dir = os.path.join(self.basedir, self.model_name, dataset_name, get_config_hash(self.params))
        self.timedir = self.run_dir
        self.outcomes_dir = self.run_dir
        self.figs_dir = os.path.join(self.run_dir, "figures")

    def set_timedir(self, run_dir):
        self.run_dir = os.path.abspath(run_dir)
        self.timedir = self.run_dir
        self.outcomes_dir = self.run_dir
        self.figs_dir = os.path.join(self.run_dir, "figures")
        manifest = load_json(os.path.join(self.run_dir, "manifest.json"))
        self.set_default_params()
        self.set_params(manifest["config"])
        self.simulation_results = load_simulation_results(self.run_dir)

    def run(self):
        manifest_path = os.path.join(self.run_dir, "manifest.json")
        if os.path.exists(manifest_path) and not self.force_run:
            self.simulation_results = load_simulation_results(self.run_dir)
            return self.simulation_results
        os.makedirs(self.run_dir, exist_ok=True)
        self.run_detail()
        save_simulation_results(self.simulation_results, self.run_dir)
        save_json(
            {
                "model": self.model_name,
                "config": self.params,
                "config_hash": get_config_hash(self.params),
                "versions": get_runtime_versions(),
            },
            manifest_path,
        )
        return self.simulation_results

    def run_detail(self):
        raise NotImplementedError

    def get_simulation_value(self, key):
        if key in self.simulation_results:
            return self.simulation_results[key]
        legacy_key = LEGACY_ARRAY_NAME_MAP.get(key)
        if legacy_key in self.simulation_results:
            return self.simulation_results[legacy_key]
        raise KeyError(key)


def load_model_runs(model_class, basedir):
    basedir = pathlib.Path(basedir)
    instances = []
    for manifest_path in sorted(basedir.glob("**/manifest.json")):
        if manifest_path.parent.name in {"simulation", "connectivity"}:
            continue
        instance = model_class()
        instance.set_timedir(manifest_path.parent)
        instances.append(instance)
    return instances
