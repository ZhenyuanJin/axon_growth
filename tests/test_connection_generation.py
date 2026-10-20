import types

import numpy as np
import pandas as pd
from utils_function import utils_function as uf


def get_config(worker_count):
    return {
        "animal": "mouse",
        "dataset_key": "gamanut",
        "animal_and_dataset_key": "mouse_gamanut",
        "direction_method": "random",
        "length_method": "exponential",
        "direction_params": {},
        "length_params": {"Lambda": 0.85},
        "na": 2000,
        "process_num": 10,
        "worker_count": worker_count,
        "the_seed": 0,
    }


def test_single_and_multiprocessing_are_identical(tmp_path):
    single = uf.get_connection_generation_results(get_config(1), tmp_path / "single", force_run=True)
    multiple = uf.get_connection_generation_results(get_config(2), tmp_path / "multiple", force_run=True)
    for key in ["ss", "tt", "indices_ss", "indices_tt"]:
        assert np.array_equal(single.get_simulation_value(key), multiple.get_simulation_value(key))
    for key in single.SC_model_dict:
        assert single.SC_model_dict[key].equals(multiple.SC_model_dict[key])


def test_saved_run_can_be_loaded(tmp_path):
    saved = uf.get_connection_generation_results(get_config(1), tmp_path, force_run=True)
    loaded = uf.ConnectionGenerationModel()
    loaded.set_timedir(saved.run_dir)
    assert loaded.params == saved.params
    assert loaded.SC_model_dict.keys() == saved.SC_model_dict.keys()
    assert loaded.get_simulation_value("ss").shape[1] == 2


def test_saved_run_can_skip_large_simulation_arrays(tmp_path):
    saved = uf.get_connection_generation_results(get_config(1), tmp_path, force_run=True)
    loaded = uf.ConnectionGenerationModel()
    loaded.set_timedir(saved.run_dir, load_simulation=False)
    assert loaded.simulation_results == {}
    assert loaded.SC_model_dict.keys() == saved.SC_model_dict.keys()


def test_removed_connections_use_explicit_logged_floor():
    connection_matrix = pd.DataFrame([[1.2, 2.3], [3.4, 4.5]])
    distance_matrix = pd.DataFrame([[0.0, 2.0], [3.0, 0.0]])
    result = uf.get_matrix_without_longrange_connections(connection_matrix, distance_matrix, threshold=1.0, replacement_value=0.9)
    expected = pd.DataFrame([[1.2, 0.9], [0.9, 4.5]])
    assert result.equals(expected)
    assert connection_matrix.iloc[0, 1] == 2.3


def test_model_relative_error_passes_data_before_model(monkeypatch):
    data = pd.DataFrame([[0.0, 1.0], [1.0, 0.0]])
    generated = pd.DataFrame([[0.0, 2.0], [2.0, 0.0]])
    expected = pd.DataFrame([[np.nan, 0.5], [0.5, np.nan]])
    model = uf.ConnectionGenerationModel()
    model.animal = "human"
    model.dataset_key = "lariviere"
    model.animal_info = types.SimpleNamespace(SC_data=data)
    monkeypatch.setattr(model, "get_suitable_SC_model_masked_as_data", lambda: generated)

    def get_relative_error(SC_data, SC_model, animal, dataset_key, diagonal=False):
        assert SC_data is data
        assert SC_model is generated
        assert animal == "human"
        assert dataset_key == "lariviere"
        assert diagonal is False
        return expected

    monkeypatch.setattr(uf, "data_model_relative_error", get_relative_error)
    assert model.get_relative_error().equals(expected)
