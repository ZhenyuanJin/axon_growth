import numpy as np
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
