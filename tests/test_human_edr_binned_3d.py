from pathlib import Path

import numpy as np
from human_edr_binned_3d import model
from utils_function import cf


def get_catalog(tmp_path):
    coordinates = np.asarray([
        [0.00, 0.0, 0.0],
        [0.12, 0.0, 0.0],
        [0.37, 0.0, 0.0],
        [0.91, 0.0, 0.0],
    ])
    return model.build_bin_catalog(
        project_root=tmp_path,
        cache_dir=Path(tmp_path) / "catalog",
        bin_width=0.1,
        coordinates=coordinates,
        area_ids=np.asarray([0, 0, 1, 1]),
        area_names=["A", "B"],
    )


def load_area_samples(cache_dir, probabilities, seed):
    catalog = model.load_bin_catalog(cache_dir)
    return model.get_area_connection_matrix(catalog, probabilities, 2000, seed).values


def test_catalog_counts_every_ordered_nonself_pair(tmp_path):
    catalog = get_catalog(tmp_path)
    point_count = catalog["metadata"]["vertex_count"]
    assert int(np.sum(catalog["global_bin_counts"])) == point_count * (point_count - 1)
    assert np.array_equal(
        np.sum(catalog["area_pair_bin_counts"], axis=(1, 2), dtype=np.int64),
        catalog["global_bin_counts"],
    )


def test_each_bin_has_exact_integrated_edr_mass(tmp_path):
    catalog = get_catalog(tmp_path)
    Lambda = 0.8
    bin_probabilities = model.get_bin_probabilities(catalog, Lambda)
    pair_weights = model.get_pair_bin_weights(catalog, Lambda)
    recovered = pair_weights * np.asarray(catalog["global_bin_counts"])
    assert np.isclose(np.sum(bin_probabilities), 1.0, rtol=0.0, atol=1e-12)
    assert np.allclose(recovered, bin_probabilities, rtol=0.0, atol=1e-15)
    edges = np.asarray(catalog["bin_edges"])
    expected = np.exp(-edges[:-1] / Lambda) - np.exp(-edges[1:] / Lambda)
    expected[np.asarray(catalog["global_bin_counts"]) == 0] = 0
    expected /= np.sum(expected)
    assert np.allclose(bin_probabilities, expected, rtol=0.0, atol=1e-15)


def test_area_probabilities_and_vertex_samples_are_consistent(tmp_path):
    catalog = get_catalog(tmp_path)
    Lambda = 0.8
    probabilities = model.get_area_probability_matrix(catalog, Lambda)
    assert np.isclose(np.sum(probabilities), 1.0, rtol=0.0, atol=1e-12)
    coordinates = np.asarray(catalog["coordinates"])
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
    bin_indices = model._get_bin_index(
        distances, catalog["metadata"]["bin_width_mm"], catalog["metadata"]["bin_count"]
    )
    pair_weights = model.get_pair_bin_weights(catalog, Lambda)[bin_indices]
    np.fill_diagonal(pair_weights, 0.0)
    expected = np.zeros((2, 2), dtype=np.float64)
    area_ids = np.asarray(catalog["area_ids"])
    for source in range(len(coordinates)):
        for target in range(len(coordinates)):
            expected[area_ids[source], area_ids[target]] += pair_weights[source, target]
    assert np.allclose(probabilities, expected, rtol=0.0, atol=1e-15)
    counts = model.get_area_connection_matrix(catalog, probabilities, 5000, seed=3)
    first = model.get_vertex_samples(catalog, counts, Lambda=0.8, seed=7)
    second = model.get_vertex_samples(catalog, counts, Lambda=0.8, seed=7)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert np.all(first[0] != first[1])
    observed = np.zeros((2, 2), dtype=np.int64)
    area_ids = np.asarray(catalog["area_ids"])
    np.add.at(observed, (area_ids[first[0]], area_ids[first[1]]), 1)
    assert np.array_equal(observed, counts.values)


def test_single_and_multiprocessing_area_samples_are_identical(tmp_path):
    catalog = get_catalog(tmp_path)
    cache_dir = Path(tmp_path) / "catalog"
    probabilities = model.get_area_probability_matrix(catalog, Lambda=0.8)
    expected = model.get_area_connection_matrix(catalog, probabilities, 2000, seed=4).values
    results = cf.multi_process(
        process_num=2,
        func=load_area_samples,
        kwargs_list=[
            {"cache_dir": cache_dir, "probabilities": probabilities, "seed": 4},
            {"cache_dir": cache_dir, "probabilities": probabilities, "seed": 4},
        ],
        func_name="binned human EDR reproducibility test",
    )
    for result in results:
        assert np.array_equal(result, expected)
