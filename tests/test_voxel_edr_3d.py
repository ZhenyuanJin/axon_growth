from pathlib import Path

import numpy as np
from utils_function import cf
from voxel_edr_3d import model


def get_synthetic_catalog(tmp_path):
    coordinates = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    area_ids = np.asarray([0, 0, 1, 1])
    return model.build_displacement_catalog(
        project_root=Path(tmp_path),
        cache_dir=Path(tmp_path) / "catalog",
        coordinates=coordinates,
        area_ids=area_ids,
        area_names=["A", "B"],
        spacing=np.ones(3),
    )


def get_samples_from_cache(cache_dir, seed):
    catalog = model.load_displacement_catalog(cache_dir)
    return model.get_connection_samples(catalog, Lambda=1.2, connection_count=2000, seed=seed)


def test_implicit_probabilities_match_dense_pair_formula(tmp_path):
    catalog = get_synthetic_catalog(tmp_path)
    Lambda = 1.7
    displacement_probabilities = model.get_displacement_probabilities(catalog, Lambda)
    pair_probabilities = {}
    for displacement, probability, pair_count in zip(
        catalog["displacements"], displacement_probabilities, catalog["displacement_counts"]
    ):
        sources, targets = model._get_valid_pair_indices(catalog, displacement)
        assert len(sources) == pair_count
        for source, target in zip(sources, targets):
            pair_probabilities[(int(source), int(target))] = probability / pair_count

    coordinates = np.asarray(catalog["coordinates"])
    distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    nonzero_distances = distances[distances > 0]
    unique_distances, counts = np.unique(nonzero_distances, return_counts=True)
    normalizer = np.sum(np.exp(-unique_distances / Lambda))
    expected = {}
    for source in range(len(coordinates)):
        for target in range(len(coordinates)):
            if source == target:
                continue
            distance = distances[source, target]
            count = counts[np.flatnonzero(np.isclose(unique_distances, distance))[0]]
            expected[(source, target)] = np.exp(-distance / Lambda) / count / normalizer
    assert np.isclose(sum(pair_probabilities.values()), 1.0, rtol=0.0, atol=1e-12)
    assert pair_probabilities.keys() == expected.keys()
    for pair in expected:
        assert np.isclose(pair_probabilities[pair], expected[pair], rtol=0.0, atol=1e-12)


def test_sampling_is_deterministic_and_has_no_self_pairs(tmp_path):
    catalog = get_synthetic_catalog(tmp_path)
    first = model.get_connection_samples(catalog, Lambda=1.0, connection_count=5000, seed=12)
    second = model.get_connection_samples(catalog, Lambda=1.0, connection_count=5000, seed=12)
    different = model.get_connection_samples(catalog, Lambda=1.0, connection_count=5000, seed=13)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert not (np.array_equal(first[0], different[0]) and np.array_equal(first[1], different[1]))
    assert np.all(first[0] != first[1])
    matrix = model.get_connection_matrix(catalog, first[0], first[1])
    assert matrix.shape == (2, 2)
    assert int(matrix.values.sum()) == 5000


def test_catalog_counts_every_ordered_distinct_pair(tmp_path):
    catalog = get_synthetic_catalog(tmp_path)
    voxel_count = catalog["metadata"]["voxel_count"]
    assert int(np.sum(catalog["displacement_counts"])) == voxel_count * (voxel_count - 1)
    assert np.isclose(np.sum(model.get_displacement_probabilities(catalog, 0.8)), 1.0)


def test_single_and_multiprocessing_samples_are_identical(tmp_path):
    catalog = get_synthetic_catalog(tmp_path)
    cache_dir = Path(tmp_path) / "catalog"
    expected = model.get_connection_samples(catalog, Lambda=1.2, connection_count=2000, seed=4)
    results = cf.multi_process(
        process_num=2,
        func=get_samples_from_cache,
        kwargs_list=[{"cache_dir": cache_dir, "seed": 4}, {"cache_dir": cache_dir, "seed": 4}],
        func_name="voxel EDR reproducibility test",
    )
    for result in results:
        assert np.array_equal(result[0], expected[0])
        assert np.array_equal(result[1], expected[1]), "Target samples differ across process modes"
