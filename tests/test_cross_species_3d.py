from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse.csgraph
from cross_species_3d import analysis
from cross_species_3d import model
from cross_species_3d import preprocess
from utils_function import cf
from utils_function import utils_function as uf


def get_voxel_catalog(tmp_path):
    coordinates = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    return model.build_displacement_catalog(
        tmp_path, Path(tmp_path) / "voxel_catalog", coordinates=coordinates,
        area_ids=np.asarray([0, 0, 1, 1]), area_names=["A", "B"], spacing=np.ones(3),
    )


def get_human_catalog(tmp_path):
    coordinates = np.asarray([[0, 0, 0], [0.12, 0, 0], [0.37, 0, 0], [0.91, 0, 0]], dtype=float)
    return model.build_human_bin_catalog(
        tmp_path, Path(tmp_path) / "human_catalog", bin_width=0.1,
        coordinates=coordinates, area_ids=np.asarray([0, 0, 1, 1]), area_names=["A", "B"],
    )


def load_voxel_samples(cache_dir, seed):
    catalog = model.load_displacement_catalog(cache_dir)
    return model.get_connection_samples(catalog, 1.2, 2000, seed)


def load_human_samples(cache_dir, probabilities, seed):
    catalog = model.load_human_bin_catalog(cache_dir)
    return model.get_binned_area_connection_matrix(catalog, probabilities, 2000, seed).values


def test_3d_matrices_follow_available_area_centers(monkeypatch, tmp_path):
    class Info:
        full_name = ["A", "B", "excluded"]
        area_center_dict_3D = {"A": np.asarray([0, 0, 0]), "B": np.asarray([3, 4, 0])}
        distance_matrix_dir = str(tmp_path / "distance") + "/"
        angle_matrix_dir = str(tmp_path / "angle") + "/"

    monkeypatch.setattr(uf, "AnimalInfo", lambda *args: Info())
    distance = uf.generate_distance_matrix("mouse", "gamanut", dim="3D")
    angles = uf.generate_angle_matrix("mouse", "gamanut", dim="3D")
    assert distance.index.tolist() == ["A", "B"]
    assert np.allclose(distance.values, [[0, 5], [5, 0]])
    assert angles["azim"].shape == (2, 2)
    assert angles["elev"].shape == (2, 2)


def test_matrix_alignment_uses_distance_regions():
    matrix = pd.DataFrame(np.arange(9).reshape(3, 3), index=["A", "B", "C"], columns=["A", "B", "C"])
    distance = pd.DataFrame(np.zeros((2, 2)), index=["C", "A"], columns=["C", "A"])
    aligned = uf.get_matrix_aligned_to_distance(matrix, distance)
    assert aligned.index.tolist() == ["C", "A"]
    assert aligned.columns.tolist() == ["C", "A"]
    assert np.array_equal(aligned.values, [[8, 6], [2, 0]])


def test_voxel_probability_matches_equal_distance_correction(tmp_path):
    catalog = get_voxel_catalog(tmp_path)
    probabilities = model.get_displacement_probabilities(catalog, 1.7)
    pair_probabilities = {}
    for displacement, probability, pair_count in zip(catalog["displacements"], probabilities, catalog["displacement_counts"]):
        sources, targets = model._get_valid_pair_indices(catalog, displacement)
        assert len(sources) == pair_count
        for source, target in zip(sources, targets):
            pair_probabilities[(int(source), int(target))] = probability / pair_count
    coordinates = np.asarray(catalog["coordinates"])
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
    unique_distances, counts = np.unique(distances[distances > 0], return_counts=True)
    normalizer = np.sum(np.exp(-unique_distances / 1.7))
    for source in range(len(coordinates)):
        for target in range(len(coordinates)):
            if source == target:
                continue
            distance = distances[source, target]
            count = counts[np.flatnonzero(np.isclose(unique_distances, distance))[0]]
            expected = np.exp(-distance / 1.7) / count / normalizer
            assert np.isclose(pair_probabilities[(source, target)], expected, rtol=0, atol=1e-12)
    assert np.isclose(sum(pair_probabilities.values()), 1.0, rtol=0, atol=1e-12)


def test_voxel_sampling_reproducible_across_process_modes(tmp_path):
    catalog = get_voxel_catalog(tmp_path)
    expected = model.get_connection_samples(catalog, 1.2, 2000, 4)
    results = cf.multi_process(
        process_num=2, func=load_voxel_samples,
        kwargs_list=[{"cache_dir": Path(tmp_path) / "voxel_catalog", "seed": 4}] * 2,
        func_name="cross-species voxel reproducibility",
    )
    for result in results:
        assert np.array_equal(result[0], expected[0])
        assert np.array_equal(result[1], expected[1])
        assert np.all(result[0] != result[1])


def test_human_bin_mass_and_sampling(tmp_path):
    catalog = get_human_catalog(tmp_path)
    probabilities = model.get_binned_distance_probabilities(catalog, 0.8)
    weights = model.get_binned_pair_weights(catalog, 0.8)
    assert np.allclose(weights * catalog["global_bin_counts"], probabilities, rtol=0, atol=1e-15)
    area_probabilities = model.get_binned_area_probability_matrix(catalog, 0.8)
    assert np.isclose(np.sum(area_probabilities), 1.0, rtol=0, atol=1e-12)
    counts = model.get_binned_area_connection_matrix(catalog, area_probabilities, 5000, 3)
    first = model.get_human_vertex_samples(catalog, counts, 0.8, 7)
    second = model.get_human_vertex_samples(catalog, counts, 0.8, 7)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    observed = np.zeros((2, 2), dtype=np.int64)
    area_ids = np.asarray(catalog["area_ids"])
    np.add.at(observed, (area_ids[first[0]], area_ids[first[1]]), 1)
    assert np.array_equal(observed, counts.values)


def test_human_area_sampling_reproducible_across_process_modes(tmp_path):
    catalog = get_human_catalog(tmp_path)
    probabilities = model.get_binned_area_probability_matrix(catalog, 0.8)
    expected = model.get_binned_area_connection_matrix(catalog, probabilities, 2000, 4).values
    kwargs = {"cache_dir": Path(tmp_path) / "human_catalog", "probabilities": probabilities, "seed": 4}
    results = cf.multi_process(process_num=2, func=load_human_samples, kwargs_list=[kwargs, kwargs], func_name="cross-species human reproducibility")
    for result in results:
        assert np.array_equal(result, expected)


def test_mouse_mapping_and_downsampling_rules():
    volume = np.zeros((4, 4, 8), dtype=np.int16)
    volume[:, :, :4] = 1
    volume[:3, :, 4:] = 2
    mapping = pd.DataFrame({"gamanut_area": ["one", "two"]})
    coordinates, area_ids, names, spacing = preprocess.get_coarse_voxels(volume, mapping)
    assert names == ["one", "two"]
    assert np.array_equal(area_ids, [0, 1])
    assert np.allclose(coordinates, [[0, 0, 0], [0, 0, 0.1]])
    assert np.allclose(spacing, 0.1)
    first = preprocess.get_hemisphere(np.zeros((8, 6, 10)), side="first")
    assert first.shape == (8, 6, 5)


def test_no_corner_cutting_graph():
    points = np.asarray([[0, 0, 0], [1, 1, 0]], dtype=np.int16)
    voxel_map = np.full((2, 2, 1), -1, dtype=np.int32)
    voxel_map[tuple(points.T)] = np.arange(len(points))
    assert preprocess.get_no_corner_cutting_graph(points, voxel_map, np.ones(3)).nnz == 0
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.int16)
    voxel_map[tuple(points.T)] = np.arange(len(points))
    graph = preprocess.get_no_corner_cutting_graph(points, voxel_map, np.ones(3))
    assert np.isclose(scipy.sparse.csgraph.dijkstra(graph, directed=False, indices=0)[3], np.sqrt(2))


def test_full_distance_catalog_counts_every_ordered_pair(monkeypatch, tmp_path):
    coordinates = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    area_ids = np.asarray([0, 0, 1, 1], dtype=np.int16)
    graph = scipy.sparse.csr_matrix(np.asarray([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=float))
    output_dir = Path(tmp_path) / "toy" / "full_pair_distances"
    metadata = preprocess.build_full_distance_catalogs(
        graph, coordinates, area_ids, ["A", "B"], output_dir, bin_width=0.5,
        worker_count=1, source_chunk_size=2, dijkstra_batch_size=1,
    )
    resumed = preprocess.build_full_distance_catalogs(
        graph, coordinates, area_ids, ["A", "B"], output_dir, bin_width=0.5,
        worker_count=1, source_chunk_size=2, dijkstra_batch_size=1,
    )
    assert resumed == metadata
    assert metadata["packed_upper_triangle_pair_count"] == 6
    distance_count = sum(
        len(np.load(path)) for path in output_dir.glob("chunks/*/geodesic_upper_float32.npy")
    )
    assert distance_count == 6
    euclidean = model.load_binned_area_catalog(output_dir / "euclidean_catalog")
    geodesic = model.load_binned_area_catalog(output_dir / "geodesic_catalog")
    assert np.array_equal(euclidean["area_pair_bin_counts"], geodesic["area_pair_bin_counts"])
    assert int(np.sum(geodesic["global_bin_counts"])) == 12
    assert geodesic["global_bin_counts"][2] == 6
    assert geodesic["global_bin_counts"][4] == 4
    assert geodesic["global_bin_counts"][6] == 2
    probabilities = model.get_binned_area_probability_matrix(geodesic, 1.0)
    assert np.isclose(np.sum(probabilities), 1.0, rtol=0, atol=1e-12)
    monkeypatch.setattr(analysis, "PREPROCESSING_DIR", Path(tmp_path))
    names, euclidean_mean, geodesic_mean = analysis.get_full_pair_area_distance_means("toy")
    assert names.tolist() == ["A", "B"]
    assert np.array_equal(euclidean_mean, geodesic_mean, equal_nan=True)


def test_human_shortcut_uses_saved_full_pair_distances(monkeypatch, tmp_path):
    vertices = np.asarray([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=float)
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    area_ids = np.asarray([0, 0, 0], dtype=np.int16)
    graph = preprocess.get_surface_graph(vertices, faces)
    full_distance_dir = Path(tmp_path) / "full_pair_distances"
    preprocess.build_full_distance_catalogs(
        graph, vertices, area_ids, ["A"], full_distance_dir, bin_width=0.5,
        worker_count=1, source_chunk_size=2, dijkstra_batch_size=1,
    )
    monkeypatch.setattr(
        preprocess, "get_human_surface_data",
        lambda: (vertices, faces, area_ids, ["A"]),
    )
    output_dir = Path(tmp_path) / "shortcut"
    metadata = preprocess.build_human_shortcut_cache(full_distance_dir, output_dir)
    resumed = preprocess.build_human_shortcut_cache(full_distance_dir, output_dir)
    assert resumed == metadata
    assert metadata["shortcut_euclidean_mm"] >= 2.0
    assert metadata["shortcut_geodesic_mm"] >= metadata["shortcut_euclidean_mm"]
    assert (output_dir / "shortcut_path_indices.npy").exists()


def test_binned_rmae_uses_shared_ratio_bins(monkeypatch):
    names = ["A", "B", "C", "D", "E"]
    row, column = np.triu_indices(len(names), k=1)
    pair_table = pd.DataFrame({
        "source_area": np.asarray(names)[row], "target_area": np.asarray(names)[column],
        "geodesic_to_euclidean_ratio": np.linspace(1, 3, len(row)),
    })
    matrix = pd.DataFrame(np.ones((5, 5)), index=names, columns=names)
    error_2d = pd.DataFrame(np.arange(25).reshape(5, 5), index=names, columns=names)
    returned = iter([error_2d, error_2d + 1])
    monkeypatch.setattr(analysis.uf, "data_model_relative_error", lambda *args, **kwargs: next(returned))
    existing = pd.DataFrame(False, index=names, columns=names)
    existing.values[row[:5], column[:5]] = True
    non_link = pd.DataFrame(False, index=names, columns=names)
    non_link.values[row[5:], column[5:]] = True
    monkeypatch.setattr(analysis.uf, "get_weight_category_mask", lambda *args, **kwargs: {"all": existing, "non": non_link})
    result = analysis.get_ratio_binned_rmae(pair_table, matrix, matrix, matrix, 4, "human", "lariviere")
    existing_result = result[result["link_category"] == "existing"]
    non_link_result = result[result["link_category"] == "non-link"]
    assert np.allclose(existing_result["ratio_mean"], non_link_result["ratio_mean"])
    occupied = result["pair_count"] > 0
    assert np.allclose(result.loc[occupied, "rmae_3d"] - result.loc[occupied, "rmae_2d"], 1)


def test_seed_rmae_summary_averages_errors_after_each_seed():
    rows = []
    for seed in range(5):
        rows.append({
            "animal": "human", "link_category": "existing", "bin": 1,
            "ratio_left": 1.0, "ratio_mean": 1.1, "ratio_right": 1.2, "pair_count": 10,
            "rmae_2d": float(seed), "rmae_3d": float(seed + 2), "seed": seed,
        })
    summary = analysis.get_seed_rmae_summary(pd.DataFrame(rows))
    assert summary.loc[0, "seed_count"] == 5
    assert np.isclose(summary.loc[0, "rmae_2d_mean"], 2)
    assert np.isclose(summary.loc[0, "rmae_3d_mean"], 4)
    assert np.isclose(summary.loc[0, "rmae_2d_std"], np.std(np.arange(5), ddof=1))


def test_geodesic_distance_rmae_uses_shared_pairs_and_bins(monkeypatch):
    names = ["A", "B", "C", "D"]
    row, column = np.triu_indices(len(names), k=1)
    pair_table = pd.DataFrame({
        "source_area": np.asarray(names)[row],
        "target_area": np.asarray(names)[column],
        "geodesic_distance_mm": np.asarray([5, 15, 25, 35, 45, 55], dtype=float),
    })
    matrix = pd.DataFrame(np.ones((4, 4)), index=names, columns=names)
    error_euclidean = pd.DataFrame(np.arange(16).reshape(4, 4), index=names, columns=names)
    returned = iter([error_euclidean, error_euclidean + 2])
    monkeypatch.setattr(analysis.uf, "data_model_relative_error", lambda *args, **kwargs: next(returned))
    existing = pd.DataFrame(False, index=names, columns=names)
    existing.values[row[:3], column[:3]] = True
    non_link = pd.DataFrame(False, index=names, columns=names)
    non_link.values[row[3:], column[3:]] = True
    monkeypatch.setattr(
        analysis.uf, "get_weight_category_mask",
        lambda *args, **kwargs: {"all": existing, "non": non_link},
    )
    result = analysis.get_geodesic_distance_binned_rmae(
        pair_table, matrix, matrix, matrix, np.asarray([0, 20, 40, 60])
    )
    existing_result = result[result["link_category"] == "existing"]
    non_link_result = result[result["link_category"] == "non-link"]
    assert np.allclose(
        existing_result["geodesic_distance_mean_mm"],
        non_link_result["geodesic_distance_mean_mm"],
    )
    occupied = result["pair_count"] > 0
    assert np.allclose(
        result.loc[occupied, "rmae_geodesic"] - result.loc[occupied, "rmae_euclidean"], 2
    )


def test_distance_matrix_similarity_uses_unique_area_pairs():
    names = ["A", "B", "C"]
    first = pd.DataFrame(
        [[0, 1, 2], [1, 0, 3], [2, 3, 0]], index=names, columns=names
    )
    second = first * 2
    third = pd.DataFrame(
        [[0, 3, 2], [3, 0, 1], [2, 1, 0]], index=names, columns=names
    )
    similarity = analysis.get_distance_matrix_similarity({
        "first": first, "second": second, "third": third,
    })
    assert np.isclose(similarity.loc["first", "second"], 1)
    assert np.isclose(similarity.loc["first", "third"], -1)
