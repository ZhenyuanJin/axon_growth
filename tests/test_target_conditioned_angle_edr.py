from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import shapely
import shapely.geometry

from target_conditioned_angle_edr import analysis
from target_conditioned_angle_edr import evaluate_results
from target_conditioned_angle_edr import model
from target_conditioned_angle_edr import parameters
from target_conditioned_angle_edr import run_experiment
from target_conditioned_angle_edr import sampling
from utils_function import cf
from utils_function import utils_function as uf


def get_mouse_task():
    return model.get_task("repeat_00_grid_016", "mouse", "gamanut", 0, 16, 0.85,
                          20261029, 150, 15.0, 10)


def test_component_weight_conservation_and_direction():
    info = SimpleNamespace(source_name=["S", "U", "T"])
    matrix = pd.DataFrame({"T": [0.6, 0.4, 0.0]}, index=info.source_name)
    components = {
        "S": (np.array([[2., 0.], [0., 2.]]), np.array([0.7, 0.3])),
        "U": (np.array([[-2., 0.]]), np.ones(1)),
        "T": (np.array([[0., 0.]]), np.ones(1)),
    }
    angles, weights = sampling.get_angle_centers_and_weights("T", info, matrix, components)
    np.testing.assert_allclose(angles, [0, np.pi / 2, np.pi])
    np.testing.assert_allclose(weights, [0.42, 0.18, 0.4])


def test_split_component_weights_and_unsplit_area_invariance():
    names = ["S", "U", "T"]
    info = SimpleNamespace(
        full_name=names, source_name=names,
        name_to_split_name_dict={"S": ["S--1", "S--2"], "U": ["U"], "T": ["T--1", "T--2"]},
    )
    polygons = {
        "S--1": shapely.geometry.box(0, 0, 7, 1),
        "S--2": shapely.geometry.box(10, 0, 13, 1),
        "U": shapely.geometry.box(20, 0, 120, 10),
        "T--1": shapely.geometry.box(0, 20, 2, 21),
        "T--2": shapely.geometry.box(10, 20, 18, 21),
    }
    matrix = pd.DataFrame({"T": [0.6, 0.4, 0]}, index=names)
    components = sampling.get_area_components(info, polygons)
    _, weights = sampling.get_angle_centers_and_weights("T", info, matrix, components)
    np.testing.assert_allclose(weights, [0.084, 0.336, 0.036, 0.144, 0.08, 0.32])
    np.testing.assert_allclose([weights[:4].sum(), weights[4:].sum()], [0.6, 0.4])
    np.testing.assert_array_equal(components["U"][1], [1.0])
    polygons["U"] = shapely.geometry.box(20, 0, 1020, 100)
    components = sampling.get_area_components(info, polygons)
    _, rescaled = sampling.get_angle_centers_and_weights("T", info, matrix, components)
    np.testing.assert_array_equal(rescaled, weights)
    info.name_to_split_name_dict["T"] = ["T"]
    polygons["T"] = shapely.geometry.box(0, 20, 50, 25)
    components = sampling.get_area_components(info, polygons)
    _, weights = sampling.get_angle_centers_and_weights("T", info, matrix, components)
    np.testing.assert_allclose(weights, [0.42, 0.18, 0.4])


@pytest.mark.parametrize("animal,key", parameters.DATASETS)
def test_fast_polygon_lookup_matches_original_including_overlaps(animal, key):
    context = model.get_context(animal, key)
    info = context["info"]
    polygons = context["polygons"]
    points = [np.array([polygon.centroid.x, polygon.centroid.y]) for polygon in polygons.values()]
    audit = model.get_geometry_audit(context)
    for row in audit.loc[audit.issue == "overlap"].itertuples():
        point = polygons[row.first].intersection(polygons[row.second]).representative_point()
        points.append(np.array([point.x, point.y]))
    points.append(np.array([1e6, 1e6]))
    points = np.asarray(points)
    expected = uf.point_in_which_area(points, polygons, context["whole_cortex"])
    ids = sampling.get_split_ids(points, context)
    actual = [info.full_split_name[index] if index >= 0 else "out_of_areas" for index in ids]
    assert actual == expected


def test_samples_use_original_conversion_and_reload_exactly(tmp_path):
    task = get_mouse_task()
    result = model.run_task(task, tmp_path)
    context = model.get_context("mouse", "gamanut")
    names = np.asarray(context["info"].full_split_name)
    for method in parameters.DIRECTION_METHODS:
        directory = tmp_path / "runs" / task["task_id"] / method
        with np.load(directory / "connection_samples.npz", allow_pickle=False) as data:
            assert data["source_points"].dtype == np.float64
            assert data["source_points"].shape == data["target_points"].shape
            ids_s, ids_t = data["source_split_ids"], data["target_split_ids"]
            assert np.all(context["split_area_ids"][ids_s] != context["split_area_ids"][ids_t])
            np.testing.assert_array_equal(sampling.get_split_ids(data["source_points"], context), ids_s)
            expected = uf.get_connectivity_from_connection_samples(
                "mouse", "gamanut", outcomes_dir=str(tmp_path / f"reconstructed_{method}"),
                indices_ss=names[ids_s], indices_tt=names[ids_t],
            )
        for matrix_key, matrix in expected.items():
            saved = pd.read_csv(directory / "connectivity" / f"{matrix_key}.csv", index_col=0)
            np.testing.assert_allclose(saved.values, matrix.values, equal_nan=True)
    reloaded = model.load_task_results(task, tmp_path)
    pd.testing.assert_frame_equal(result["metrics"], reloaded["metrics"], check_exact=False)
    changed = {**task, "seed": task["seed"] + 1}
    with pytest.raises(ValueError, match="configuration changed"):
        model.load_task_results(changed, tmp_path)


def test_single_and_parallel_samples_and_metrics_are_identical(tmp_path):
    task = get_mouse_task()
    single = model.run_task(task, tmp_path / "single")
    results = cf.multi_process(process_num=2, func=model.run_task,
                               kwargs_list=[{"task": task, "output_dir": tmp_path / name}
                                            for name in ["parallel_a", "parallel_b"]],
                               func_name="target-conditioned determinism")
    for name, parallel in zip(["parallel_a", "parallel_b"], results):
        assert parallel is not None
        pd.testing.assert_frame_equal(single["metrics"].drop(columns="seconds"),
                                      parallel["metrics"].drop(columns="seconds"), check_exact=True)
        pd.testing.assert_frame_equal(single["targets"], parallel["targets"], check_exact=True)
        for method in parameters.DIRECTION_METHODS:
            relative = f"runs/{task['task_id']}/{method}/connection_samples.npz"
            with np.load(tmp_path / "single" / relative) as first, np.load(tmp_path / name / relative) as second:
                for key in first.files:
                    np.testing.assert_array_equal(first[key], second[key])


def test_saved_task_evaluation_matches_generated_metrics(tmp_path):
    task = get_mouse_task()
    generated = model.run_task(task, tmp_path)
    archives = list(tmp_path.glob("runs/*/*/connection_samples.npz"))
    modified = [path.stat().st_mtime_ns for path in archives]
    evaluated = model.evaluate_task(task, tmp_path)
    pd.testing.assert_frame_equal(generated["metrics"], evaluated["metrics"], check_exact=False)
    pd.testing.assert_frame_equal(generated["targets"], evaluated["targets"], check_exact=False)
    assert modified == [path.stat().st_mtime_ns for path in archives]


def test_generation_and_reevaluation_produce_consistent_reports(tmp_path):
    generated = run_experiment.run_experiment(tmp_path, [("mouse", "gamanut")], 2, 150, 2, [16], 0)[0]
    output_dir = tmp_path / "mouse_gamanut"
    tables = {path.name: pd.read_csv(path) for path in (output_dir / "tables").glob("*.csv")}
    matrices = {path.name: pd.read_csv(path, index_col=0) for path in (output_dir / "optimal_matrices").glob("*.csv")}
    evaluated = evaluate_results.evaluate_saved_dataset(output_dir, worker_count=2)
    for name, expected in generated.items():
        if name in ["optimal_models", "metric_optima"]:
            pd.testing.assert_frame_equal(pd.DataFrame(evaluated[name]), pd.DataFrame(expected),
                                          check_exact=False, rtol=1e-12, atol=1e-14)
        elif isinstance(expected, float):
            assert evaluated[name] == pytest.approx(expected, rel=1e-12, abs=1e-14)
        else:
            assert evaluated[name] == expected
    for name, expected in tables.items():
        pd.testing.assert_frame_equal(pd.read_csv(output_dir / "tables" / name), expected)
    for name, expected in matrices.items():
        pd.testing.assert_frame_equal(pd.read_csv(output_dir / "optimal_matrices" / name, index_col=0), expected)
    for method in parameters.DIRECTION_METHODS:
        repeats = [matrices[f"{method}_repeat_{repeat:02d}_FLNe.csv"].values for repeat in range(2)]
        np.testing.assert_allclose(matrices[f"{method}_mean_FLNe.csv"].values, np.mean(repeats, axis=0))
    for name in ["correlation_improvement", "metric_scans"]:
        for extension in ["png", "pdf"]:
            assert (output_dir / "figures" / f"{name}.{extension}").stat().st_size > 0
    assert (tmp_path / "complete.json").exists()
    assert (tmp_path / "provenance" / "manifest.json").exists()
    assert len(list((output_dir / "angle_distributions").glob("*.png"))) == 19


def test_summary_averages_metrics_and_rejects_missing_or_duplicate_repeats():
    frame = pd.DataFrame({"method": ["A"] * 5, "repeat_id": range(5), "correlation": [0., 0., 0., 0., 1.]})
    result = analysis.get_repeated_summary(frame, ["method"], ["correlation"], 5)
    assert result.correlation_mean.iloc[0] == 0.2
    assert result.correlation_std.iloc[0] == pytest.approx(np.std(frame.correlation, ddof=1))
    with pytest.raises(ValueError, match="Incomplete"):
        analysis.get_repeated_summary(frame.iloc[:-1], ["method"], ["correlation"], 5)
    with pytest.raises(ValueError, match="Duplicate"):
        analysis.get_repeated_summary(pd.concat([frame, frame.iloc[:1]]), ["method"], ["correlation"], 5)


def test_human_pipeline_uses_raw_counts_before_symmetrizing(tmp_path):
    info = uf.AnimalInfo("human", "lariviere")
    names = info.full_split_name
    source = [names[0]] * 9 + [names[1]]
    target = [names[1]] * 9 + [names[0]]
    connectivity = uf.get_connectivity_from_connection_samples(
        "human", "lariviere", outcomes_dir=str(tmp_path), indices_ss=source, indices_tt=target,
    )
    raw = connectivity["CCMt"]
    assert raw.values.sum() == 10
    symmetric = connectivity[uf.SC_type_to_SC_dict_key(info.SC_type)]
    np.testing.assert_allclose(symmetric.values, symmetric.values.T, equal_nan=True)
    assert np.nansum(symmetric.values) == pytest.approx(len(info.target_name))
