import functools
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

from target_conditioned_angle_edr import parameters
from target_conditioned_angle_edr import sampling
from utils_function import cf
from utils_function import model_runtime
from utils_function import utils_function as uf


@functools.lru_cache(maxsize=5)
def get_context(animal, dataset_key):
    """Cache read-only empirical and geometry inputs in each process."""
    return sampling.get_context(uf.AnimalInfo(animal, dataset_key))


def get_geometry_audit(context):
    """Return deterministic records of invalid polygons and positive-area overlaps."""
    polygons = context["polygons"]
    names = list(polygons)
    rows = []
    for index, name in enumerate(names):
        if not polygons[name].is_valid:
            rows.append({"issue": "invalid_polygon", "first": name, "second": "",
                         "overlap_area": 0.0, "detail": shapely.is_valid_reason(polygons[name])})
        for other in names[index + 1:]:
            area = polygons[name].intersection(polygons[other]).area
            if area > 1e-9:
                rows.append({"issue": "overlap", "first": name, "second": other,
                             "overlap_area": area, "detail": "First match in polygon insertion order."})
    return pd.DataFrame(rows, columns=["issue", "first", "second", "overlap_area", "detail"])


def get_performance(SC_data, SC_model, animal_info):
    """Evaluate SC with project functions; return global and target metrics excluding self-connections."""
    animal, key = animal_info.animal, animal_info.dataset_key
    selected = SC_model.loc[animal_info.source_name, animal_info.target_name]
    off_diagonal = np.asarray(selected.index)[:, None] != np.asarray(selected.columns)[None, :]
    if not np.all(np.isfinite(selected.to_numpy()[off_diagonal])):
        raise ValueError("Generated SC contains nonfinite measured off-diagonal entries.")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log", category=RuntimeWarning)
        logged_data = uf.log_SC(SC_data, animal, key, apply_c=True, apply_w=True)
        logged_model = uf.log_SC(SC_model, animal, key, apply_c=True, apply_w=True)
        relative_error = uf.data_model_relative_error(SC_data, SC_model, animal, key, diagonal=False)
    target_correlations = uf.area_level_correlation(SC_data, SC_model, animal, key, diagonal=False)
    target_logged_correlations = uf.area_level_correlation(logged_data, logged_model, animal, key, diagonal=False)
    target_rows = []
    for target in animal_info.target_name:
        target_rows.append({
            "target": target, "correlation": target_correlations[target],
            "log_correlation": target_logged_correlations[target],
            "rmae_both": float(relative_error[target].mean()),
        })
    values = np.asarray(list(target_correlations.values()))
    if not np.all(np.isfinite(values)):
        raise ValueError("A target correlation is undefined.")
    metrics = {
        "correlation": uf.data_model_correlation(SC_data, SC_model, animal, key, diagonal=False),
        "log_correlation": uf.data_model_correlation(logged_data, logged_model, animal, key, diagonal=False),
        "rmae_both": float(np.nanmean(relative_error.to_numpy())),
        "median_target_correlation": float(np.median(values)),
        "mean_target_correlation": float(values.mean()),
    }
    if not np.all(np.isfinite(list(metrics.values()))):
        raise ValueError("A global metric is undefined.")
    return metrics, pd.DataFrame(target_rows)


def get_task(task_id, animal, dataset_key, repeat_id, grid_index, Lambda, base_seed, na_per_target, bandwidth_degrees, max_iteration):
    """Return a JSON-compatible task with a paired repeat seed independent of worker order."""
    return {
        "task_id": task_id, "animal": animal, "dataset_key": dataset_key,
        "repeat_id": repeat_id, "grid_index": grid_index, "Lambda": Lambda,
        "seed": sampling.get_stage_seed(base_seed, animal, dataset_key, repeat_id),
        "na_per_target": na_per_target, "bandwidth_degrees": bandwidth_degrees,
        "max_iteration": max_iteration, "protocol_version": 1,
    }


def load_task_results(task, output_dir):
    """Load completed task tables only when the saved configuration matches exactly."""
    task_dir = Path(output_dir) / "runs" / task["task_id"]
    manifest_path = task_dir / "complete.json"
    if not manifest_path.exists():
        return None
    manifest = model_runtime.load_json(manifest_path)
    if manifest["task"] != task:
        raise ValueError(f"Task configuration changed: {task_dir}")
    return {
        "task_id": task["task_id"],
        "metrics": pd.read_csv(task_dir / "metrics.csv"),
        "targets": pd.read_csv(task_dir / "target_metrics.csv", dtype={"target": str}),
    }


def _get_method_evaluation(task, method, matrix, acceptance, seconds, context):
    metrics, targets = get_performance(context["SC_data"], matrix, context["info"])
    targets = targets.merge(acceptance, on="target", validate="one_to_one")
    identity = {name: task[name] for name in ["task_id", "repeat_id", "grid_index", "Lambda", "seed"]}
    identity["direction_method"] = method
    row = {**identity, **metrics, "mean_acceptance_rate": acceptance.acceptance_rate.mean(),
           "minimum_acceptance_rate": acceptance.acceptance_rate.min(),
           "accepted_count": int(acceptance.accepted_count.sum()),
           "unresolved_count": int(acceptance.unresolved_count.sum()), "seconds": seconds}
    for name, value in identity.items():
        targets[name] = value
    return row, targets


def evaluate_task(task, output_dir):
    """Evaluate saved SC matrices and acceptance counts; return deterministic global and target tables."""
    context = get_context(task["animal"], task["dataset_key"])
    info = context["info"]
    selected_key = uf.SC_type_to_SC_dict_key(info.SC_type)
    task_dir = Path(output_dir) / "runs" / task["task_id"]
    previous = load_task_results(task, output_dir)
    if previous is None:
        raise FileNotFoundError(f"Missing completed task: {task_dir}")
    acceptance_columns = ["target", "requested_count", "accepted_count", "unresolved_count",
                          "acceptance_rate", "proposal_count", "self_rejections", "outside_rejections"]
    rows, target_frames = [], []
    for method in parameters.DIRECTION_METHODS:
        matrix = pd.read_csv(task_dir / method / "connectivity" / f"{selected_key}.csv", index_col=0)
        matrix.index, matrix.columns = matrix.index.astype(str), matrix.columns.astype(str)
        acceptance = previous["targets"].loc[
            previous["targets"].direction_method == method, acceptance_columns,
        ].copy()
        seconds = float(previous["metrics"].loc[previous["metrics"].direction_method == method, "seconds"].iloc[0])
        row, targets = _get_method_evaluation(task, method, matrix, acceptance, seconds, context)
        rows.append(row)
        target_frames.append(targets)
    metrics, targets = pd.DataFrame(rows), pd.concat(target_frames, ignore_index=True)
    metrics.to_csv(task_dir / "metrics.csv", index=False)
    targets.to_csv(task_dir / "target_metrics.csv", index=False)
    return {"task_id": task["task_id"], "metrics": metrics, "targets": targets}


def run_task(task, output_dir):
    """Generate both paired methods, persist samples, and call the project samples-to-SC pipeline."""
    cached = load_task_results(task, output_dir)
    if cached is not None:
        return cached
    start = time.monotonic()
    context = get_context(task["animal"], task["dataset_key"])
    info = context["info"]
    task_dir = Path(output_dir) / "runs" / task["task_id"]
    task_dir.mkdir(parents=True, exist_ok=True)
    target_samples = {
        target: sampling.get_target_points(
            target, task["na_per_target"], info, context["polygons"],
            sampling.get_stage_seed(task["seed"], target, "target_points"),
        ) for target in info.target_name
    }
    rows, target_frames = [], []
    split_names = np.asarray(info.full_split_name)
    for method in parameters.DIRECTION_METHODS:
        method_start = time.monotonic()
        samples, diagnostics = [], []
        for target, (points, split_ids) in target_samples.items():
            result, diagnostic = sampling.get_target_samples(
                context, target, points, split_ids, task["Lambda"], method,
                task["bandwidth_degrees"], task["max_iteration"], task["seed"],
            )
            samples.append(result)
            diagnostics.append(diagnostic)
        combined = {name: np.concatenate([item[name] for item in samples]) for name in samples[0]}
        method_dir = task_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(method_dir / "connection_samples.npz", **combined)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in log", category=RuntimeWarning)
            connectivity = uf.get_connectivity_from_connection_samples(
                info.animal, info.dataset_key, outcomes_dir=str(method_dir),
                indices_ss=split_names[combined["source_split_ids"]],
                indices_tt=split_names[combined["target_split_ids"]],
            )
        selected_key = uf.SC_type_to_SC_dict_key(info.SC_type)
        acceptance = pd.DataFrame(diagnostics)
        row, targets = _get_method_evaluation(task, method, connectivity[selected_key], acceptance,
                                               time.monotonic() - method_start, context)
        rows.append(row)
        target_frames.append(targets)
    metrics = pd.DataFrame(rows)
    targets = pd.concat(target_frames, ignore_index=True)
    metrics.to_csv(task_dir / "metrics.csv", index=False)
    targets.to_csv(task_dir / "target_metrics.csv", index=False)
    model_runtime.save_json({"task": task, "seconds": time.monotonic() - start,
                             "SC_key": selected_key}, task_dir / "complete.json")
    return {"task_id": task["task_id"], "metrics": metrics, "targets": targets}
