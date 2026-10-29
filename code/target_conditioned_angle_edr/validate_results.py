import hashlib
import warnings
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from target_conditioned_angle_edr import model
from target_conditioned_angle_edr import parameters
from utils_function import model_runtime
from utils_function import utils_function as uf


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _load_array_headers(path):
    headers = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as file:
                version = np.lib.format.read_magic(file)
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = reader(file)
                headers[name.removesuffix(".npy")] = (shape, dtype)
    return headers


def get_dataset_validation(animal, key, output_root):
    """Validate repeat tables, sample headers and plots; reconstruct one saved pair per dataset."""
    dataset = f"{animal}_{key}"
    output_dir = output_root / dataset
    metadata = model_runtime.load_json(output_dir / "metadata.json")
    context = model.get_context(animal, key)
    info = context["info"]
    tables = output_dir / "tables"
    tasks = pd.read_csv(tables / "task_table.csv")
    metrics = pd.read_csv(tables / "lambda_scan_repeats.csv")
    scan = pd.read_csv(tables / "lambda_scan_summary.csv")
    targets = pd.read_csv(tables / "target_metrics_repeats.csv", dtype={"target": str})
    target_summary = pd.read_csv(tables / "target_metrics_summary.csv", dtype={"target": str})
    optimal = pd.read_csv(tables / "optimal_models.csv")
    repeat_count = metadata["repeat_count"]
    _check(repeat_count == 5, "Formal results must contain five repeats.")
    _check(tasks.task_id.is_unique, "Task IDs overlap.")
    _check(len(metrics) == len(tasks) * 2, "Not all task methods are present.")
    _check(len(targets) == len(metrics) * len(info.target_name), "Target metrics are incomplete.")
    counts = metrics.groupby(["direction_method", "grid_index"]).repeat_id.nunique()
    _check((counts == repeat_count).all(), "A parameter combination has missing repeats.")
    for name, digest in model_runtime.load_json(output_dir / "input_sha256.json").items():
        _check(hashlib.sha256((parameters.PROJECT_ROOT / name).read_bytes()).hexdigest() == digest,
               f"Input changed: {name}")
    checked_files = 0
    for row in metrics.itertuples():
        directory = output_dir / "runs" / row.task_id / row.direction_method
        headers = _load_array_headers(directory / "connection_samples.npz")
        for field in ["source_points", "target_points"]:
            _check(headers[field] == ((row.accepted_count, 2), np.dtype("float64")), f"Invalid coordinate shape/dtype: {directory}")
        for field in ["source_split_ids", "target_split_ids", "sample_ids"]:
            _check(headers[field] == ((row.accepted_count,), np.dtype("int32")), f"Invalid integer-array shape/dtype: {directory}")
        group = targets.loc[(targets.task_id == row.task_id) & (targets.direction_method == row.direction_method)]
        _check(int(group.accepted_count.sum()) == row.accepted_count, "Target and global accepted counts differ.")
        _check(np.all(group.accepted_count + group.unresolved_count == metadata["na_per_target"]), "Rejection bookkeeping differs.")
        checked_files += 1
    for row in scan.itertuples():
        group = metrics.loc[(metrics.direction_method == row.direction_method) & (metrics.grid_index == row.grid_index)]
        for metric in parameters.METRIC_OBJECTIVES:
            np.testing.assert_allclose(getattr(row, f"{metric}_mean"), group[metric].mean(), rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(getattr(row, f"{metric}_std"), group[metric].std(ddof=1), rtol=1e-10, atol=1e-14)
    target_mean = targets.groupby(["direction_method", "grid_index", "target"]).correlation.mean()
    observed = target_summary.set_index(["direction_method", "grid_index", "target"]).correlation_mean
    np.testing.assert_allclose(observed.sort_index(), target_mean.sort_index(), rtol=1e-12, atol=1e-14)
    for row in optimal.itertuples():
        subset = scan.loc[scan.direction_method == row.direction_method]
        _check(row.correlation_mean == subset.correlation_mean.max(), "Optimum is not selected from mean correlation.")
    pngs = sorted((output_dir / "angle_distributions").glob("*.png"))
    pdfs = sorted((output_dir / "angle_distributions").glob("*.pdf"))
    _check(len(pngs) == len(pdfs) == len(info.target_name), "Target angle plots are incomplete.")
    for path in pngs + list((output_dir / "figures").glob("*.png")):
        with Image.open(path) as image:
            image.verify()
    _check(not list(output_dir.glob("*.png")), "A figure was saved outside a subdirectory.")
    chosen_task = tasks.loc[(tasks.repeat_id == 0) & (tasks.grid_index == 16)].iloc[0]
    split_names = np.asarray(info.full_split_name)
    for method in parameters.DIRECTION_METHODS:
        directory = output_dir / "runs" / chosen_task.task_id / method
        with np.load(directory / "connection_samples.npz", allow_pickle=False) as samples:
            ids_s, ids_t = samples["source_split_ids"], samples["target_split_ids"]
            _check(np.all(context["split_area_ids"][ids_s] != context["split_area_ids"][ids_t]), "Self-connections were saved.")
            _check(np.isfinite(samples["source_points"]).all() and np.isfinite(samples["target_points"]).all(), "Nonfinite coordinates.")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="divide by zero encountered in log", category=RuntimeWarning)
                rebuilt = uf.get_connectivity_from_connection_samples(
                    animal, key, outcomes_dir=str(output_root / "validation" / dataset / method),
                    indices_ss=split_names[ids_s], indices_tt=split_names[ids_t],
                )
        for matrix_key, frame in rebuilt.items():
            saved = pd.read_csv(directory / "connectivity" / f"{matrix_key}.csv", index_col=0)
            np.testing.assert_allclose(saved.values, frame.values, rtol=1e-12, atol=1e-14, equal_nan=True)
        scores, _ = model.get_performance(context["SC_data"], rebuilt[metadata["SC_key"]], info)
        saved_row = metrics.loc[(metrics.task_id == chosen_task.task_id) & (metrics.direction_method == method)].iloc[0]
        for metric, value in scores.items():
            np.testing.assert_allclose(saved_row[metric], value, rtol=1e-12, atol=1e-14)
    boundary_methods = []
    optimal_acceptance = []
    for row in optimal.itertuples():
        if row.grid_index in [tasks.grid_index.min(), tasks.grid_index.max()]:
            boundary_methods.append(row.direction_method)
        optimal_acceptance.extend(targets.loc[
            (targets.direction_method == row.direction_method) & (targets.grid_index == row.grid_index),
            "acceptance_rate",
        ])
    return {
        "dataset": dataset, "paired_task_count": len(tasks), "sample_files_checked": checked_files,
        "angle_plot_count": len(pngs), "optimal_methods_on_grid_edge": boundary_methods,
        "minimum_acceptance_rate": float(targets.acceptance_rate.min()),
        "optimal_minimum_acceptance_rate": float(np.min(optimal_acceptance)),
        "bytes": sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file()),
    }


def main():
    _check((parameters.OUTPUT_DIR / "complete.json").exists(), "The formal experiment is incomplete.")
    results = [get_dataset_validation(animal, key, parameters.OUTPUT_DIR) for animal, key in parameters.DATASETS]
    model_runtime.save_json({"passed": True, "datasets": results}, parameters.OUTPUT_DIR / "validation.json")
    rows = []
    for result in results:
        metadata = model_runtime.load_json(parameters.OUTPUT_DIR / result["dataset"] / "metadata.json")
        selected = {row["direction_method"]: row for row in metadata["optimal_models"]}
        random, kde = selected["random"], selected["SC_angle_kde"]
        rows.append({"dataset": result["dataset"], "random_Lambda": random["Lambda"],
                     "random_correlation_mean": random["correlation_mean"],
                     "random_correlation_std": random["correlation_std"],
                     "SC_angle_kde_Lambda": kde["Lambda"],
                     "SC_angle_kde_correlation_mean": kde["correlation_mean"],
                     "SC_angle_kde_correlation_std": kde["correlation_std"],
                     "gain_mean": metadata["correlation_gain_mean"], "gain_std": metadata["correlation_gain_std"],
                     "targets_improved": metadata["targets_improved"], "target_count": metadata["target_count"]})
    summary = pd.DataFrame(rows)
    summary.to_csv(parameters.OUTPUT_DIR / "results_summary.csv", index=False)
    report = ["# Repeated target-conditioned angle experiment", "",
              "All five datasets use five independent repeats and the project connection-samples-to-SC pipeline.",
              "Each target requests 20,000 samples with up to ten attempts per sample. Correlations align area names and exclude self-connections by matching source and target labels.",
              "Metrics are evaluated separately for each repeat before arithmetic averaging. Uncertainty below is sample SD, not SEM.", "",
              "| Dataset | Random correlation | SC angle KDE correlation | Paired gain | Targets improved |",
              "|---|---:|---:|---:|---:|"]
    for row in rows:
        report.append(f"| {row['dataset']} | {row['random_correlation_mean']:.4f} +/- {row['random_correlation_std']:.4f} "
                      f"| {row['SC_angle_kde_correlation_mean']:.4f} +/- {row['SC_angle_kde_correlation_std']:.4f} "
                      f"| {row['gain_mean']:.4f} +/- {row['gain_std']:.4f} | {row['targets_improved']}/{row['target_count']} |")
    report.extend(["", "Optima maximize mean overall correlation over the tested grid, separately for each method.",
                   "See `results_summary.csv` for selected Lambdas; all other metric optima are in each dataset's `tables/metric_optima.csv`.", "",
                   "All 335 measured target areas have KDE-only angle plots in their dataset's `angle_distributions/` folder.",
                   "Correlation comparisons and other metric curves are in `figures/` with mean +/- sample SD.",
                   "Every task retains source/target coordinates, split IDs, rejection counts, and all SC representations produced by the project conversion.", "",
                   "## Validation and interpretation", "",
                   "All sample-file headers and repeat groups were checked. Saved samples from one paired task per dataset were reconstructed; every SC representation and reported metric matched.",
                   "The empirical matrix both supplies the angular distribution and evaluates reconstruction. Repeated seeds quantify sampling variability.", "",
                   "| Dataset | Minimum acceptance at selected optima |",
                   "|---|---:|"])
    for result in results:
        report.append(f"| {result['dataset']} | {result['optimal_minimum_acceptance_rate']:.3f} |")
    report.extend(["", "Acceptance below one means fewer valid samples than requested; low-acceptance records are retained in `tables/low_acceptance.csv`.",
                   "Polygon validity and overlapping areas are recorded in each `tables/geometry_audit.csv`.",
                   ""])
    (parameters.OUTPUT_DIR / "RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
