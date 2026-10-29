import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from target_conditioned_angle_edr import analysis
from target_conditioned_angle_edr import parameters
from target_conditioned_angle_edr import plotting
from utils_function import model_runtime
from utils_function import utils_function as uf


def save_tables(metrics, targets, repeat_count, output_dir):
    """Save raw repeated metrics, strict summaries, optimum selections, and paired differences."""
    tables = Path(output_dir) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics = metrics.sort_values(["task_id", "direction_method"]).reset_index(drop=True)
    targets = targets.sort_values(["task_id", "direction_method", "target"]).reset_index(drop=True)
    scan, target_summary, optimal, optima = analysis.get_summaries(metrics, targets, repeat_count)
    paired, target_paired, comparison = analysis.get_paired_comparison(metrics, targets, optimal, repeat_count)
    frames = {
        "lambda_scan_repeats": metrics, "target_metrics_repeats": targets,
        "lambda_scan_summary": scan, "target_metrics_summary": target_summary,
        "optimal_models": optimal, "metric_optima": optima,
        "optimal_paired_repeats": paired, "optimal_target_paired_repeats": target_paired,
        "optimal_target_comparison": comparison,
    }
    for name, frame in frames.items():
        frame.to_csv(tables / f"{name}.csv", index=False)
    return scan, optimal, comparison, paired, optima


def save_optimal_matrices(optimal, repeat_count, output_dir, matrix_key):
    """Copy each optimum's five original-pipeline SC matrices; save their mean separately."""
    destination = Path(output_dir) / "optimal_matrices"
    destination.mkdir(exist_ok=True)
    for _, row in optimal.iterrows():
        matrices = []
        for repeat in range(repeat_count):
            task_id = f"repeat_{repeat:02d}_grid_{int(row.grid_index):03d}"
            source = Path(output_dir) / "runs" / task_id / row.direction_method / "connectivity" / f"{matrix_key}.csv"
            frame = pd.read_csv(source, index_col=0)
            matrices.append(frame)
            shutil.copy2(source, destination / f"{row.direction_method}_repeat_{repeat:02d}_{matrix_key}.csv")
        mean_matrix = matrices[0].copy()
        mean_matrix.iloc[:, :] = np.mean([frame.to_numpy() for frame in matrices], axis=0)
        mean_matrix.to_csv(destination / f"{row.direction_method}_mean_{matrix_key}.csv")


def save_provenance(output_dir, config):
    """Snapshot executable experiment code, dependencies, and the run configuration."""
    directory = Path(output_dir) / "provenance"
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    paths = list(Path(__file__).parent.glob("*.py")) + [
        parameters.PROJECT_ROOT / "code/utils_function/utils_function.py",
        parameters.PROJECT_ROOT / "code/utils_function/cf/_implementation.py",
    ]
    for path in paths:
        relative = path.relative_to(parameters.PROJECT_ROOT)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    model_runtime.save_json({"configuration": config, "code_sha256": hashes,
                             "versions": model_runtime.get_runtime_versions()}, directory / "manifest.json")


def save_dataset_results(metrics, targets, summaries, info, point_count, output_dir, elapsed_seconds=None):
    """Export optimum SCs, figures and metadata from independently evaluated repeat tables."""
    output_dir = Path(output_dir)
    scan, optimal, comparison, paired, optima = summaries
    repeat_count = int(metrics.repeat_id.nunique())
    matrix_key = uf.SC_type_to_SC_dict_key(info.SC_type)
    save_optimal_matrices(optimal, repeat_count, output_dir, matrix_key)
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    plotting.plot_correlation_improvement(scan, optimal, comparison, figures, info.SC_type)
    plotting.plot_metric_scans(scan, figures, info.SC_type)
    low_acceptance = targets.loc[targets.acceptance_rate < 0.95]
    low_acceptance.to_csv(output_dir / "tables" / "low_acceptance.csv", index=False)
    indices = sorted(metrics.grid_index.unique().tolist())
    metadata_path = output_dir / "metadata.json"
    metadata = model_runtime.load_json(metadata_path) if metadata_path.exists() else {}
    metadata.update({
        "dataset": output_dir.name, "SC_type": info.SC_type, "SC_key": matrix_key,
        "repeat_count": repeat_count, "na_per_target": point_count,
        "target_count": len(info.target_name), "grid_indices": indices,
        "Lambda_values": sorted(scan.Lambda.unique().tolist()),
        "correlation_grid_needs_further_extension": bool(analysis.get_boundary_extensions(scan, indices)),
        "targets_improved": int((comparison.correlation_difference_mean > 0).sum()),
        "correlation_gain_mean": float(paired.correlation_difference.mean()),
        "correlation_gain_std": float(paired.correlation_difference.std(ddof=1)),
        "low_acceptance_rows": len(low_acceptance), "optimal_models": optimal.to_dict(orient="records"),
        "metric_optima": optima.to_dict(orient="records"),
    })
    if elapsed_seconds is not None:
        metadata["seconds"] = elapsed_seconds
    model_runtime.save_json(metadata, metadata_path)
    return metadata


def save_experiment_summary(summaries, output_dir):
    """Save the completed dataset index and each method's mean-correlation optimum."""
    output_dir = Path(output_dir)
    model_runtime.save_json(summaries, output_dir / "dataset_summaries.json")
    rows = [{"dataset": summary["dataset"], "SC_type": summary["SC_type"], **row}
            for summary in summaries for row in summary["optimal_models"]]
    pd.DataFrame(rows).to_csv(output_dir / "cross_dataset_optimal_models.csv", index=False)
