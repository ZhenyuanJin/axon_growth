import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from target_conditioned_angle_edr import analysis
from target_conditioned_angle_edr import model
from target_conditioned_angle_edr import parameters
from target_conditioned_angle_edr import plotting
from target_conditioned_angle_edr import reporting
from utils_function import cf
from utils_function import model_runtime


def get_tasks(animal, key, indices, repeat_count, point_count, max_iteration):
    """Return unique deterministic paired-method tasks ordered by repeat and grid index."""
    tasks = [model.get_task(
        f"repeat_{repeat:02d}_grid_{index:03d}", animal, key, repeat, index,
        parameters.get_lambda(animal, key, index), parameters.BASE_SEED,
        point_count, parameters.ANGLE_KDE_BANDWIDTH_DEGREES, max_iteration,
    ) for repeat in range(repeat_count) for index in sorted(indices)]
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("Task IDs are not unique.")
    return tasks


def run_tasks(tasks, output_dir, worker_count):
    """Use the project parallel helper, reject failed workers, and collect results by task ID."""
    completed = [model.load_task_results(task, output_dir) for task in tasks]
    pending = [task for task, result in zip(tasks, completed) if result is None]
    if pending:
        results = cf.multi_process(
            process_num=min(worker_count, len(pending)), func=model.run_task,
            kwargs_list=[{"task": task, "output_dir": output_dir} for task in pending],
            func_name=f"{Path(output_dir).name}: {len(pending)} paired tasks",
        )
        if any(result is None for result in results):
            raise RuntimeError("A worker failed; completed task files are retained for resumption.")
        completed = [result for result in completed if result is not None] + results
    completed.sort(key=lambda item: item["task_id"])
    if [item["task_id"] for item in completed] != sorted(task["task_id"] for task in tasks):
        raise ValueError("Collected task identities do not match the task table.")
    return (pd.concat([item["metrics"] for item in completed], ignore_index=True),
            pd.concat([item["targets"] for item in completed], ignore_index=True))


def run_dataset(animal, key, output_root, repeat_count, point_count, worker_count, initial_indices, max_extensions):
    """Run one resumable repeated scan, extend correlation boundaries, and save all target curves."""
    start = time.monotonic()
    dataset = f"{animal}_{key}"
    output_dir = Path(output_root) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    context = model.get_context(animal, key)
    info = context["info"]
    tables = output_dir / "tables"
    tables.mkdir(exist_ok=True)
    model.get_geometry_audit(context).to_csv(tables / "geometry_audit.csv", index=False)
    context["SC_data"].to_csv(tables / "empirical_SC.csv")
    input_paths = [Path(info.SC_dir) / "SC_data.xlsx", Path(info.area_coords_dir) / "polygon_dict.joblib",
                   Path(info.area_coords_dir) / "whole_cortex_polygon.joblib"]
    input_hashes = {str(path.relative_to(parameters.PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in input_paths}
    model_runtime.save_json(input_hashes, output_dir / "input_sha256.json")
    model_runtime.save_json({"full_split_name": info.full_split_name, "source_name": info.source_name,
                             "target_name": info.target_name}, output_dir / "area_names.json")
    plotting.plot_all_angle_distributions(context["angles"], output_dir / "angle_distributions",
                                           parameters.ANGLE_KDE_BANDWIDTH_DEGREES)
    indices = sorted(initial_indices)
    for extension_round in range(max_extensions + 1):
        tasks = get_tasks(animal, key, indices, repeat_count, point_count, parameters.MAX_ITERATION)
        pd.DataFrame(tasks).to_csv(tables / "task_table.csv", index=False)
        print(f"{dataset}: {len(indices)} Lambda values, {len(tasks)} paired tasks, round {extension_round}", flush=True)
        metrics, targets = run_tasks(tasks, output_dir, worker_count)
        summaries = reporting.save_tables(metrics, targets, repeat_count, output_dir)
        extra = analysis.get_boundary_extensions(summaries[0], indices)
        if not extra or extension_round == max_extensions:
            break
        indices = sorted(set(indices + extra))
    summary = reporting.save_dataset_results(
        metrics, targets, summaries, info, point_count, output_dir,
        elapsed_seconds=time.monotonic() - start,
    )
    print(f"Completed {dataset}: gain={summary['correlation_gain_mean']:.6f}, seconds={summary['seconds']:.1f}", flush=True)
    return summary


def run_experiment(output_dir, datasets, repeat_count, point_count, worker_count, grid_indices, max_extensions):
    """Run the explicit dataset list and save a cross-dataset repeated-result index."""
    if repeat_count < 2:
        raise ValueError("At least two repeats are required for sample SD.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {"datasets": datasets, "repeat_count": repeat_count, "na_per_target": point_count,
              "base_seed": parameters.BASE_SEED, "initial_grid_indices": grid_indices,
              "max_extensions": max_extensions, "worker_count": worker_count,
              "bandwidth_degrees": parameters.ANGLE_KDE_BANDWIDTH_DEGREES,
              "max_iteration": parameters.MAX_ITERATION,
              "SC_conversion": "uf.get_connectivity_from_connection_samples",
              "summary": "Arithmetic mean and sample SD of independently evaluated repeats.",
              "in_sample": True}
    manifest_path = output_dir / "configuration.json"
    if manifest_path.exists():
        previous = model_runtime.load_json(manifest_path)
        comparable = {name: value for name, value in config.items() if name != "worker_count"}
        previous.pop("worker_count", None)
        if json.dumps(previous, sort_keys=True) != json.dumps(comparable, sort_keys=True):
            raise ValueError("Output directory belongs to a different configuration.")
    model_runtime.save_json(config, manifest_path)
    reporting.save_provenance(output_dir, config)
    summaries = []
    for animal, key in datasets:
        summaries.append(run_dataset(animal, key, output_dir, repeat_count, point_count,
                                     worker_count, grid_indices, max_extensions))
        reporting.save_experiment_summary(summaries, output_dir)
    model_runtime.save_json({"complete": True, "datasets": len(summaries)}, output_dir / "complete.json")
    return summaries


def main():
    run_experiment(parameters.OUTPUT_DIR, parameters.DATASETS, parameters.REPEAT_COUNT,
                   parameters.NA_PER_TARGET, parameters.WORKER_COUNT,
                   parameters.INITIAL_GRID_INDICES, parameters.MAX_GRID_EXTENSIONS)


if __name__ == "__main__":
    main()
