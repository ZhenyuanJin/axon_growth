from pathlib import Path

import numpy as np
import pandas as pd
from utils_function import cf
from utils_function import utils_function as uf
from voxel_edr_3d import model
from voxel_edr_3d import plotting

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "voxel_edr_3d"
CACHE_DIR = OUTPUT_DIR / "preprocessing"
WORKER_COUNT = 10
COARSE_CONNECTION_COUNT = 1_000_000
FINE_CONNECTION_COUNT = 8_000_000
COARSE_SEEDS = range(3)
FINE_SEEDS = range(5)
COARSE_POINT_COUNT = 20
FINE_POINT_COUNT = 11
INITIAL_MIN_LAMBDA = 0.25
BOUNDARY_MIN_LAMBDA = 0.05


def get_task_table(lambda_values, seeds, connection_count, stage, animal="marmoset", dataset_key="majka"):
    """Return deterministic, uniquely identified parameter-search tasks."""
    tasks = []
    for lambda_index, Lambda in enumerate(lambda_values):
        for seed in seeds:
            tasks.append({
                "task_id": f"{stage}_lambda_{lambda_index:03d}_seed_{seed:02d}",
                "animal": animal,
                "dataset_key": dataset_key,
                "Lambda": float(Lambda),
                "seed": int(seed),
                "connection_count": int(connection_count),
            })
    task_ids = [task["task_id"] for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Task identifiers must be unique.")
    return tasks


def _run_task(task, cache_dir, output_dir, save_pairs=False):
    return model.run_connection_task(task, cache_dir, output_dir, save_pairs=save_pairs)


def run_tasks(tasks, stage_dir, cache_dir, worker_count, save_pairs=False):
    """Run task-isolated sampling jobs and save a task-id-sorted summary."""
    kwargs_list = [
        {"task": task, "cache_dir": cache_dir, "output_dir": stage_dir, "save_pairs": save_pairs}
        for task in tasks
    ]
    results = cf.multi_process(
        process_num=worker_count,
        func=_run_task,
        kwargs_list=kwargs_list,
        func_name=f"voxel EDR {stage_dir.name}",
    )
    results = sorted(results, key=lambda item: item["task_id"])
    summary = pd.DataFrame(results)
    summary.to_csv(stage_dir / "summary.csv", index=False)
    return summary


def get_best_lambda(summary):
    """Select lambda by highest mean seed correlation, breaking ties low."""
    grouped = summary.groupby("Lambda")["correlation"].agg(["mean", "std", "count"]).reset_index()
    grouped = grouped.sort_values(["mean", "Lambda"], ascending=[False, True]).reset_index(drop=True)
    return float(grouped.loc[0, "Lambda"]), grouped


def get_initial_coarse_lambdas(animal="marmoset", dataset_key="majka"):
    """Return the initial logarithmic scan and its upper boundary."""
    info = uf.AnimalInfo(animal, dataset_key)
    upper = float(np.max(info.distance_3D.values) / 2)
    return np.geomspace(INITIAL_MIN_LAMBDA, upper, COARSE_POINT_COUNT), upper


def get_boundary_extension(coarse_lambdas, best_lambda, initial_upper, max_distance):
    """Return one-sided extension values only when the optimum is a boundary."""
    if np.isclose(best_lambda, np.min(coarse_lambdas)):
        return np.geomspace(BOUNDARY_MIN_LAMBDA, np.min(coarse_lambdas), 11)[:-1]
    if np.isclose(best_lambda, np.max(coarse_lambdas)):
        return np.geomspace(initial_upper, max_distance, 11)[1:]
    return np.asarray([], dtype=float)


def get_fine_lambdas(coarse_lambdas, best_lambda):
    """Return linear refinement values spanning the neighbors of the optimum."""
    coarse_lambdas = np.unique(np.sort(coarse_lambdas))
    best_index = int(np.argmin(np.abs(coarse_lambdas - best_lambda)))
    lower_index = max(0, best_index - 1)
    upper_index = min(len(coarse_lambdas) - 1, best_index + 1)
    if lower_index == upper_index:
        raise ValueError("At least two coarse lambda values are required for refinement.")
    return np.linspace(coarse_lambdas[lower_index], coarse_lambdas[upper_index], FINE_POINT_COUNT)


def _prepare_catalog(animal, dataset_key, cache_dir):
    if not (cache_dir / "metadata.json").exists():
        print(f"Building exact {animal}_{dataset_key} voxel displacement catalog")
        model.build_displacement_catalog(PROJECT_ROOT, cache_dir, animal=animal, dataset_key=dataset_key)
    catalog = model.load_displacement_catalog(cache_dir)
    if catalog["metadata"]["animal"] != animal or catalog["metadata"]["dataset_key"] != dataset_key:
        raise ValueError("Cached displacement catalog belongs to a different dataset.")
    return catalog


def main(animal="marmoset", dataset_key="majka", output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    cache_dir = output_dir / "preprocessing"
    catalog = _prepare_catalog(animal, dataset_key, cache_dir)
    max_distance = float(np.max(catalog["class_distances"]))
    coarse_lambdas, initial_upper = get_initial_coarse_lambdas(animal, dataset_key)
    coarse_tasks = get_task_table(coarse_lambdas, COARSE_SEEDS, COARSE_CONNECTION_COUNT, "coarse", animal, dataset_key)
    coarse_summary = run_tasks(coarse_tasks, output_dir / "coarse", cache_dir, WORKER_COUNT)
    coarse_best, _ = get_best_lambda(coarse_summary)

    extension = get_boundary_extension(coarse_lambdas, coarse_best, initial_upper, max_distance)
    if len(extension):
        extension_tasks = get_task_table(extension, COARSE_SEEDS, COARSE_CONNECTION_COUNT, "extension", animal, dataset_key)
        extension_summary = run_tasks(extension_tasks, output_dir / "extension", cache_dir, WORKER_COUNT)
        coarse_summary = pd.concat([coarse_summary, extension_summary], ignore_index=True)
        coarse_summary.to_csv(output_dir / "coarse_with_extension_summary.csv", index=False)
        coarse_lambdas = np.unique(np.concatenate([coarse_lambdas, extension]))
        coarse_best, _ = get_best_lambda(coarse_summary)

    fine_lambdas = get_fine_lambdas(coarse_lambdas, coarse_best)
    fine_tasks = get_task_table(fine_lambdas, FINE_SEEDS, FINE_CONNECTION_COUNT, "fine", animal, dataset_key)
    fine_summary = run_tasks(fine_tasks, output_dir / "fine", cache_dir, WORKER_COUNT)
    best_lambda, fine_grouped = get_best_lambda(fine_summary)
    fine_grouped.to_csv(output_dir / "lambda_summary.csv", index=False)

    best_tasks = get_task_table([best_lambda], FINE_SEEDS, FINE_CONNECTION_COUNT, "best", animal, dataset_key)
    best_summary = run_tasks(best_tasks, output_dir / "best", cache_dir, min(WORKER_COUNT, len(best_tasks)), save_pairs=True)
    selected = {
        "best_lambda": best_lambda,
        "animal": animal,
        "dataset_key": dataset_key,
        "mean_correlation": float(best_summary["correlation"].mean()),
        "std_correlation": float(best_summary["correlation"].std()),
        "seed_correlations": {str(int(row.seed)): float(row.correlation) for row in best_summary.itertuples()},
        "coarse_connection_count": COARSE_CONNECTION_COUNT,
        "fine_connection_count": FINE_CONNECTION_COUNT,
    }
    model._save_json(selected, output_dir / "optimal_parameters.json")

    figures_dir = output_dir / "figures"
    plotting.plot_lambda_correlation(fine_summary, figures_dir / "lambda_correlation", best_lambda)
    representative = best_summary.sort_values("seed").iloc[0]
    plotting.plot_connectivity_comparison(
        plotting.load_flne(representative["run_dir"]),
        figures_dir / "best_connectivity_comparison",
        animal=animal,
        dataset_key=dataset_key,
    )
    source_indices = np.load(Path(representative["run_dir"]) / "source_voxel_indices.npy", allow_pickle=False)
    target_indices = np.load(Path(representative["run_dir"]) / "target_voxel_indices.npy", allow_pickle=False)
    plotting.plot_distance_distribution(catalog, source_indices, target_indices, best_lambda, figures_dir / "distance_distribution_validation")
    print(f"Best lambda: {best_lambda:.8g}")
    print(f"Mean correlation: {selected['mean_correlation']:.6f}")


if __name__ == "__main__":
    main()
