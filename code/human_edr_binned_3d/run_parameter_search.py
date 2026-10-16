from pathlib import Path

import numpy as np
import pandas as pd
from human_edr_binned_3d import model
from human_edr_binned_3d import plotting
from utils_function import cf
from utils_function import utils_function as uf
from voxel_edr_3d.run_parameter_search import COARSE_CONNECTION_COUNT
from voxel_edr_3d.run_parameter_search import COARSE_POINT_COUNT
from voxel_edr_3d.run_parameter_search import COARSE_SEEDS
from voxel_edr_3d.run_parameter_search import FINE_CONNECTION_COUNT
from voxel_edr_3d.run_parameter_search import FINE_POINT_COUNT
from voxel_edr_3d.run_parameter_search import FINE_SEEDS
from voxel_edr_3d.run_parameter_search import INITIAL_MIN_LAMBDA
from voxel_edr_3d.run_parameter_search import get_best_lambda
from voxel_edr_3d.run_parameter_search import get_boundary_extension
from voxel_edr_3d.run_parameter_search import get_fine_lambdas

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "human_edr_binned_3d"
CACHE_DIR = OUTPUT_DIR / "preprocessing"
BIN_WIDTH_MM = 0.1
WORKER_COUNT = 10


def _prepare_catalog():
    if not (CACHE_DIR / "metadata.json").exists():
        print("Building corrected human 0.1 mm distance-bin catalog")
        model.build_bin_catalog(PROJECT_ROOT, CACHE_DIR, bin_width=BIN_WIDTH_MM)
    catalog = model.load_bin_catalog(CACHE_DIR)
    if catalog["metadata"]["bin_width_mm"] != BIN_WIDTH_MM:
        raise ValueError("Cached bin width does not match BIN_WIDTH_MM.")
    return catalog


def _probability_path(stage_dir, lambda_index):
    return stage_dir / "probabilities" / f"lambda_{lambda_index:03d}.npy"


def prepare_probabilities(lambda_values, stage_dir):
    """Save one corrected area-probability matrix per lambda."""
    for lambda_index, Lambda in enumerate(lambda_values):
        model.save_area_probability_matrix(
            CACHE_DIR, float(Lambda), _probability_path(stage_dir, lambda_index)
        )


def get_task_table(lambda_values, seeds, connection_count, stage, stage_dir):
    """Return deterministic seed tasks for a parameter-search stage."""
    tasks = []
    for lambda_index, Lambda in enumerate(lambda_values):
        probability_path = _probability_path(stage_dir, lambda_index)
        for seed in seeds:
            tasks.append({
                "task_id": f"{stage}_lambda_{lambda_index:03d}_seed_{seed:02d}",
                "Lambda": float(Lambda),
                "seed": int(seed),
                "connection_count": int(connection_count),
                "probability_path": str(probability_path),
            })
    return tasks


def _run_task(task, cache_dir, output_dir, save_pairs=False):
    return model.run_connection_task(task, cache_dir, output_dir, save_pairs=save_pairs)


def run_tasks(tasks, stage_dir, save_pairs=False):
    """Run tasks and save a task-id-sorted summary."""
    results = cf.multi_process(
        process_num=min(WORKER_COUNT, len(tasks)),
        func=_run_task,
        kwargs_list=[
            {"task": task, "cache_dir": CACHE_DIR, "output_dir": stage_dir, "save_pairs": save_pairs}
            for task in tasks
        ],
        func_name=f"corrected human binned EDR {stage_dir.name}",
    )
    results = sorted(results, key=lambda item: item["task_id"])
    summary = pd.DataFrame(results)
    summary.to_csv(stage_dir / "summary.csv", index=False)
    return summary


def run_stage(lambda_values, seeds, connection_count, stage, save_pairs=False):
    stage_dir = OUTPUT_DIR / stage
    prepare_probabilities(lambda_values, stage_dir)
    tasks = get_task_table(lambda_values, seeds, connection_count, stage, stage_dir)
    return run_tasks(tasks, stage_dir, save_pairs=save_pairs)


def main():
    catalog = _prepare_catalog()
    info = uf.AnimalInfo("human", "lariviere")
    initial_upper = float(np.max(info.distance_3D.values) / 2)
    max_distance = catalog["metadata"]["maximum_distance_mm"]
    coarse_lambdas = np.geomspace(INITIAL_MIN_LAMBDA, initial_upper, COARSE_POINT_COUNT)
    coarse_summary = run_stage(coarse_lambdas, COARSE_SEEDS, COARSE_CONNECTION_COUNT, "coarse")
    coarse_best, _ = get_best_lambda(coarse_summary)
    extension = get_boundary_extension(coarse_lambdas, coarse_best, initial_upper, max_distance)
    if len(extension):
        extension_summary = run_stage(extension, COARSE_SEEDS, COARSE_CONNECTION_COUNT, "extension")
        coarse_summary = pd.concat([coarse_summary, extension_summary], ignore_index=True)
        coarse_summary.to_csv(OUTPUT_DIR / "coarse_with_extension_summary.csv", index=False)
        coarse_lambdas = np.unique(np.concatenate([coarse_lambdas, extension]))
        coarse_best, _ = get_best_lambda(coarse_summary)

    fine_lambdas = get_fine_lambdas(coarse_lambdas, coarse_best)
    fine_summary = run_stage(fine_lambdas, FINE_SEEDS, FINE_CONNECTION_COUNT, "fine")
    best_lambda, fine_grouped = get_best_lambda(fine_summary)
    fine_grouped.to_csv(OUTPUT_DIR / "lambda_summary.csv", index=False)

    best_stage_dir = OUTPUT_DIR / "best"
    best_probability_path = _probability_path(best_stage_dir, 0)
    model.save_area_probability_matrix(CACHE_DIR, best_lambda, best_probability_path)
    best_tasks = get_task_table([best_lambda], FINE_SEEDS, FINE_CONNECTION_COUNT, "best", best_stage_dir)
    best_summary = run_tasks(best_tasks, best_stage_dir, save_pairs=True)
    selected = {
        "animal": "human",
        "dataset_key": "lariviere",
        "bin_width_mm": BIN_WIDTH_MM,
        "best_lambda": best_lambda,
        "mean_correlation": float(best_summary["correlation"].mean()),
        "std_correlation": float(best_summary["correlation"].std()),
        "seed_correlations": {str(int(row.seed)): float(row.correlation) for row in best_summary.itertuples()},
        "coarse_connection_count": COARSE_CONNECTION_COUNT,
        "fine_connection_count": FINE_CONNECTION_COUNT,
    }
    model._save_json(selected, OUTPUT_DIR / "optimal_parameters.json")

    figures_dir = OUTPUT_DIR / "figures"
    plotting.plot_lambda_correlation(fine_summary, figures_dir / "lambda_correlation", best_lambda)
    representative = best_summary.sort_values("seed").iloc[0]
    plotting.plot_connectivity_comparison(
        plotting.load_connectivity(representative["run_dir"]),
        figures_dir / "best_connectivity_comparison",
    )
    run_dir = Path(representative["run_dir"])
    sources = np.load(run_dir / "source_vertex_indices.npy", mmap_mode="r")
    targets = np.load(run_dir / "target_vertex_indices.npy", mmap_mode="r")
    centers, expected, empirical = model.get_distance_histograms(catalog, sources, targets, best_lambda)
    histogram = pd.DataFrame({
        "distance_mm": centers,
        "target_probability": expected,
        "empirical_probability": empirical,
    })
    histogram.to_csv(OUTPUT_DIR / "distance_distribution_validation.csv", index=False)
    plotting.plot_distance_histogram(
        centers, expected, empirical, figures_dir / "distance_distribution_validation"
    )
    print(f"Best lambda: {best_lambda:.8g}")
    print(f"Mean correlation: {selected['mean_correlation']:.6f}")


if __name__ == "__main__":
    main()
