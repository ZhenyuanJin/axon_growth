import functools
import shutil
from pathlib import Path

import pandas as pd

from cross_species_3d import analysis
from cross_species_3d import model
from utils_function import cf
from utils_function import model_runtime
from utils_function import utils_function as uf


WORKER_COUNT = 16


@functools.lru_cache(maxsize=5)
def get_empirical_matrix(animal, key):
    """Return a read-only empirical matrix for repeated SC evaluation."""
    return uf.AnimalInfo(animal, key).SC_data


def get_saved_correlation(run_dir, common_regions=False):
    """Evaluate a saved SC matrix using the dataset's measured source and target labels."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    config = model_runtime.load_json(manifest_path)["config"] if manifest_path.exists() else model_runtime.load_json(run_dir / "config.json")
    animal, key = config["animal"], config["dataset_key"]
    common_file = run_dir / "gamanut_allen_common_region_FLNe.csv"
    if common_file.exists() or (animal == "mouse" and common_regions):
        metadata = model_runtime.load_json(analysis.PREPROCESSING_DIR / "mouse_gamanut" / "common_regions" / "metadata.json")
        sources, targets = metadata["source_names"], metadata["target_names"]
        path = common_file if common_file.exists() else run_dir / "connectivity" / "FLNe.csv"
        matrix = pd.read_csv(path, index_col=0).loc[sources, targets]
        matrix = matrix.div(matrix.sum(axis=0), axis=1)
        measured = model.get_empirical_common_regions(sources, targets)
        return float(cf.get_pd_rectangle_corr(measured, matrix, diagonal=False))
    matrix_key = "CCMe_sym_normalized" if animal == "human" else "FLNe"
    matrix = pd.read_csv(run_dir / "connectivity" / f"{matrix_key}.csv", index_col=0)
    matrix.index, matrix.columns = matrix.index.astype(str), matrix.columns.astype(str)
    return float(uf.data_model_correlation(get_empirical_matrix(animal, key), matrix, animal, key, diagonal=False))


def evaluate_run(metrics_path):
    """Save the Pearson correlation for one completed connection-generation run."""
    metrics_path = Path(metrics_path)
    result = model_runtime.load_json(metrics_path)
    result["correlation"] = get_saved_correlation(metrics_path.parent)
    model_runtime.save_json(result, metrics_path)
    return str(metrics_path.parent), result["correlation"]


def save_best_models(dataset_dir):
    """Select the finest saved scan's mean-correlation optimum and materialize its five seed models."""
    fine = pd.read_csv(dataset_dir / "fine" / "summary.csv")
    best_lambda, grouped = model.get_best_lambda(fine)
    grouped.to_csv(dataset_dir / "lambda_summary.csv", index=False)
    selected = fine.loc[fine.Lambda == best_lambda].sort_values("seed")
    if selected.seed.astype(int).tolist() != list(range(5)):
        raise ValueError(f"Five optimal seeds are required: {dataset_dir}")
    best_rows = []
    for result in selected.to_dict(orient="records"):
        task_id = f"best_lambda_000_seed_{int(result['seed']):02d}"
        source, destination = Path(result["run_dir"]), dataset_dir / "best" / task_id
        if source == destination:
            raise ValueError("Fine and best task paths overlap.")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        for name in ["config.json", "metrics.json"]:
            values = model_runtime.load_json(destination / name)
            values["task_id"] = task_id
            model_runtime.save_json(values, destination / name)
        best_rows.append({**result, "task_id": task_id, "run_dir": str(destination)})
    best = pd.DataFrame(best_rows)
    best.to_csv(dataset_dir / "best" / "summary.csv", index=False)
    parameters = {
        "animal": str(best.animal.iloc[0]), "dataset_key": str(best.dataset_key.iloc[0]),
        "best_lambda": best_lambda, "mean_correlation": float(best.correlation.mean()),
        "std_correlation": float(best.correlation.std(ddof=1)),
        "seed_correlations": {str(int(row.seed)): float(row.correlation) for row in best.itertuples()},
    }
    if "distance_metric" in best:
        parameters["distance_metric"] = str(best.distance_metric.iloc[0])
    model_runtime.save_json(parameters, dataset_dir / "optimal_parameters.json")
    return best


def plot_saved_comparisons():
    """Save seed-level model comparisons, RMAE summaries and sampled distance distributions."""
    correlations = pd.read_csv(analysis.MODELS_DIR / "optimal_seed_correlations.csv")
    analysis.plot_correlation_bar(correlations, analysis.FIGURES_DIR / "correlation_2d_3d_bar")
    correlations = pd.read_csv(analysis.MODELS_DIR / "distance_metric_comparison" / "optimal_seed_correlations.csv")
    analysis.plot_distance_metric_correlation_bar(correlations, analysis.FIGURES_DIR / "euclidean_geodesic_correlation_bar")
    frames = []
    for animal, key, loader in [("macaque", "markov40", analysis.get_macaque_pair_table),
                                 ("human", "lariviere", analysis.get_human_pair_table)]:
        data, matrices_2d, matrices_3d = analysis.load_seed_comparison_matrices(animal, key)
        pairs = loader()
        for seed in matrices_2d:
            frame = analysis.get_ratio_binned_rmae(pairs, data, matrices_2d[seed], matrices_3d[seed],
                                                   analysis.RMAE_BIN_COUNT[f"{animal}_{key}"], animal, key)
            frame["seed"] = seed
            frames.append(frame)
    seeds = pd.concat(frames, ignore_index=True)
    seeds.to_csv(analysis.ANALYSIS_DIR / "geodesic_ratio_rmae_by_seed.csv", index=False)
    summary = analysis.get_seed_rmae_summary(seeds)
    summary.to_csv(analysis.ANALYSIS_DIR / "geodesic_ratio_rmae.csv", index=False)
    analysis.plot_rmae(summary, analysis.FIGURES_DIR / "geodesic_ratio_rmae")
    distributions = pd.concat([analysis.get_distance_distribution_validation(f"{animal}_{key}")
                               for animal, key in model.DATASETS], ignore_index=True)
    distributions.to_csv(analysis.ANALYSIS_DIR / "connection_distance_distribution_validation.csv", index=False)
    analysis.plot_distance_distribution_validation(distributions, analysis.FIGURES_DIR / "connection_distance_distribution_validation")


def main():
    roots = [model.PROJECT_ROOT / "results" / "models", analysis.MODELS_DIR]
    paths = sorted(path for root in roots for path in root.rglob("metrics.json"))
    results = cf.multi_process(process_num=WORKER_COUNT, func=evaluate_run,
                               kwargs_list=[{"metrics_path": path} for path in paths],
                               func_name="Saved connection-model evaluation")
    if any(result is None for result in results):
        raise RuntimeError("A saved model could not be evaluated.")
    correlations = dict(results)
    for path in sorted(analysis.MODELS_DIR.rglob("*.csv")):
        if path.name not in ["summary.csv", "2d_summary.csv", "optimal_seed_correlations.csv"]:
            continue
        frame = pd.read_csv(path)
        if not {"run_dir", "correlation"}.issubset(frame.columns):
            continue
        values = []
        for row in frame.to_dict(orient="records"):
            common = row.get("animal") == "mouse" and (row.get("dimension") == "2D" or path.name == "2d_summary.csv")
            values.append(get_saved_correlation(row["run_dir"], common_regions=True) if common else correlations[str(Path(row["run_dir"]))])
        frame["correlation"] = values
        frame.to_csv(path, index=False)
    rows = [pd.read_csv(analysis.MODELS_DIR / "2d_summary.csv")]
    for animal, key in model.DATASETS:
        best = save_best_models(analysis.MODELS_DIR / "3d" / f"{animal}_{key}")
        best["dimension"] = "3D"
        rows.append(best)
    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(analysis.MODELS_DIR / "optimal_seed_correlations.csv", index=False)
    summary.to_csv(analysis.ANALYSIS_DIR / "optimal_seed_correlations.csv", index=False)
    rows = []
    for animal, key in model.DISTANCE_METRIC_DATASETS:
        for metric in ["euclidean", "geodesic"]:
            best = save_best_models(analysis.MODELS_DIR / "distance_metric_comparison" / f"{animal}_{key}" / metric)
            best["distance_metric"] = metric
            rows.append(best)
    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(analysis.MODELS_DIR / "distance_metric_comparison" / "optimal_seed_correlations.csv", index=False)
    summary.to_csv(analysis.ANALYSIS_DIR / "distance_metric_optimal_seed_correlations.csv", index=False)
    plot_saved_comparisons()
    print(f"Evaluated {len(paths)} saved models.", flush=True)


if __name__ == "__main__":
    main()
