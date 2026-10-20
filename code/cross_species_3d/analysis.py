import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
import scipy.special
import scipy.spatial
from cross_species_3d import model
from utils_function import cf
from utils_function import utils_function as uf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "cross_species_3d"
PREPROCESSING_DIR = OUTPUT_DIR / "preprocessing"
MODELS_DIR = OUTPUT_DIR / "models"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"
SPECIES_ORDER = ["Mouse", "Marmoset", "Macaque", "Human"]
RMAE_BIN_COUNT = {"macaque_markov40": 20, "human_lariviere": 10}
HUMAN_GEODESIC_RMAE_BIN_WIDTH_MM = 20.0
DISTANCE_DISTRIBUTION_BIN_COUNT = 40
DISTANCE_DISTRIBUTION_CONNECTION_COUNT = 8_000_000
DISTANCE_DISTRIBUTION_SEEDS = range(5)


def load_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def get_mouse_allen_distance_matrices():
    """Return aligned Gamanut 2D and Allen CCF2017 3D distances for common regions."""
    info = uf.AnimalInfo("mouse", "gamanut")
    distance_3d = info.distance_3D
    area_names = distance_3d.index
    distance_2d = info.distance_2D.loc[area_names, area_names]
    return distance_2d, distance_3d


def get_connectivity_path(run_dir, dataset_name, dimension):
    """Return the saved comparison matrix for one final seed."""
    run_dir = Path(run_dir)
    if dimension == "3D" and dataset_name == "mouse_gamanut":
        return run_dir / "gamanut_allen_common_region_FLNe.csv"
    filename = "CCMe_sym_normalized.csv" if dataset_name == "human_lariviere" else "FLNe.csv"
    return run_dir / "connectivity" / filename


def load_seed_model_matrices(dataset_name, dimension):
    """Load optimal connectivity matrices keyed by seed without averaging."""
    if dimension == "2D":
        summary = pd.read_csv(MODELS_DIR / "2d_summary.csv")
        summary = summary[(summary["animal"] + "_" + summary["dataset_key"]) == dataset_name]
    else:
        summary = pd.read_csv(MODELS_DIR / "3d" / dataset_name / "best" / "summary.csv")
    summary = summary.sort_values("seed")
    expected_seeds = list(range(5))
    seeds = summary["seed"].astype(int).tolist()
    if seeds != expected_seeds:
        raise ValueError(f"Expected {dimension} seeds {expected_seeds} for {dataset_name}, found {seeds}.")
    matrices = {
        int(row.seed): pd.read_csv(get_connectivity_path(row.run_dir, dataset_name, dimension), index_col=0)
        for row in summary.itertuples()
    }
    reference = matrices[expected_seeds[0]]
    if any(not matrix.index.equals(reference.index) or not matrix.columns.equals(reference.columns) for matrix in matrices.values()):
        raise ValueError(f"Unaligned {dimension} matrices for {dataset_name}.")
    return matrices


def load_seed_comparison_matrices(animal, dataset_key):
    """Return data and aligned 2D/3D model matrices keyed by seed."""
    dataset_name = f"{animal}_{dataset_key}"
    model_2d = load_seed_model_matrices(dataset_name, "2D")
    model_3d = load_seed_model_matrices(dataset_name, "3D")
    if model_2d.keys() != model_3d.keys():
        raise ValueError(f"2D and 3D seeds differ for {dataset_name}.")
    info = uf.AnimalInfo(animal, dataset_key)
    if dataset_name == "mouse_gamanut":
        metadata = load_json(PREPROCESSING_DIR / dataset_name / "common_regions" / "metadata.json")
        source_names = metadata["source_names"]
        target_names = metadata["target_names"]
        data = info.SC_data.loc[source_names, target_names].astype(float)
        data = data.div(data.sum(axis=0), axis=1)
        for seed in model_2d:
            matrix = model_2d[seed].loc[source_names, target_names].astype(float)
            model_2d[seed] = matrix.div(matrix.sum(axis=0), axis=1)
            model_3d[seed] = model_3d[seed].loc[source_names, target_names]
    else:
        names = info.full_name
        data = info.SC_data.loc[names, names]
        for seed in model_2d:
            model_2d[seed] = model_2d[seed].loc[names, names]
            model_3d[seed] = model_3d[seed].loc[names, names]
    return data, model_2d, model_3d


def load_distance_metric_seed_matrices(dataset_name, distance_metric):
    """Load five optimal full-pair EDR matrices for one distance metric."""
    best_dir = MODELS_DIR / "distance_metric_comparison" / dataset_name / distance_metric / "best"
    summary = pd.read_csv(best_dir / "summary.csv").sort_values("seed")
    expected_seeds = list(range(5))
    seeds = summary["seed"].astype(int).tolist()
    if seeds != expected_seeds:
        raise ValueError(f"Expected {distance_metric} seeds {expected_seeds}, found {seeds}.")
    filename = "CCMe_sym_normalized.csv" if dataset_name == "human_lariviere" else "FLNe.csv"
    matrices = {
        int(row.seed): pd.read_csv(Path(row.run_dir) / "connectivity" / filename, index_col=0)
        for row in summary.itertuples()
    }
    reference = matrices[expected_seeds[0]]
    if any(not matrix.index.equals(reference.index) or not matrix.columns.equals(reference.columns) for matrix in matrices.values()):
        raise ValueError(f"Unaligned {distance_metric} matrices for {dataset_name}.")
    return matrices


def get_full_pair_area_distance_means(dataset_name):
    """Return full-pair area means from fixed-width distance catalogs."""
    root = PREPROCESSING_DIR / dataset_name / "full_pair_distances"
    means = {}
    names = None
    for metric in ["euclidean", "geodesic"]:
        catalog_dir = root / f"{metric}_catalog"
        metadata = load_json(catalog_dir / "metadata.json")
        local_names = np.asarray(metadata["area_names"])
        if names is not None and not np.array_equal(names, local_names):
            raise ValueError(f"Euclidean/geodesic area names differ for {dataset_name}.")
        names = local_names
        edges = np.load(catalog_dir / "bin_edges.npy", mmap_mode="r")
        counts = np.load(catalog_dir / "area_pair_bin_counts.npy", mmap_mode="r")
        centers = (edges[:-1] + edges[1:]) / 2
        pair_counts = np.sum(counts, axis=0, dtype=np.uint64)
        distance_sums = np.tensordot(centers, counts, axes=(0, 0))
        means[metric] = np.divide(
            distance_sums, pair_counts, out=np.full(distance_sums.shape, np.nan), where=pair_counts > 0
        )
    return names, means["euclidean"], means["geodesic"]


def get_human_pair_table():
    """Return full-vertex human area-pair geodesic/Euclidean ratios."""
    names, euclidean_mean, geodesic_mean = get_full_pair_area_distance_means("human_lariviere")
    row, column = np.triu_indices(len(names), k=1)
    ratio = geodesic_mean[row, column] / euclidean_mean[row, column]
    if np.any(~np.isfinite(ratio)) or np.any(ratio < 1):
        raise ValueError("Invalid human geodesic-to-Euclidean area-pair ratios.")
    return pd.DataFrame({
        "source_area": names[row],
        "target_area": names[column],
        "euclidean_distance_mm": euclidean_mean[row, column],
        "geodesic_distance_mm": geodesic_mean[row, column],
        "geodesic_to_euclidean_ratio": ratio,
    })


def get_human_center_distance_matrices():
    """Return aligned 2D, 3D Euclidean, and surface-geodesic center distances."""
    dataset_name = "human_lariviere"
    cache_dir = PREPROCESSING_DIR / dataset_name / "full_pair_distances"
    metadata = load_json(cache_dir / "metadata.json")
    names = list(metadata["area_names"])
    coordinates = np.load(cache_dir / "coordinates.npy", mmap_mode="r")
    area_ids = np.load(cache_dir / "area_ids.npy", mmap_mode="r")
    graph = scipy.sparse.load_npz(cache_dir / "largest_component_graph.npz")
    center_indices = []
    for area_id in range(len(names)):
        local_indices = np.flatnonzero(area_ids == area_id)
        if len(local_indices) == 0:
            raise ValueError(f"Human area {names[area_id]} has no surface vertices.")
        centroid = np.mean(coordinates[local_indices], axis=0)
        local_center = int(np.argmin(np.linalg.norm(coordinates[local_indices] - centroid, axis=1)))
        center_indices.append(int(local_indices[local_center]))
    center_indices = np.asarray(center_indices, dtype=np.int32)
    center_coordinates = np.asarray(coordinates[center_indices], dtype=np.float64)
    euclidean_3d = scipy.spatial.distance.cdist(center_coordinates, center_coordinates)
    geodesic_3d = scipy.sparse.csgraph.dijkstra(
        graph, directed=False, indices=center_indices, return_predecessors=False
    )[:, center_indices]
    if not np.all(np.isfinite(geodesic_3d)) or np.any(geodesic_3d + 1e-10 < euclidean_3d):
        raise ValueError("Invalid human center geodesic distances.")
    info = uf.AnimalInfo("human", "lariviere")
    distance_2d = info.distance_2D.loc[names, names].astype(float)
    matrices = {
        "2D Euclidean": distance_2d,
        "3D Euclidean": pd.DataFrame(euclidean_3d, index=names, columns=names),
        "3D geodesic": pd.DataFrame(geodesic_3d, index=names, columns=names),
    }
    return center_indices, matrices


def get_distance_matrix_similarity(distance_matrices):
    """Return pairwise Pearson correlations across aligned upper triangles."""
    labels = list(distance_matrices)
    first = distance_matrices[labels[0]]
    if any(not matrix.index.equals(first.index) or not matrix.columns.equals(first.columns) for matrix in distance_matrices.values()):
        raise ValueError("Center-distance matrices are not aligned.")
    row, column = np.triu_indices(len(first), k=1)
    vectors = np.asarray([distance_matrices[label].values[row, column] for label in labels])
    return pd.DataFrame(np.corrcoef(vectors), index=labels, columns=labels)


def get_macaque_pair_table():
    """Return full-voxel macaque comparison pairs and physical distance ratios."""
    names, euclidean_mean, geodesic_mean = get_full_pair_area_distance_means("macaque_markov40")
    info = uf.AnimalInfo("macaque", "markov40")
    lookup = {name: index for index, name in enumerate(names)}
    row_names = np.asarray(info.source_name)
    column_names = np.asarray(info.target_name)
    row_ids = np.asarray([lookup[name] for name in row_names])[:, np.newaxis]
    column_ids = np.asarray([lookup[name] for name in column_names])[np.newaxis, :]
    ratio = geodesic_mean[row_ids, column_ids] / euclidean_mean[row_ids, column_ids]
    row, column = np.indices(ratio.shape)
    selected = row_names[:, None] != column_names[None, :]
    return pd.DataFrame({
        "source_area": row_names[row[selected]],
        "target_area": column_names[column[selected]],
        "geodesic_to_euclidean_ratio": ratio[selected],
    })


def get_ratio_binned_rmae(pair_table, data, model_2d, model_3d, bin_count, animal, dataset_key):
    """Return existing- and non-link RMAE on shared physical-ratio bins."""
    relative_error_2d = uf.data_model_relative_error(data, model_2d, animal, dataset_key, diagonal=False)
    relative_error_3d = uf.data_model_relative_error(data, model_3d, animal, dataset_key, diagonal=False)
    source_index = relative_error_2d.index.get_indexer(pair_table["source_area"])
    target_index = relative_error_2d.columns.get_indexer(pair_table["target_area"])
    if np.any(source_index < 0) or np.any(target_index < 0):
        raise ValueError("Area-pair names are not aligned with the RMAE matrices.")
    ratio = pair_table["geodesic_to_euclidean_ratio"].to_numpy(dtype=np.float64)
    error_2d = relative_error_2d.values[source_index, target_index]
    error_3d = relative_error_3d.values[source_index, target_index]
    if not np.all(np.isfinite(ratio)) or not np.all(np.isfinite(error_2d)) or not np.all(np.isfinite(error_3d)):
        raise ValueError("Distance ratios and pairwise relative errors must be finite.")
    edges = np.quantile(ratio, np.linspace(0, 1, bin_count + 1))
    if len(np.unique(edges)) != len(edges):
        raise ValueError("Distance-ratio quantiles do not define unique bin edges.")
    bin_id = np.clip(np.searchsorted(edges, ratio, side="right") - 1, 0, bin_count - 1)
    pooled_ratio_mean = np.asarray([np.mean(ratio[bin_id == local_bin]) for local_bin in range(bin_count)])
    masks = uf.get_weight_category_mask(animal, dataset_key, diagonal=False)
    rows = []
    for category_label, category_key in [("existing", "all"), ("non-link", "non")]:
        category = masks[category_key].values[source_index, target_index]
        for local_bin in range(bin_count):
            selected = category & (bin_id == local_bin)
            pair_count = int(np.sum(selected))
            rows.append({
                "animal": animal, "link_category": category_label, "bin": local_bin + 1,
                "ratio_left": float(edges[local_bin]), "ratio_mean": float(pooled_ratio_mean[local_bin]),
                "ratio_right": float(edges[local_bin + 1]), "pair_count": pair_count,
                "rmae_2d": float(np.mean(error_2d[selected])) if pair_count else np.nan,
                "rmae_3d": float(np.mean(error_3d[selected])) if pair_count else np.nan,
            })
    return pd.DataFrame(rows)


def get_seed_rmae_summary(seed_table):
    """Average independently computed seed RMAEs and retain seed variability."""
    group_columns = [
        "animal", "link_category", "bin", "ratio_left", "ratio_mean", "ratio_right", "pair_count"
    ]
    summary = seed_table.groupby(group_columns, sort=False, dropna=False).agg(
        seed_count=("seed", "nunique"),
        rmae_2d_mean=("rmae_2d", "mean"),
        rmae_2d_std=("rmae_2d", "std"),
        rmae_3d_mean=("rmae_3d", "mean"),
        rmae_3d_std=("rmae_3d", "std"),
    ).reset_index()
    if not np.all(summary["seed_count"] == 5):
        raise ValueError("Every RMAE bin must contain five independently computed seeds.")
    return summary


def get_geodesic_distance_binned_error(pair_table, relative_error, bin_edges):
    """Return one model's pairwise relative error on fixed geodesic-distance bins."""
    source_index = relative_error.index.get_indexer(pair_table["source_area"])
    target_index = relative_error.columns.get_indexer(pair_table["target_area"])
    if np.any(source_index < 0) or np.any(target_index < 0):
        raise ValueError("Human area-pair names are not aligned with the RMAE matrices.")
    distances = pair_table["geodesic_distance_mm"].to_numpy(dtype=np.float64)
    error = relative_error.values[source_index, target_index]
    if not np.all(np.isfinite(distances)) or not np.all(np.isfinite(error)):
        raise ValueError("Geodesic distances and pairwise relative errors must be finite.")
    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    if np.any(np.diff(bin_edges) <= 0) or distances.min() < bin_edges[0] or distances.max() > bin_edges[-1]:
        raise ValueError("Geodesic-distance bin edges do not cover all area pairs.")
    bin_count = len(bin_edges) - 1
    bin_id = np.clip(np.searchsorted(bin_edges, distances, side="right") - 1, 0, bin_count - 1)
    pooled_distance_mean = np.asarray([
        np.mean(distances[bin_id == local_bin]) if np.any(bin_id == local_bin) else np.nan
        for local_bin in range(bin_count)
    ])
    masks = uf.get_weight_category_mask("human", "lariviere", diagonal=False)
    rows = []
    for category_label, category_key in [("existing", "all"), ("non-link", "non")]:
        category = masks[category_key].values[source_index, target_index]
        for local_bin in range(bin_count):
            selected = category & (bin_id == local_bin)
            pair_count = int(np.sum(selected))
            rows.append({
                "link_category": category_label,
                "bin": local_bin + 1,
                "distance_left_mm": float(bin_edges[local_bin]),
                "geodesic_distance_mean_mm": float(pooled_distance_mean[local_bin]),
                "distance_right_mm": float(bin_edges[local_bin + 1]),
                "pair_count": pair_count,
                "rmae": float(np.mean(error[selected])) if pair_count else np.nan,
            })
    return pd.DataFrame(rows)


def get_geodesic_distance_binned_rmae(pair_table, data, euclidean_model, geodesic_model, bin_edges):
    """Return human Euclidean/geodesic model RMAE on shared geodesic bins."""
    tables = []
    for label, matrix in [("euclidean", euclidean_model), ("geodesic", geodesic_model)]:
        relative_error = uf.data_model_relative_error(
            data, matrix, "human", "lariviere", diagonal=False
        )
        local = get_geodesic_distance_binned_error(pair_table, relative_error, bin_edges)
        local["distance_metric"] = label
        tables.append(local)
    long_table = pd.concat(tables, ignore_index=True)
    index_columns = [
        "link_category", "bin", "distance_left_mm", "geodesic_distance_mean_mm",
        "distance_right_mm", "pair_count",
    ]
    result = long_table.pivot(index=index_columns, columns="distance_metric", values="rmae").reset_index()
    result.columns.name = None
    return result.rename(columns={"euclidean": "rmae_euclidean", "geodesic": "rmae_geodesic"})


def get_geodesic_distance_rmae_summary(seed_table):
    """Average independently computed human distance-bin RMAEs across seeds."""
    group_columns = [
        "link_category", "bin", "distance_left_mm", "geodesic_distance_mean_mm",
        "distance_right_mm", "pair_count",
    ]
    summary = seed_table.groupby(group_columns, sort=False, dropna=False).agg(
        seed_count=("seed", "nunique"),
        rmae_euclidean_mean=("rmae_euclidean", "mean"),
        rmae_euclidean_std=("rmae_euclidean", "std"),
        rmae_geodesic_mean=("rmae_geodesic", "mean"),
        rmae_geodesic_std=("rmae_geodesic", "std"),
    ).reset_index()
    if not np.all(summary["seed_count"] == 5):
        raise ValueError("Every geodesic-distance RMAE bin must contain five seeds.")
    summary["rmae_difference_mean"] = summary["rmae_geodesic_mean"] - summary["rmae_euclidean_mean"]
    return summary


def get_human_distance_lambda_control_summary():
    """Return seed and aggregate metrics for the human distance-by-lambda control."""
    path = MODELS_DIR / "distance_lambda_control" / "human_lariviere" / "summary.csv"
    control = pd.read_csv(path).sort_values(["distance_metric", "Lambda", "seed"])
    info = uf.AnimalInfo("human", "lariviere")
    data = info.SC_data.loc[info.full_name, info.full_name]
    masks = uf.get_weight_category_mask("human", "lariviere", diagonal=False)
    rows = []
    for result in control.itertuples():
        matrix = pd.read_csv(Path(result.run_dir) / "connectivity" / "CCMe_sym_normalized.csv", index_col=0)
        matrix = matrix.loc[info.full_name, info.full_name]
        relative_error = uf.data_model_relative_error(
            data, matrix, "human", "lariviere", diagonal=False
        )
        row = result._asdict()
        for label, key in [("existing", "all"), ("non_link", "non")]:
            mask = masks[key].loc[info.full_name, info.full_name].values
            row[f"rmae_{label}"] = float(np.nansum(relative_error.values * mask) / np.sum(mask))
        rows.append(row)
    seed_summary = pd.DataFrame(rows)
    group_columns = ["distance_metric", "lambda_source", "Lambda"]
    aggregate = seed_summary.groupby(group_columns, sort=False).agg(
        seed_count=("seed", "nunique"),
        correlation_mean=("correlation", "mean"),
        correlation_std=("correlation", "std"),
        rmae_existing_mean=("rmae_existing", "mean"),
        rmae_existing_std=("rmae_existing", "std"),
        rmae_non_link_mean=("rmae_non_link", "mean"),
        rmae_non_link_std=("rmae_non_link", "std"),
    ).reset_index()
    if len(aggregate) != 4 or not np.all(aggregate["seed_count"] == 5):
        raise ValueError("Human distance-by-lambda control must contain four conditions and five seeds.")
    return seed_summary, aggregate


def get_human_distance_lambda_control_binned_rmae(pair_table, data, control, bin_edges):
    """Return per-seed RMAE for all human distance-by-lambda conditions on shared bins."""
    rows = []
    for result in control.itertuples():
        matrix = pd.read_csv(Path(result.run_dir) / "connectivity" / "CCMe_sym_normalized.csv", index_col=0)
        relative_error = uf.data_model_relative_error(
            data, matrix, "human", "lariviere", diagonal=False
        )
        local = get_geodesic_distance_binned_error(pair_table, relative_error, bin_edges)
        local["distance_metric"] = result.distance_metric
        local["lambda_source"] = result.lambda_source
        local["Lambda"] = result.Lambda
        local["seed"] = result.seed
        rows.append(local)
    return pd.concat(rows, ignore_index=True)


def get_human_distance_lambda_control_binned_summary(seed_table):
    """Average independently computed distance-by-lambda RMAEs across five seeds."""
    group_columns = [
        "distance_metric", "lambda_source", "Lambda", "link_category", "bin",
        "distance_left_mm", "geodesic_distance_mean_mm", "distance_right_mm", "pair_count",
    ]
    summary = seed_table.groupby(group_columns, sort=False, dropna=False).agg(
        seed_count=("seed", "nunique"), rmae_mean=("rmae", "mean"), rmae_std=("rmae", "std")
    ).reset_index()
    if not np.all(summary["seed_count"] == 5):
        raise ValueError("Every distance-by-lambda RMAE bin must contain five seeds.")
    return summary


def get_native_distance_distribution(dataset_name):
    """Return native distance-class probabilities and their analytical EDR target."""
    parameter_path = MODELS_DIR / "3d" / dataset_name / "optimal_parameters.json"
    Lambda = float(load_json(parameter_path)["best_lambda"])
    cache_dir = PREPROCESSING_DIR / dataset_name / "distance_catalog"
    if dataset_name == "human_lariviere":
        catalog = model.load_human_bin_catalog(cache_dir)
        edges = np.asarray(catalog["bin_edges"])
        distances = (edges[:-1] + edges[1:]) / 2
        target = model.get_binned_distance_probabilities(catalog, Lambda)
        pair_weights = model.get_binned_pair_weights(catalog, Lambda)
        represented = pair_weights * np.asarray(catalog["global_bin_counts"])
    else:
        catalog = model.load_displacement_catalog(cache_dir)
        distances = np.asarray(catalog["class_distances"])
        log_weights = -distances / Lambda
        target = np.exp(log_weights - scipy.special.logsumexp(log_weights))
        displacement_probabilities = model.get_displacement_probabilities(catalog, Lambda)
        represented = np.bincount(
            np.asarray(catalog["distance_class_ids"]),
            weights=displacement_probabilities,
            minlength=len(distances),
        )
    occupied = target > 0
    distances = distances[occupied]
    target = target[occupied]
    represented = represented[occupied]
    maximum_error = float(np.max(np.abs(represented - target)))
    if maximum_error > 1e-12 or not np.isclose(np.sum(represented), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"Distance-class correction failed for {dataset_name}: {maximum_error}")
    return distances, represented / np.sum(represented), target, Lambda, maximum_error


def get_distance_distribution_validation(dataset_name):
    """Return count-corrected sampled and target distance distributions."""
    distances, probabilities, target, Lambda, maximum_error = get_native_distance_distribution(dataset_name)
    cumulative = np.cumsum(target)
    cutoff_index = min(int(np.searchsorted(cumulative, 0.999, side="left")), len(distances) - 1)
    plot_maximum = float(distances[cutoff_index])
    edges = np.linspace(float(np.min(distances)), plot_maximum, DISTANCE_DISTRIBUTION_BIN_COUNT + 1)
    display_bin = np.searchsorted(edges, distances, side="right") - 1
    display_bin = np.clip(display_bin, 0, DISTANCE_DISTRIBUTION_BIN_COUNT - 1)
    in_range = distances <= plot_maximum
    native_class_count = np.bincount(display_bin[in_range], minlength=DISTANCE_DISTRIBUTION_BIN_COUNT)
    target_mass = np.bincount(display_bin[in_range], weights=target[in_range], minlength=DISTANCE_DISTRIBUTION_BIN_COUNT)
    sampled = []
    for seed in DISTANCE_DISTRIBUTION_SEEDS:
        rng = cf.get_local_rng(seed=int(seed))
        counts = rng.multinomial(DISTANCE_DISTRIBUTION_CONNECTION_COUNT, probabilities)
        sampled_mass = np.bincount(
            display_bin[in_range], weights=counts[in_range], minlength=DISTANCE_DISTRIBUTION_BIN_COUNT
        ) / DISTANCE_DISTRIBUTION_CONNECTION_COUNT
        sampled.append(sampled_mass)
    sampled = np.asarray(sampled)
    valid = native_class_count > 0
    centers = np.asarray([
        np.mean(distances[in_range][display_bin[in_range] == local_bin])
        for local_bin in np.flatnonzero(valid)
    ])
    return pd.DataFrame({
        "dataset_name": dataset_name,
        "Lambda": Lambda,
        "distance_mm": centers,
        "native_distance_class_count": native_class_count[valid],
        "target_probability_per_class": target_mass[valid] / native_class_count[valid],
        "sampled_probability_per_class_mean": np.mean(sampled[:, valid], axis=0) / native_class_count[valid],
        "sampled_probability_per_class_std": np.std(sampled[:, valid], axis=0, ddof=1) / native_class_count[valid],
        "represented_class_maximum_error": maximum_error,
    })


def plot_correlation_bar(summary, output_path):
    """Plot optimal 2D and 3D mean correlations with seed standard deviations."""
    figure_data = summary.copy()
    figure_data["species"] = figure_data["animal"].str.capitalize()
    grouped = figure_data.groupby(["species", "dimension"])["correlation"].agg(["mean", "std"])
    x = np.arange(len(SPECIES_ORDER), dtype=np.float64)
    width = 0.36
    mean_2d = grouped.loc[(SPECIES_ORDER, "2D"), "mean"].to_numpy()
    mean_3d = grouped.loc[(SPECIES_ORDER, "3D"), "mean"].to_numpy()
    std_2d = grouped.loc[(SPECIES_ORDER, "2D"), "std"].to_numpy()
    std_3d = grouped.loc[(SPECIES_ORDER, "3D"), "std"].to_numpy()
    fig, ax = cf.get_fig_ax()
    cf.plt_bar(ax, x - width / 2, mean_2d, label="2D", color=cf.BLUE, err=std_2d, width=width)
    cf.plt_bar(ax, x + width / 2, mean_3d, label="3D", color=cf.ORANGE, err=std_3d, width=width)
    cf.set_ax(ax, xlabel="", ylabel="Pearson correlation", xtick=x, xtick_label=SPECIES_ORDER, legend_loc="lower center", bbox_to_anchor=(0.5, 1.0), legend_kwargs={"ncols": 2})
    cf.save_fig(fig, str(output_path), formats=["png", "pdf"], pkl=False)


def plot_distance_metric_correlation_bar(summary, output_path):
    """Plot optimal Euclidean and geodesic EDR correlations with seed errors."""
    figure_data = summary.copy()
    figure_data["species"] = figure_data["animal"].str.capitalize()
    grouped = figure_data.groupby(["species", "distance_metric"])["correlation"].agg(["mean", "std"])
    species_order = ["Macaque", "Human"]
    x = np.arange(len(species_order), dtype=np.float64)
    width = 0.36
    euclidean_mean = grouped.loc[(species_order, "euclidean"), "mean"].to_numpy()
    geodesic_mean = grouped.loc[(species_order, "geodesic"), "mean"].to_numpy()
    euclidean_std = grouped.loc[(species_order, "euclidean"), "std"].to_numpy()
    geodesic_std = grouped.loc[(species_order, "geodesic"), "std"].to_numpy()
    fig, ax = cf.get_fig_ax()
    cf.plt_bar(ax, x - width / 2, euclidean_mean, label="Euclidean", color=cf.BLUE, err=euclidean_std, width=width)
    cf.plt_bar(ax, x + width / 2, geodesic_mean, label="Geodesic", color=cf.ORANGE, err=geodesic_std, width=width)
    cf.set_ax(
        ax, xlabel="", ylabel="Pearson correlation", xtick=x, xtick_label=species_order,
        legend_loc="lower center", bbox_to_anchor=(0.5, 1.0), legend_kwargs={"ncols": 2},
    )
    cf.save_fig(fig, str(output_path), formats=["png", "pdf"], pkl=False)


def _plot_finite_segments(ax, x, y, label, **kwargs):
    finite = np.isfinite(y)
    starts = np.flatnonzero(finite & ~np.r_[False, finite[:-1]])
    stops = np.flatnonzero(finite & ~np.r_[finite[1:], False]) + 1
    for segment_index, (start, stop) in enumerate(zip(starts, stops)):
        cf.plt_line(ax, x[start:stop], y[start:stop], label=label if segment_index == 0 else None, **kwargs)


def plot_rmae(binned, output_path):
    """Plot seed-mean macaque and human physical-ratio RMAE in two panels."""
    fig, axes = cf.get_fig_ax(ncols=2)
    for ax, animal in zip(axes, ["macaque", "human"]):
        local = binned[binned["animal"] == animal]
        for category, marker, linestyle in [("existing", "o", "-"), ("non-link", "s", "--")]:
            table = local[local["link_category"] == category]
            x = table["ratio_mean"].to_numpy()
            _plot_finite_segments(ax, x, table["rmae_2d_mean"].to_numpy(), f"2D {category}", color=cf.BLUE, marker=marker, linestyle=linestyle)
            _plot_finite_segments(ax, x, table["rmae_3d_mean"].to_numpy(), f"3D {category}", color=cf.ORANGE, marker=marker, linestyle=linestyle)
        cf.set_ax(ax, xlabel="Mean geodesic / Euclidean distance", ylabel="RMAE", title=animal.capitalize(), legend=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4)
    cf.save_fig(fig, str(output_path), formats=["png", "pdf"], pkl=False)


def plot_human_geodesic_distance_rmae(binned, output_path):
    """Plot human optimal-model RMAE on shared surface-geodesic bins."""
    fig, ax = cf.get_fig_ax()
    for category, marker, linestyle in [("existing", "o", "-"), ("non-link", "s", "--")]:
        table = binned[binned["link_category"] == category]
        x = table["geodesic_distance_mean_mm"].to_numpy()
        _plot_finite_segments(
            ax, x, table["rmae_euclidean_mean"].to_numpy(), f"Euclidean {category}",
            color=cf.BLUE, marker=marker, linestyle=linestyle,
        )
        _plot_finite_segments(
            ax, x, table["rmae_geodesic_mean"].to_numpy(), f"Geodesic {category}",
            color=cf.ORANGE, marker=marker, linestyle=linestyle,
        )
    cf.set_ax(
        ax, xlabel="Mean surface geodesic distance (mm)", ylabel="RMAE",
        legend_loc="lower center", bbox_to_anchor=(0.5, 1.0), legend_kwargs={"ncols": 2},
    )
    cf.save_fig(fig, str(output_path), formats=["png", "pdf"], pkl=False)


def plot_human_distance_lambda_control(summary, output_path):
    """Plot correlation and category RMAE for the four controlled conditions."""
    lambda_order = ["Euclidean-optimal lambda", "Geodesic-optimal lambda"]
    x = np.arange(4, dtype=np.float64)
    xtick_label = ["λE", "λG", "λE", "λG"]
    fig, axes = cf.get_fig_ax(ncols=3)
    panels = [
        ("correlation_mean", "correlation_std", "Pearson correlation"),
        ("rmae_existing_mean", "rmae_existing_std", "RMAE (existing links)"),
        ("rmae_non_link_mean", "rmae_non_link_std", "RMAE (non-links)"),
    ]
    ordered = summary.set_index(["distance_metric", "lambda_source"])
    for ax, (mean_column, std_column, ylabel) in zip(axes, panels):
        euclidean = ordered.loc[[("euclidean", label) for label in lambda_order]]
        geodesic = ordered.loc[[("geodesic", label) for label in lambda_order]]
        cf.plt_bar(
            ax, x[:2], euclidean[mean_column].to_numpy(), label="Euclidean distance",
            color=cf.BLUE, err=euclidean[std_column].to_numpy(),
        )
        cf.plt_bar(
            ax, x[2:], geodesic[mean_column].to_numpy(), label="Geodesic distance",
            color=cf.ORANGE, err=geodesic[std_column].to_numpy(),
        )
        cf.set_ax(ax, xlabel="", ylabel=ylabel, xtick=x, xtick_label=xtick_label, legend=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2)
    cf.save_fig(fig, str(output_path), formats=["png", "pdf"], pkl=False)


def plot_human_distance_lambda_control_binned_rmae(binned, output_path):
    """Plot the four human distance-by-lambda conditions on shared geodesic bins."""
    lambda_order = ["Euclidean-optimal lambda", "Geodesic-optimal lambda"]
    styles = {
        ("euclidean", lambda_order[0]): (cf.BLUE, "o", "-", "Euclidean, λE"),
        ("euclidean", lambda_order[1]): (cf.BLUE, "s", "--", "Euclidean, λG"),
        ("geodesic", lambda_order[0]): (cf.ORANGE, "o", "-", "Geodesic, λE"),
        ("geodesic", lambda_order[1]): (cf.ORANGE, "s", "--", "Geodesic, λG"),
    }
    fig, axes = cf.get_fig_ax(ncols=2)
    for ax, category, title in zip(axes, ["existing", "non-link"], ["Existing links", "Non-links"]):
        category_table = binned[binned["link_category"] == category]
        for condition, (color, marker, linestyle, label) in styles.items():
            distance_metric, lambda_source = condition
            table = category_table[
                (category_table["distance_metric"] == distance_metric)
                & (category_table["lambda_source"] == lambda_source)
            ].sort_values("bin")
            _plot_finite_segments(
                ax, table["geodesic_distance_mean_mm"].to_numpy(), table["rmae_mean"].to_numpy(),
                label, color=color, marker=marker, linestyle=linestyle,
            )
        cf.set_ax(
            ax, xlabel="Mean surface geodesic distance (mm)", ylabel="RMAE",
            title=title, legend=False,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4)
    cf.save_fig(fig, str(output_path), formats=["png", "pdf"], pkl=False)


def plot_human_center_distance_similarity(similarity, output_path):
    """Plot annotated correlations among human center-distance matrices."""
    display = similarity.copy()
    display_labels = ["2D", "3D\nEuclidean", "3D\ngeodesic"]
    display.index = display_labels
    display.columns = display_labels
    fig, ax = cf.get_fig_ax()
    cf.sns_heatmap(
        ax, display, cmap=cf.get_cmap([cf.WHITE, cf.RED]), vmin=0, vmax=1,
        xtick_rotation=0, ytick_rotation=0,
        heatmap_kwargs={"annot": True, "fmt": ".2f", "annot_kws": {"fontsize": cf.TICK_SIZE}},
    )
    cf.set_ax(ax, xlabel="", ylabel="")
    cf.save_fig(fig, str(output_path), formats=["png", "pdf"], pkl=False)


def plot_distance_distribution_validation(validation, output_path):
    """Plot sampled and analytical EDR probability per native distance class."""
    fig, axes = cf.get_fig_ax(nrows=2, ncols=2)
    dataset_order = ["mouse_gamanut", "marmoset_majka", "macaque_markov40", "human_lariviere"]
    for ax, dataset_name in zip(axes.ravel(), dataset_order):
        local = validation[validation["dataset_name"] == dataset_name]
        x = local["distance_mm"].to_numpy()
        cf.plt_line(ax, x, local["sampled_probability_per_class_mean"].to_numpy(), label="Sampled", color=cf.BLUE, marker="o")
        cf.plt_line(ax, x, local["target_probability_per_class"].to_numpy(), label="Target EDR", color=cf.ORANGE)
        title = dataset_name.split("_", 1)[0].capitalize()
        cf.set_ax(ax, xlabel="3D Euclidean distance (mm)", ylabel="Probability per distance class", title=title, ylog=True, legend=False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2)
    cf.save_fig(fig, str(output_path), formats=["png", "pdf"], pkl=False)


def plot_human_shortcut(output_path):
    """Plot one real human surface geodesic and its Euclidean shortcut."""
    cache_dir = PREPROCESSING_DIR / "human_lariviere" / "shortcut"
    vertices = np.load(cache_dir / "surface_vertices.npy")
    faces = np.load(cache_dir / "surface_faces.npy")
    path = np.load(cache_dir / "shortcut_path_indices.npy")
    path_coordinates = vertices[path]
    local_vertices = scipy.spatial.cKDTree(path_coordinates).query(vertices, k=1)[0] <= 18.0
    local_faces = faces[np.all(local_vertices[faces], axis=1)]
    used = np.unique(local_faces)
    global_to_local = np.full(len(vertices), -1, dtype=np.int32)
    global_to_local[used] = np.arange(len(used), dtype=np.int32)
    local_faces = global_to_local[local_faces]
    edge_start = np.concatenate((local_faces[:, 0], local_faces[:, 1], local_faces[:, 2]))
    edge_stop = np.concatenate((local_faces[:, 1], local_faces[:, 2], local_faces[:, 0]))
    adjacency = scipy.sparse.coo_matrix((np.ones(len(edge_start)), (edge_start, edge_stop)), shape=(len(used), len(used)))
    _, component_labels = scipy.sparse.csgraph.connected_components(adjacency, directed=False)
    path_local = global_to_local[path]
    path_local = path_local[path_local >= 0]
    main_component = np.argmax(np.bincount(component_labels[path_local]))
    component_vertices = component_labels == main_component
    local_faces = local_faces[np.all(component_vertices[local_faces], axis=1)]
    component_used = used[component_vertices]
    component_remap = np.full(len(used), -1, dtype=np.int32)
    component_remap[np.flatnonzero(component_vertices)] = np.arange(len(component_used), dtype=np.int32)
    local_faces = component_remap[local_faces]
    used = component_used
    center = np.mean(vertices[used], axis=0)
    _, _, rotation = np.linalg.svd(vertices[used] - center, full_matrices=False)
    transformed = (vertices - center) @ rotation.T
    fig, ax = cf.get_fig_ax(subplots_params={"projection": "3d"})
    local_coordinates = transformed[used]
    ax.plot_trisurf(local_coordinates[:, 0], local_coordinates[:, 1], local_coordinates[:, 2], triangles=local_faces, color=cf.GRAY, alpha=0.35)
    transformed_path = transformed[path]
    ax.plot(transformed_path[:, 0], transformed_path[:, 1], transformed_path[:, 2], label="Geodesic", color=cf.BLUE)
    endpoints = transformed[path[[0, -1]]]
    ax.plot(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], label="Euclidean shortcut", color=cf.ORANGE)
    ax.scatter(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], color=cf.BLACK)
    cf.set_ax(ax, legend_loc="upper left", elev=20, azim=-60)
    ax.set_axis_off()
    cf.save_fig(fig, str(output_path), formats=["png", "pdf"], pkl=False)


def main():
    """Create the final cross-species 3D figures and numerical tables."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    correlations = pd.read_csv(MODELS_DIR / "optimal_seed_correlations.csv")
    correlations["dimension"] = correlations["dimension"].fillna("2D")
    correlations.to_csv(ANALYSIS_DIR / "optimal_seed_correlations.csv", index=False)
    plot_correlation_bar(correlations, FIGURES_DIR / "correlation_2d_3d_bar")
    seed_rmae_tables = []
    for animal, dataset_key, pair_loader in [("macaque", "markov40", get_macaque_pair_table), ("human", "lariviere", get_human_pair_table)]:
        data, model_2d, model_3d = load_seed_comparison_matrices(animal, dataset_key)
        dataset_name = f"{animal}_{dataset_key}"
        pair_table = pair_loader()
        for seed in model_2d:
            local = get_ratio_binned_rmae(
                pair_table, data, model_2d[seed], model_3d[seed], RMAE_BIN_COUNT[dataset_name], animal, dataset_key
            )
            local["seed"] = seed
            seed_rmae_tables.append(local)
    seed_binned = pd.concat(seed_rmae_tables, ignore_index=True)
    seed_binned.to_csv(ANALYSIS_DIR / "geodesic_ratio_rmae_by_seed.csv", index=False)
    binned = get_seed_rmae_summary(seed_binned)
    binned.to_csv(ANALYSIS_DIR / "geodesic_ratio_rmae.csv", index=False)
    plot_rmae(binned, FIGURES_DIR / "geodesic_ratio_rmae")
    human_pairs = get_human_pair_table()
    maximum_geodesic = float(human_pairs["geodesic_distance_mm"].max())
    geodesic_edges = np.arange(
        0.0,
        np.ceil(maximum_geodesic / HUMAN_GEODESIC_RMAE_BIN_WIDTH_MM) * HUMAN_GEODESIC_RMAE_BIN_WIDTH_MM
        + HUMAN_GEODESIC_RMAE_BIN_WIDTH_MM,
        HUMAN_GEODESIC_RMAE_BIN_WIDTH_MM,
    )
    euclidean_models = load_distance_metric_seed_matrices("human_lariviere", "euclidean")
    geodesic_models = load_distance_metric_seed_matrices("human_lariviere", "geodesic")
    human_data = uf.AnimalInfo("human", "lariviere").SC_data
    geodesic_seed_tables = []
    for seed in range(5):
        local = get_geodesic_distance_binned_rmae(
            human_pairs, human_data, euclidean_models[seed], geodesic_models[seed], geodesic_edges
        )
        local["seed"] = seed
        geodesic_seed_tables.append(local)
    geodesic_seed_rmae = pd.concat(geodesic_seed_tables, ignore_index=True)
    geodesic_seed_rmae.to_csv(ANALYSIS_DIR / "human_geodesic_distance_rmae_by_seed.csv", index=False)
    geodesic_rmae = get_geodesic_distance_rmae_summary(geodesic_seed_rmae)
    geodesic_rmae.to_csv(ANALYSIS_DIR / "human_geodesic_distance_rmae.csv", index=False)
    plot_human_geodesic_distance_rmae(
        geodesic_rmae, FIGURES_DIR / "human_geodesic_distance_rmae"
    )
    control_seed, control_summary = get_human_distance_lambda_control_summary()
    control_seed.to_csv(ANALYSIS_DIR / "human_distance_lambda_control_by_seed.csv", index=False)
    control_summary.to_csv(ANALYSIS_DIR / "human_distance_lambda_control.csv", index=False)
    plot_human_distance_lambda_control(
        control_summary, FIGURES_DIR / "human_distance_lambda_control"
    )
    control_binned_seed = get_human_distance_lambda_control_binned_rmae(
        human_pairs, human_data, control_seed, geodesic_edges
    )
    control_binned_seed.to_csv(
        ANALYSIS_DIR / "human_distance_lambda_control_binned_rmae_by_seed.csv", index=False
    )
    control_binned = get_human_distance_lambda_control_binned_summary(control_binned_seed)
    control_binned.to_csv(
        ANALYSIS_DIR / "human_distance_lambda_control_binned_rmae.csv", index=False
    )
    plot_human_distance_lambda_control_binned_rmae(
        control_binned, FIGURES_DIR / "human_distance_lambda_control_binned_rmae"
    )
    center_indices, center_distances = get_human_center_distance_matrices()
    np.save(ANALYSIS_DIR / "human_center_vertex_indices.npy", center_indices, allow_pickle=False)
    for label, matrix in center_distances.items():
        filename = label.lower().replace(" ", "_") + "_center_distances.csv"
        matrix.to_csv(ANALYSIS_DIR / filename)
    distance_similarity = get_distance_matrix_similarity(center_distances)
    distance_similarity.to_csv(ANALYSIS_DIR / "human_center_distance_similarity.csv")
    plot_human_center_distance_similarity(
        distance_similarity, FIGURES_DIR / "human_center_distance_similarity"
    )
    distance_validation = pd.concat([
        get_distance_distribution_validation(f"{animal}_{dataset_key}")
        for animal, dataset_key in [("mouse", "gamanut"), ("marmoset", "majka"), ("macaque", "markov40"), ("human", "lariviere")]
    ], ignore_index=True)
    distance_validation.to_csv(ANALYSIS_DIR / "connection_distance_distribution_validation.csv", index=False)
    plot_distance_distribution_validation(
        distance_validation, FIGURES_DIR / "connection_distance_distribution_validation"
    )
    plot_human_shortcut(FIGURES_DIR / "human_euclidean_shortcut")
    metric_path = MODELS_DIR / "distance_metric_comparison" / "optimal_seed_correlations.csv"
    if not metric_path.exists():
        raise FileNotFoundError(f"Run model.py first. Missing: {metric_path}")
    metric_correlations = pd.read_csv(metric_path)
    metric_correlations.to_csv(ANALYSIS_DIR / "distance_metric_optimal_seed_correlations.csv", index=False)
    plot_distance_metric_correlation_bar(
        metric_correlations, FIGURES_DIR / "euclidean_geodesic_correlation_bar"
    )


if __name__ == "__main__":
    main()
