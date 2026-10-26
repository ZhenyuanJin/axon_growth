import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.special
import shapely
import shapely.geometry
from shapely.strtree import STRtree

from utils_function import cf
from utils_function import utils_function as uf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "results" / "experiments" / "marmoset_target_conditioned_angle_edr"
ANIMAL = "marmoset"
DATASET_KEY = "majka"
BASE_SEED = 20261026
NA_PER_TARGET = 20_000
LAMBDA_VALUES = np.arange(0.75, 3.501, 0.125)
ANGLE_KDE_BANDWIDTH_DEGREES = 15.0
MAX_ITERATION = 10
DIRECTION_METHOD_LIST = ["random", "SC_angle_kde"]


def get_stage_seed(the_seed, *identity):
    """Return a deterministic uint64-derived seed from a base seed and task identity."""
    text = "|".join([str(the_seed)] + [str(item) for item in identity])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def get_points_in_polygon(polygon, point_count, seed):
    """Return uniformly sampled float64 points inside one polygon, shape (point_count, 2)."""
    if point_count < 0:
        raise ValueError("point_count must be non-negative.")
    if point_count == 0:
        return np.empty((0, 2), dtype=float)
    rng = np.random.default_rng(seed)
    min_x, min_y, max_x, max_y = polygon.bounds
    points = np.empty((point_count, 2), dtype=float)
    filled = 0
    acceptance_estimate = polygon.area / ((max_x - min_x) * (max_y - min_y))
    if not 0 < acceptance_estimate <= 1:
        raise ValueError("Polygon has an invalid area-to-bounds ratio.")

    while filled < point_count:
        remaining = point_count - filled
        batch_size = max(64, int(np.ceil(remaining / acceptance_estimate * 1.1)))
        x = rng.uniform(min_x, max_x, batch_size)
        y = rng.uniform(min_y, max_y, batch_size)
        inside = shapely.contains_xy(polygon, x, y)
        accepted = np.column_stack([x[inside], y[inside]])[:remaining]
        points[filled:filled + len(accepted)] = accepted
        filled += len(accepted)
    return points


def get_target_points(target_name, point_count, animal_info, polygon_dict, seed):
    """Return area-weighted uniform target points, shape (point_count, 2)."""
    split_names = animal_info.name_to_split_name_dict[target_name]
    split_areas = np.array([polygon_dict[name].area for name in split_names], dtype=float)
    probabilities = split_areas / np.sum(split_areas)
    rng = np.random.default_rng(seed)
    split_indices = rng.choice(len(split_names), size=point_count, p=probabilities)
    points = np.empty((point_count, 2), dtype=float)
    for split_index, split_name in enumerate(split_names):
        mask = split_indices == split_index
        split_seed = get_stage_seed(seed, "split", split_name)
        points[mask] = get_points_in_polygon(polygon_dict[split_name], np.sum(mask), split_seed)
    return points


def get_source_area_lookup(animal_info, polygon_dict):
    """Return a polygon tree and split-polygon to source-index lookup."""
    split_names = list(polygon_dict)
    polygons = np.array([polygon_dict[name] for name in split_names], dtype=object)
    source_index = {name: index for index, name in enumerate(animal_info.source_name)}
    split_to_source_index = np.array(
        [source_index[animal_info.split_name_to_name_dict[name]] for name in split_names],
        dtype=int,
    )
    return STRtree(polygons), split_to_source_index


def get_source_indices(points, polygon_tree, split_to_source_index):
    """Map float64 points, shape (n, 2), to source indices; unmatched points are -1."""
    point_geometries = shapely.points(points)
    matches = polygon_tree.query(point_geometries, predicate="within")
    source_indices = np.full(len(points), -1, dtype=int)
    if matches.shape[1] == 0:
        return source_indices
    point_indices, polygon_indices = matches
    unique_points, counts = np.unique(point_indices, return_counts=True)
    if np.any(counts > 1):
        duplicated = unique_points[counts > 1]
        raise ValueError(f"Generated points matched multiple cortical areas: {duplicated[:10].tolist()}")
    source_indices[point_indices] = split_to_source_index[polygon_indices]
    return source_indices


def get_area_components(animal_info, polygon_dict):
    """Return float64 centroids (n, 2) and area fractions (n,) for each area's components."""
    components = {}
    for name in animal_info.full_name:
        polygons = [polygon_dict[split] for split in animal_info.name_to_split_name_dict[name]]
        areas = np.array([polygon.area for polygon in polygons], dtype=float)
        centers = np.array([[polygon.centroid.x, polygon.centroid.y] for polygon in polygons])
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(areas)) or np.any(areas <= 0):
            raise ValueError(f"Invalid components for area {name}.")
        fractions = np.ones(1) if len(polygons) == 1 else areas / areas.sum()
        components[name] = (centers, fractions)
    return components


def get_angle_centers_and_weights(target_name, animal_info, SC_data, area_components):
    """Return component-pair angles and normalized FLNe times source/target area fractions."""
    target_centers, target_fractions = area_components[target_name]
    angles, weights = [], []
    for source_name in animal_info.source_name:
        weight = float(SC_data.loc[source_name, target_name])
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid FLNe angle weight: {source_name}, {target_name}.")
        if source_name == target_name or weight == 0:
            continue
        source_centers, source_fractions = area_components[source_name]
        displacement = source_centers[:, None, :] - target_centers[None, :, :]
        angles.extend(np.arctan2(displacement[..., 1], displacement[..., 0]).ravel())
        weights.extend((weight * source_fractions[:, None] * target_fractions[None, :]).ravel())
    angles, weights = np.asarray(angles), np.asarray(weights)
    if not np.all(np.isfinite(angles)) or weights.sum() <= 0:
        raise ValueError(f"Invalid FLNe angle weights for target {target_name}.")
    return angles, weights / np.sum(weights)


def get_angles(direction_method, point_count, angle_centers, angle_weights, angle_kde_bandwidth_degrees, seed):
    """Return deterministic random or SC-kernel angles, shape (point_count,)."""
    rng = np.random.default_rng(seed)
    if direction_method == "random":
        return rng.uniform(-np.pi, np.pi, point_count)
    if direction_method == "SC_angle_kde":
        bandwidth_radians = np.radians(angle_kde_bandwidth_degrees)
        if not 0 < bandwidth_radians < np.pi:
            raise ValueError("angle_kde_bandwidth_degrees must be between zero and 180.")
        concentration = 1.0 / bandwidth_radians ** 2
        component_indices = rng.choice(len(angle_centers), size=point_count, p=angle_weights)
        return rng.vonmises(angle_centers[component_indices], concentration)
    raise ValueError(f"Unknown direction_method: {direction_method}")


def get_target_connection_counts(target_name, target_points, Lambda, direction_method, angle_kde_bandwidth_degrees, max_iteration, the_seed, animal_info, angle_data_by_target, polygon_tree, split_to_source_index):
    """Generate one target's source counts and return counts plus acceptance statistics."""
    source_names = animal_info.source_name
    target_source_index = source_names.index(target_name)
    angle_centers, angle_weights = angle_data_by_target[target_name]
    counts = np.zeros(len(source_names), dtype=np.int64)
    unresolved = np.arange(len(target_points))
    lambda_key = f"{Lambda:.6f}"

    for iteration in range(max_iteration):
        if len(unresolved) == 0:
            break
        distance_seed = get_stage_seed(the_seed, target_name, lambda_key, iteration, "distance")
        angle_seed = get_stage_seed(the_seed, target_name, lambda_key, iteration, direction_method, "angle")
        distance_rng = np.random.default_rng(distance_seed)
        distances = distance_rng.exponential(Lambda, len(unresolved))
        angles = get_angles(
            direction_method, len(unresolved), angle_centers, angle_weights,
            angle_kde_bandwidth_degrees, angle_seed,
        )
        source_points = target_points[unresolved] + distances[:, None] * np.column_stack(
            [np.cos(angles), np.sin(angles)]
        )
        source_indices = get_source_indices(source_points, polygon_tree, split_to_source_index)
        valid = (source_indices >= 0) & (source_indices != target_source_index)
        counts += np.bincount(source_indices[valid], minlength=len(source_names))
        unresolved = unresolved[~valid]

    accepted_count = int(np.sum(counts))
    return counts, accepted_count / len(target_points), len(unresolved)


def get_SC_model(Lambda, direction_method, target_points_by_name, animal_info, angle_data_by_target, polygon_tree, split_to_source_index, na_per_target, angle_kde_bandwidth_degrees, max_iteration, the_seed):
    """Return a generated FLNe matrix and target-level acceptance table."""
    SC_model = pd.DataFrame(
        0.0, index=animal_info.source_name, columns=animal_info.target_name,
    )
    acceptance_rows = []
    for target_name in animal_info.target_name:
        counts, acceptance_rate, unresolved_count = get_target_connection_counts(
            target_name, target_points_by_name[target_name], Lambda, direction_method,
            angle_kde_bandwidth_degrees, max_iteration, the_seed, animal_info,
            angle_data_by_target, polygon_tree, split_to_source_index,
        )
        if np.sum(counts) <= 0:
            raise ValueError(f"No valid generated source points for target {target_name}.")
        SC_model.loc[:, target_name] = counts / np.sum(counts)
        acceptance_rows.append(
            {
                "target": target_name,
                "acceptance_rate": acceptance_rate,
                "unresolved_count": unresolved_count,
                "accepted_count": int(np.sum(counts)),
                "requested_count": na_per_target,
            }
        )
    return SC_model, pd.DataFrame(acceptance_rows)


def get_performance(SC_data, SC_model, animal_info):
    """Return global and target-level performance for a generated FLNe matrix."""
    correlation = uf.data_model_correlation(
        SC_data, SC_model, ANIMAL, DATASET_KEY, diagonal=False,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log", category=RuntimeWarning)
        logged_data = uf.log_SC(SC_data, ANIMAL, DATASET_KEY, apply_c=True, apply_w=True)
        logged_model = uf.log_SC(SC_model, ANIMAL, DATASET_KEY, apply_c=True, apply_w=True)
        relative_error = uf.data_model_relative_error(
            SC_data, SC_model, ANIMAL, DATASET_KEY, diagonal=False,
        )
    log_correlation = uf.data_model_correlation(
        logged_data, logged_model, ANIMAL, DATASET_KEY, diagonal=False,
    )
    target_correlations = uf.area_level_correlation(
        SC_data, SC_model, ANIMAL, DATASET_KEY, diagonal=False,
    )
    finite_target_correlations = np.array(
        [value for value in target_correlations.values() if np.isfinite(value)]
    )
    return {
        "correlation": correlation,
        "log_correlation": log_correlation,
        "rmae_both": np.nanmean(relative_error.to_numpy()),
        "median_target_correlation": np.median(finite_target_correlations),
        "target_correlations": target_correlations,
    }


def run_scan(lambda_values, na_per_target, angle_kde_bandwidth_degrees, max_iteration, the_seed):
    """Run the deterministic Lambda scan and return metrics, matrices, and target results."""
    animal_info = uf.AnimalInfo(ANIMAL, DATASET_KEY)
    SC_data = animal_info.SC_data.loc[animal_info.source_name, animal_info.target_name]
    polygon_dict = animal_info.polygon_dict
    area_components = get_area_components(animal_info, polygon_dict)
    polygon_tree, split_to_source_index = get_source_area_lookup(animal_info, polygon_dict)
    angle_data_by_target = {
        target_name: get_angle_centers_and_weights(
            target_name, animal_info, SC_data, area_components,
        )
        for target_name in animal_info.target_name
    }
    target_points_by_name = {
        target_name: get_target_points(
            target_name, na_per_target, animal_info, polygon_dict,
            get_stage_seed(the_seed, target_name, "target_points"),
        )
        for target_name in animal_info.target_name
    }
    scan_rows = []
    SC_model_by_task = {}
    target_rows = []

    for Lambda in lambda_values:
        for direction_method in DIRECTION_METHOD_LIST:
            SC_model, acceptance = get_SC_model(
                Lambda, direction_method, target_points_by_name, animal_info,
                angle_data_by_target, polygon_tree, split_to_source_index,
                na_per_target, angle_kde_bandwidth_degrees, max_iteration, the_seed,
            )
            performance = get_performance(SC_data, SC_model, animal_info)
            key = (direction_method, float(Lambda))
            SC_model_by_task[key] = SC_model
            scan_rows.append(
                {
                    "direction_method": direction_method,
                    "Lambda": float(Lambda),
                    "correlation": performance["correlation"],
                    "log_correlation": performance["log_correlation"],
                    "rmae_both": performance["rmae_both"],
                    "median_target_correlation": performance["median_target_correlation"],
                    "mean_acceptance_rate": acceptance["acceptance_rate"].mean(),
                    "minimum_acceptance_rate": acceptance["acceptance_rate"].min(),
                }
            )
            for target_name, target_correlation in performance["target_correlations"].items():
                target_acceptance = acceptance.loc[acceptance["target"] == target_name].iloc[0]
                target_rows.append(
                    {
                        "direction_method": direction_method,
                        "Lambda": float(Lambda),
                        "target": target_name,
                        "correlation": target_correlation,
                        "acceptance_rate": target_acceptance["acceptance_rate"],
                        "unresolved_count": int(target_acceptance["unresolved_count"]),
                    }
                )
            print(
                f"{direction_method} Lambda={Lambda:.3f} "
                f"corr={performance['correlation']:.6f} "
                f"log_corr={performance['log_correlation']:.6f} "
                f"acceptance={acceptance['acceptance_rate'].mean():.4f}"
            )
    return pd.DataFrame(scan_rows), SC_model_by_task, pd.DataFrame(target_rows), animal_info


def get_optimal_rows(scan_metrics):
    """Return the maximum-correlation row for each model."""
    indices = scan_metrics.groupby("direction_method")["correlation"].idxmax()
    return scan_metrics.loc[indices].sort_values("direction_method").reset_index(drop=True)


def get_metric_optima(scan_metrics):
    """Return the best Lambda and value for each fit metric and direction method."""
    objectives = {
        "correlation": "max",
        "log_correlation": "max",
        "rmae_both": "min",
        "median_target_correlation": "max",
    }
    rows = []
    for direction_method in DIRECTION_METHOD_LIST:
        subset = scan_metrics.loc[scan_metrics["direction_method"] == direction_method]
        for metric, objective in objectives.items():
            index = subset[metric].idxmax() if objective == "max" else subset[metric].idxmin()
            best = subset.loc[index]
            rows.append(
                {
                    "direction_method": direction_method,
                    "metric": metric,
                    "objective": objective,
                    "Lambda": best["Lambda"],
                    "value": best[metric],
                }
            )
    return pd.DataFrame(rows)


def get_angle_density(angle_grid, angle_centers, angle_weights, angle_kde_bandwidth_degrees):
    """Return the normalized von Mises mixture density on a radian grid."""
    bandwidth_radians = np.radians(angle_kde_bandwidth_degrees)
    concentration = 1.0 / bandwidth_radians ** 2
    kernel = np.exp(concentration * np.cos(angle_grid[:, None] - angle_centers[None, :]))
    kernel /= 2 * np.pi * scipy.special.i0(concentration)
    density = kernel @ angle_weights
    return density / np.max(density)


def plot_method_overview(SC_data, animal_info, output_dir):
    """Plot the empirical weighted angles and circular KDE for one target."""
    target_name = "TPO"
    area_components = get_area_components(animal_info, animal_info.polygon_dict)
    angle_centers, angle_weights = get_angle_centers_and_weights(
        target_name, animal_info, SC_data, area_components,
    )
    angle_grid = np.linspace(-np.pi, np.pi, 361)
    density = get_angle_density(
        angle_grid, angle_centers, angle_weights, ANGLE_KDE_BANDWIDTH_DEGREES,
    )
    normalized_weights = angle_weights / np.max(angle_weights)
    positive = angle_weights > 0
    fig, ax = cf.get_fig_ax()
    cf.plt_scatter(
        ax, np.degrees(angle_centers[positive]), normalized_weights[positive],
        color=cf.BLUE, label="FLNe-weighted",
    )
    cf.plt_line(
        ax, np.degrees(angle_grid), density,
        color=cf.RED, label="KDE",
    )
    cf.set_ax(
        ax, xlabel="Target-to-source angle (degree)",
        ylabel="Normalized weight / density", title=f"{target_name} angle distribution",
        legend_loc="upper left",
    )
    cf.save_fig(fig, str(output_dir / "method_overview"), formats=["png", "pdf"], pkl=False)


def plot_correlation_improvement(scan_metrics, optimal_rows, target_comparison, output_dir):
    """Plot overall and target-level correlation improvements."""
    fig, axes = cf.get_fig_ax(ncols=2)
    colors = {"random": cf.GRAY, "SC_angle_kde": cf.RED}
    labels = {"random": "random", "SC_angle_kde": "SC angle KDE"}
    for direction_method in DIRECTION_METHOD_LIST:
        subset = scan_metrics.loc[
            scan_metrics["direction_method"] == direction_method
        ].sort_values("Lambda")
        cf.plt_line(
            axes[0], subset["Lambda"].to_numpy(), subset["correlation"].to_numpy(),
            color=colors[direction_method],
        )
        best = optimal_rows.loc[
            optimal_rows["direction_method"] == direction_method
        ].iloc[0]
        cf.plt_scatter(
            axes[0], best["Lambda"], best["correlation"],
            color=colors[direction_method], s=(2 * cf.MARKER_SIZE) ** 2,
        )
        final_row = subset.iloc[-1]
        cf.add_text(
            axes[0], labels[direction_method], x=final_row["Lambda"] + 0.08,
            y=final_row["correlation"], transform="data", color=colors[direction_method],
        )
    correlation_gain = (
        optimal_rows.loc[optimal_rows["direction_method"] == "SC_angle_kde", "correlation"].iloc[0]
        - optimal_rows.loc[optimal_rows["direction_method"] == "random", "correlation"].iloc[0]
    )
    cf.add_text(
        axes[0], f"Correlation gain = {correlation_gain:.3f}",
        x=2.35, y=0.805, transform="data",
    )
    cf.set_ax(
        axes[0], xlabel="Lambda", ylabel="FLNe correlation",
        title="Overall correlation", legend=False,
        xlim=[scan_metrics["Lambda"].min(), scan_metrics["Lambda"].max() + 0.65],
    )

    finite = target_comparison[["random_correlation", "SC_angle_kde_correlation"]].to_numpy()
    finite = finite[np.all(np.isfinite(finite), axis=1)]
    lower = np.min(finite) - 0.05
    upper = np.max(finite) + 0.08
    cf.plt_line(axes[1], [lower, upper], [lower, upper], color=cf.GRAY, linestyle="--")
    cf.plt_scatter(
        axes[1], target_comparison["random_correlation"].to_numpy(),
        target_comparison["SC_angle_kde_correlation"].to_numpy(), color=cf.BLUE,
    )
    label_positions = {
        "PE": (0.29, 0.78),
        "AuCPB": (0.35, 0.93),
        "A19M": (0.37, 0.83),
        "TPO": (0.49, 0.69),
        "MST": (0.40, 0.34),
        "A9": (0.64, 1.02),
        "A6DR": (0.80, 0.62),
    }
    selected_names = ["PE", "AuCPB", "A19M", "TPO", "MST", "A9", "A6DR"]
    comparison_by_target = target_comparison.set_index("target")
    for target_name in selected_names:
        row = comparison_by_target.loc[target_name]
        x, y = label_positions[target_name]
        cf.add_annotation(
            axes[1], f"{target_name} {row['improvement']:+.2f}",
            xy=(row["random_correlation"], row["SC_angle_kde_correlation"]),
            xytext=(x, y), arrowprops={"fc": cf.GRAY, "ec": cf.GRAY},
        )
    cf.set_ax(
        axes[1], xlabel="random target correlation",
        ylabel="SC-angle-KDE target correlation",
        title="Target-level correlation", legend=False,
        xlim=[lower, upper], ylim=[lower, upper],
    )
    cf.save_fig(fig, str(output_dir / "correlation_improvement"), formats=["png", "pdf"], pkl=False)


def get_target_comparison(optimal_rows, target_metrics):
    """Return target-level correlations at each direction method's optimal Lambda."""
    random_Lambda = optimal_rows.loc[
        optimal_rows["direction_method"] == "random", "Lambda"
    ].iloc[0]
    SC_angle_kde_Lambda = optimal_rows.loc[
        optimal_rows["direction_method"] == "SC_angle_kde", "Lambda"
    ].iloc[0]
    random = target_metrics.loc[
        (target_metrics["direction_method"] == "random")
        & (target_metrics["Lambda"] == random_Lambda),
        ["target", "correlation"],
    ].rename(columns={"correlation": "random_correlation"})
    SC_angle_kde = target_metrics.loc[
        (target_metrics["direction_method"] == "SC_angle_kde")
        & (target_metrics["Lambda"] == SC_angle_kde_Lambda),
        ["target", "correlation"],
    ].rename(columns={"correlation": "SC_angle_kde_correlation"})
    comparison = random.merge(SC_angle_kde, on="target", validate="one_to_one")
    comparison["improvement"] = comparison["SC_angle_kde_correlation"] - comparison["random_correlation"]
    return comparison.sort_values("improvement", ascending=False).reset_index(drop=True)


def save_results(scan_metrics, SC_model_by_task, target_metrics, optimal_rows, metric_optima, target_comparison, output_dir, na_per_target, angle_kde_bandwidth_degrees, max_iteration, the_seed):
    """Save scan tables, optimal matrices, target comparisons, and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_metrics.to_csv(output_dir / "lambda_scan_metrics.csv", index=False)
    target_metrics.to_csv(output_dir / "target_metrics.csv", index=False)
    optimal_rows.to_csv(output_dir / "optimal_models.csv", index=False)
    metric_optima.to_csv(output_dir / "metric_optima.csv", index=False)
    target_comparison.to_csv(output_dir / "optimal_target_comparison.csv", index=False)
    for _, row in optimal_rows.iterrows():
        SC_model_by_task[(row["direction_method"], row["Lambda"])].to_csv(
            output_dir / f"optimal_{row['direction_method']}_FLNe.csv"
        )
    metadata = {
        "animal": ANIMAL,
        "dataset_key": DATASET_KEY,
        "the_seed": the_seed,
        "na_per_target": na_per_target,
        "Lambda_values": [float(value) for value in LAMBDA_VALUES],
        "direction_method_list": DIRECTION_METHOD_LIST,
        "length_method": "exponential",
        "length_params": {"Lambda": "scanned"},
        "angle_kde_bandwidth_degrees": angle_kde_bandwidth_degrees,
        "angle_component_weighting": "FLNe * source_component_area_fraction * target_component_area_fraction",
        "von_mises_concentration": float(1.0 / np.radians(angle_kde_bandwidth_degrees) ** 2),
        "max_iteration": max_iteration,
        "self_connections_excluded": True,
        "selection_metric": "correlation",
        "optimal_models": optimal_rows.to_dict(orient="records"),
        "metric_optima": metric_optima.to_dict(orient="records"),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scan_metrics, SC_model_by_task, target_metrics, animal_info = run_scan(
        LAMBDA_VALUES, NA_PER_TARGET, ANGLE_KDE_BANDWIDTH_DEGREES,
        MAX_ITERATION, BASE_SEED,
    )
    optimal_rows = get_optimal_rows(scan_metrics)
    metric_optima = get_metric_optima(scan_metrics)
    target_comparison = get_target_comparison(optimal_rows, target_metrics)
    SC_data = animal_info.SC_data.loc[animal_info.source_name, animal_info.target_name]
    plot_method_overview(SC_data, animal_info, OUTPUT_DIR)
    plot_correlation_improvement(scan_metrics, optimal_rows, target_comparison, OUTPUT_DIR)
    save_results(
        scan_metrics, SC_model_by_task, target_metrics, optimal_rows, metric_optima,
        target_comparison, OUTPUT_DIR, NA_PER_TARGET,
        ANGLE_KDE_BANDWIDTH_DEGREES, MAX_ITERATION, BASE_SEED,
    )
    print("Optimal models:")
    print(optimal_rows.to_string(index=False))
    improved_count = int(np.sum(target_comparison["improvement"] > 0))
    print(f"Targets improved by SC_angle_kde: {improved_count}/{len(target_comparison)}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
