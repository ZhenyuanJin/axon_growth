import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely.geometry

from experiments.midpoint_utils import get_area_midpoint, get_area_size
from utils_function import cf
from utils_function import utils_function as uf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "results" / "experiments" / "marmoset_midpoint_source_bias"
ANIMAL = "marmoset"
DATASET_KEY = "majka"
CANDIDATE_COUNT = 6
INTERIOR_QUANTILE = 0.5
TOP_CONNECTION_COUNT = 12


def get_direction_vector(weights, unit_vectors):
    """Return the weighted mean unit direction, shape (2,), from deterministic inputs."""
    weight_sum = np.sum(weights)
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("Direction weights must have a positive finite sum.")
    return np.sum(weights[:, None] * unit_vectors, axis=0) / weight_sum


def get_target_metrics(animal_info, connection_matrix, lambda_value):
    """Return one deterministic midpoint-based directional-bias row per target."""
    source_names = animal_info.source_name
    source_midpoints = np.array([get_area_midpoint(name, animal_info) for name in source_names])
    source_sizes = np.array([get_area_size(name, animal_info) for name in source_names])
    rows = []

    for target_name in animal_info.target_name:
        target_midpoint = get_area_midpoint(target_name, animal_info)
        displacements = source_midpoints - target_midpoint
        distances = np.linalg.norm(displacements, axis=1)
        unit_vectors = np.divide(
            displacements,
            distances[:, None],
            out=np.zeros_like(displacements),
            where=distances[:, None] > 0,
        )
        self_mask = np.asarray(source_names) == target_name

        empirical_weights = connection_matrix.loc[source_names, target_name].to_numpy(dtype=float).copy()
        empirical_weights[self_mask] = 0.0
        if np.any(empirical_weights < 0) or not np.all(np.isfinite(empirical_weights)):
            raise ValueError(f"Invalid connection weights for target {target_name}.")

        baseline_weights = source_sizes * np.exp(-distances / lambda_value)
        baseline_weights[self_mask] = 0.0
        empirical_vector = get_direction_vector(empirical_weights, unit_vectors)
        baseline_vector = get_direction_vector(baseline_weights, unit_vectors)
        corrected_vector = empirical_vector - baseline_vector
        empirical_resultant = np.linalg.norm(empirical_vector)
        baseline_resultant = np.linalg.norm(baseline_vector)
        corrected_resultant = np.linalg.norm(corrected_vector)
        empirical_heading = np.degrees(np.arctan2(empirical_vector[1], empirical_vector[0]))
        direction_projection = unit_vectors @ empirical_vector
        dominant_half_mask = direction_projection >= 0
        dominant_half_weight = np.sum(empirical_weights[dominant_half_mask]) / np.sum(empirical_weights)
        normalized_weights = empirical_weights / np.sum(empirical_weights)
        positive_weights = normalized_weights[normalized_weights > 0]
        effective_source_count = np.exp(-np.sum(positive_weights * np.log(positive_weights)))
        boundary_clearance = animal_info.whole_cortex_polygon.boundary.distance(
            shapely.geometry.Point(target_midpoint)
        )

        rows.append(
            {
                "target": target_name,
                "midpoint_x": target_midpoint[0],
                "midpoint_y": target_midpoint[1],
                "boundary_clearance": boundary_clearance,
                "empirical_resultant": empirical_resultant,
                "baseline_resultant": baseline_resultant,
                "corrected_resultant": corrected_resultant,
                "empirical_heading_degrees": empirical_heading,
                "dominant_half_weight": dominant_half_weight,
                "positive_source_count": int(np.sum(empirical_weights > 0)),
                "effective_source_count": effective_source_count,
                "strongest_source_weight": np.max(empirical_weights),
            }
        )
    return pd.DataFrame(rows)


def get_candidate_targets(metrics, candidate_count, interior_quantile):
    """Return metrics with interior status, percentile score, rank, and selection."""
    if not 0 <= interior_quantile <= 1:
        raise ValueError("interior_quantile must be between zero and one.")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive.")

    ranked = metrics.copy()
    clearance_threshold = ranked["boundary_clearance"].quantile(interior_quantile)
    ranked["is_interior"] = ranked["boundary_clearance"] >= clearance_threshold
    ranked["empirical_percentile"] = np.nan
    ranked["corrected_percentile"] = np.nan
    ranked["candidate_score"] = np.nan
    interior_mask = ranked["is_interior"]
    ranked.loc[interior_mask, "empirical_percentile"] = ranked.loc[
        interior_mask, "empirical_resultant"
    ].rank(pct=True)
    ranked.loc[interior_mask, "corrected_percentile"] = ranked.loc[
        interior_mask, "corrected_resultant"
    ].rank(pct=True)
    ranked.loc[interior_mask, "candidate_score"] = ranked.loc[
        interior_mask, ["empirical_percentile", "corrected_percentile"]
    ].mean(axis=1)
    ranked["candidate_rank"] = np.nan
    ordered_indices = ranked.loc[interior_mask].sort_values(
        ["candidate_score", "empirical_resultant", "boundary_clearance"],
        ascending=[False, False, False],
    ).index
    ranked.loc[ordered_indices, "candidate_rank"] = np.arange(1, len(ordered_indices) + 1)
    ranked["is_candidate"] = ranked["candidate_rank"] <= candidate_count
    ranked = ranked.sort_values(["is_candidate", "candidate_rank", "target"], ascending=[False, True, True])
    return ranked.reset_index(drop=True), clearance_threshold


def plot_flatmap_background(ax, animal_info):
    """Plot the project flat map as a neutral background on one axes."""
    for coords in animal_info.coords_dict.values():
        cf.plt_polygon(
            ax, coords, color=None, facecolor=(0.94, 0.94, 0.94), edgecolor=cf.WHITE,
            adjust_lim=False,
        )
    whole_coords = animal_info.whole_cortex_coords
    cf.plt_polygon(
        ax, whole_coords, color=None, fill=False, edgecolor=cf.BLACK,
        adjust_lim=False,
    )
    ax.set_xlim(np.min(whole_coords[:, 0]), np.max(whole_coords[:, 0]))
    ax.set_ylim(np.min(whole_coords[:, 1]), np.max(whole_coords[:, 1]))
    ax.set_aspect("equal")
    ax.axis("off")


def plot_overview(metrics, clearance_threshold, animal_info, output_dir):
    """Plot target locations, boundary filtering, and midpoint baseline comparison."""
    fig, axes = cf.get_fig_ax(ncols=3)
    candidate_metrics = metrics.loc[metrics["is_candidate"]].sort_values("candidate_rank")

    plot_flatmap_background(axes[0], animal_info)
    all_targets = metrics[["midpoint_x", "midpoint_y"]].to_numpy()
    cf.plt_scatter(axes[0], all_targets[:, 0], all_targets[:, 1], color=cf.GRAY)
    candidate_points = candidate_metrics[["midpoint_x", "midpoint_y"]].to_numpy()
    cf.plt_scatter(
        axes[0], candidate_points[:, 0], candidate_points[:, 1], color=cf.RED,
    )
    for row in candidate_metrics.itertuples():
        cf.add_text(
            axes[0], row.target, x=row.midpoint_x, y=row.midpoint_y,
            transform="data",
        )
    cf.set_ax(axes[0], title="Selected interior targets", legend=False)

    exterior = metrics.loc[~metrics["is_interior"]]
    interior = metrics.loc[metrics["is_interior"]]
    cf.plt_scatter(
        axes[1], exterior["boundary_clearance"].to_numpy(), exterior["empirical_resultant"].to_numpy(),
        color=cf.GRAY, label="Boundary half",
    )
    cf.plt_scatter(
        axes[1], interior["boundary_clearance"].to_numpy(), interior["empirical_resultant"].to_numpy(),
        color=cf.BLUE, label="Interior half",
    )
    cf.plt_scatter(
        axes[1], candidate_metrics["boundary_clearance"].to_numpy(),
        candidate_metrics["empirical_resultant"].to_numpy(), color=cf.RED,
    )
    cf.add_vline(axes[1], clearance_threshold, color=cf.BLACK, linestyle="--")
    for row in candidate_metrics.itertuples():
        cf.add_text(
            axes[1], row.target, x=row.boundary_clearance, y=row.empirical_resultant,
            transform="data",
        )
    cf.set_ax(
        axes[1], xlabel="Boundary clearance", ylabel="Directional resultant",
        title="Bias vs. boundary clearance",
    )

    maximum = max(metrics["empirical_resultant"].max(), metrics["baseline_resultant"].max()) * 1.05
    cf.plt_line(axes[2], [0, maximum], [0, maximum], color=cf.GRAY, linestyle="--")
    cf.plt_scatter(
        axes[2], interior["baseline_resultant"].to_numpy(), interior["empirical_resultant"].to_numpy(),
        color=cf.BLUE,
    )
    cf.plt_scatter(
        axes[2], candidate_metrics["baseline_resultant"].to_numpy(),
        candidate_metrics["empirical_resultant"].to_numpy(), color=cf.RED,
    )
    for row in candidate_metrics.itertuples():
        cf.add_text(
            axes[2], row.target, x=row.baseline_resultant, y=row.empirical_resultant,
            transform="data",
        )
    cf.set_ax(
        axes[2], xlabel="EDR baseline resultant", ylabel="Empirical resultant",
        title="Empirical vs. midpoint baseline",
        legend=False, xlim=[0, maximum], ylim=[0, maximum],
    )
    cf.save_fig(fig, str(output_dir / "midpoint_bias_overview"), formats=["png", "pdf"], pkl=False)


def plot_candidate_source_patterns(metrics, animal_info, connection_matrix, output_dir):
    """Plot the strongest source connections for each selected target."""
    candidates = metrics.loc[metrics["is_candidate"]].sort_values("candidate_rank")
    fig, axes = cf.get_fig_ax(nrows=2, ncols=3)
    source_names = np.asarray(animal_info.source_name)
    source_midpoints = np.array([get_area_midpoint(name, animal_info) for name in source_names])

    for ax, row in zip(axes.flat, candidates.itertuples()):
        plot_flatmap_background(ax, animal_info)
        weights = connection_matrix.loc[source_names, row.target].to_numpy(dtype=float).copy()
        weights[source_names == row.target] = 0.0
        strongest_indices = np.argsort(weights)[::-1][:TOP_CONNECTION_COUNT]
        strongest_indices = strongest_indices[weights[strongest_indices] > 0]
        selected_points = source_midpoints[strongest_indices]
        target_point = np.array([row.midpoint_x, row.midpoint_y])

        for point in selected_points:
            cf.plt_line(
                ax, [point[0], target_point[0]], [point[1], target_point[1]],
                color=cf.BLUE,
            )
        cf.plt_scatter(
            ax, selected_points[:, 0], selected_points[:, 1], color=cf.BLUE,
        )
        cf.plt_scatter(
            ax, target_point[0], target_point[1], color=cf.RED,
        )
        heading_radians = np.radians(row.empirical_heading_degrees)
        arrow_length = 4.0 * row.empirical_resultant
        cf.add_arrow(
            ax, target_point[0], target_point[1],
            target_point[0] + arrow_length * np.cos(heading_radians),
            target_point[1] + arrow_length * np.sin(heading_radians),
            fc=cf.RED, ec=cf.RED,
        )
        cf.set_ax(
            ax,
            title=f"{int(row.candidate_rank)}. {row.target}  R={row.empirical_resultant:.2f}",
            legend=False,
        )
    cf.save_fig(fig, str(output_dir / "selected_source_patterns"), formats=["png", "pdf"], pkl=False)


def save_results(metrics, clearance_threshold, lambda_value, output_dir):
    """Save the full metric table, selected targets, and experiment metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "area_metrics.csv", index=False)
    candidates = metrics.loc[metrics["is_candidate"]].sort_values("candidate_rank")
    candidates.to_csv(output_dir / "selected_regions.csv", index=False)
    metadata = {
        "animal": ANIMAL,
        "dataset_key": DATASET_KEY,
        "midpoint_definition": "centroid of largest 2D area component",
        "self_connections_excluded": True,
        "interior_quantile": INTERIOR_QUANTILE,
        "boundary_clearance_threshold": float(clearance_threshold),
        "lambda": float(lambda_value),
        "candidate_count": CANDIDATE_COUNT,
        "candidate_targets": candidates["target"].tolist(),
        "candidate_score": "mean percentile of empirical and geometry-corrected resultant among interior targets",
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def main():
    animal_info = uf.AnimalInfo(ANIMAL, DATASET_KEY)
    connection_matrix = animal_info.SC_data
    lambda_value = uf.OPTIMAL_PARAMS[f"{ANIMAL}_{DATASET_KEY}"]["Lambda"]
    metrics = get_target_metrics(animal_info, connection_matrix, lambda_value)
    metrics, clearance_threshold = get_candidate_targets(metrics, CANDIDATE_COUNT, INTERIOR_QUANTILE)
    save_results(metrics, clearance_threshold, lambda_value, OUTPUT_DIR)
    plot_overview(metrics, clearance_threshold, animal_info, OUTPUT_DIR)
    plot_candidate_source_patterns(metrics, animal_info, connection_matrix, OUTPUT_DIR)
    selected = metrics.loc[metrics["is_candidate"]].sort_values("candidate_rank")
    print(selected[["candidate_rank", "target", "boundary_clearance", "empirical_resultant", "corrected_resultant"]].to_string(index=False))
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
