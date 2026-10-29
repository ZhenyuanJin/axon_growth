import numpy as np
import pandas as pd

from target_conditioned_angle_edr import parameters


def get_repeated_summary(frame, group_columns, metric_columns, repeat_count):
    """Return arithmetic means, sample SDs and SEMs after validating complete independent repeats."""
    if frame.duplicated(group_columns + ["repeat_id"]).any():
        raise ValueError("Duplicate repeat in a summary group.")
    groups = frame.groupby(group_columns, sort=True, dropna=False)
    counts = groups.repeat_id.nunique()
    if not (counts == repeat_count).all():
        raise ValueError(f"Incomplete repeats: {counts[counts != repeat_count].to_dict()}")
    if not np.all(np.isfinite(frame[metric_columns].to_numpy())):
        raise ValueError("Cannot average nonfinite metrics.")
    summary = groups[metric_columns].agg(["mean", "std", "sem"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary["repeat_count"] = counts
    return summary.reset_index()


def get_summaries(metrics, targets, repeat_count):
    """Summarize independently evaluated task metrics and target metrics."""
    global_columns = list(parameters.METRIC_OBJECTIVES) + [
        "mean_acceptance_rate", "minimum_acceptance_rate",
        "accepted_count", "unresolved_count",
    ]
    target_columns = ["correlation", "log_correlation", "rmae_both",
                      "acceptance_rate", "accepted_count", "unresolved_count", "proposal_count",
                      "self_rejections", "outside_rejections"]
    scan = get_repeated_summary(metrics, ["direction_method", "grid_index", "Lambda"], global_columns, repeat_count)
    target_summary = get_repeated_summary(
        targets, ["direction_method", "grid_index", "Lambda", "target"], target_columns, repeat_count,
    )
    optimal_rows, optima = [], []
    for method, subset in scan.groupby("direction_method", sort=True):
        optimal_rows.append(subset.loc[subset.correlation_mean.idxmax()])
        for metric, objective in parameters.METRIC_OBJECTIVES.items():
            index = subset[f"{metric}_mean"].idxmax() if objective == "max" else subset[f"{metric}_mean"].idxmin()
            row = subset.loc[index]
            optima.append({"direction_method": method, "metric": metric, "objective": objective,
                           "Lambda": row.Lambda, "grid_index": row.grid_index,
                           "mean": row[f"{metric}_mean"], "std": row[f"{metric}_std"],
                           "sem": row[f"{metric}_sem"], "repeat_count": repeat_count})
    return scan, target_summary, pd.DataFrame(optimal_rows).reset_index(drop=True), pd.DataFrame(optima)


def get_paired_comparison(metrics, targets, optimal_rows, repeat_count):
    """Pair repeat IDs at each method's mean-correlation optimum, then summarize differences."""
    selected_metrics, selected_targets = [], []
    for _, row in optimal_rows.iterrows():
        for frame, destination in [(metrics, selected_metrics), (targets, selected_targets)]:
            destination.append(frame.loc[(frame.direction_method == row.direction_method)
                                         & (frame.grid_index == row.grid_index)])
    selected_metrics = pd.concat(selected_metrics, ignore_index=True)
    selected_targets = pd.concat(selected_targets, ignore_index=True)
    paired_frames = []
    for frame, keys in [(selected_metrics, ["repeat_id"]), (selected_targets, ["repeat_id", "target"])]:
        random = frame.loc[frame.direction_method == "random"]
        kde = frame.loc[frame.direction_method == "SC_angle_kde"]
        paired = random.merge(kde, on=keys, suffixes=("_random", "_SC_angle_kde"), validate="one_to_one")
        for metric in ["correlation", "log_correlation", "rmae_both"]:
            paired[f"{metric}_difference"] = paired[f"{metric}_SC_angle_kde"] - paired[f"{metric}_random"]
        paired_frames.append(paired)
    target_paired = paired_frames[1]
    target_columns = [f"{metric}_{suffix}" for metric in ["correlation", "log_correlation", "rmae_both"]
                      for suffix in ["random", "SC_angle_kde", "difference"]]
    comparison = get_repeated_summary(target_paired, ["target"], target_columns, repeat_count)
    comparison = comparison.sort_values("correlation_difference_mean", ascending=False).reset_index(drop=True)
    return paired_frames[0], target_paired, comparison


def get_boundary_extensions(scan, current_indices):
    """Extend a common grid when either method's mean correlation peaks at a boundary."""
    lower, upper = min(current_indices), max(current_indices)
    best = scan.loc[scan.groupby("direction_method").correlation_mean.idxmax()]
    indices = []
    if (best.grid_index == lower).any() and lower > 1:
        indices.extend(range(max(1, lower - parameters.GRID_EXTENSION_COUNT), lower))
    if (best.grid_index == upper).any():
        indices.extend(range(upper + 1, upper + 1 + parameters.GRID_EXTENSION_COUNT))
    return indices
