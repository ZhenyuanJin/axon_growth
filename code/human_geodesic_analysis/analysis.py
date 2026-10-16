import json
from pathlib import Path

import numpy as np
import pandas as pd
from utils_function import utils_function as uf


def load_geodesic_pair_table(cache_dir):
    """Load one row per unordered area pair with its mean distance ratio.

    The returned DataFrame contains area names and the ratio between mean mesh
    geodesic distance and mean Euclidean distance. No randomness is used.
    """
    cache_dir = Path(cache_dir)
    with (cache_dir / "metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    pair_counts = np.load(cache_dir / "pair_counts.npy", mmap_mode="r")
    sum_euclidean = np.load(cache_dir / "sum_euclidean.npy", mmap_mode="r")
    sum_geodesic = np.load(cache_dir / "sum_geodesic.npy", mmap_mode="r")
    area_names = np.asarray(metadata["area_names"])
    row, col = np.triu_indices(len(area_names), k=1)
    if np.any(pair_counts[row, col] <= 0) or np.any(sum_euclidean[row, col] <= 0):
        raise ValueError("Every unordered area pair must contain positive distances.")
    ratio = sum_geodesic[row, col] / sum_euclidean[row, col]
    if not np.all(np.isfinite(ratio)) or np.any(ratio < 1):
        raise ValueError("Geodesic-to-Euclidean ratios must be finite and at least one.")
    return pd.DataFrame({
        "source_area": area_names[row],
        "target_area": area_names[col],
        "geodesic_to_euclidean_ratio": ratio,
    })


def load_human_matrices(project_root):
    """Load aligned measured, accepted 2D, and corrected 3D matrices."""
    project_root = Path(project_root)
    info = uf.AnimalInfo("human", "lariviere")
    names = info.full_name
    measured = info.SC_data.loc[names, names]
    model_2d_path = project_root / "results" / "models" / "connection_generation" / "human_lariviere" / "dc162c4415b8" / "connectivity" / "CCMe_sym_normalized.csv"
    model_2d = pd.read_csv(model_2d_path, index_col=0).loc[names, names]
    model_3d_paths = sorted(
        (project_root / "results" / "human_edr_binned_3d" / "best").glob(
            "best_lambda_000_seed_*/connectivity/CCMe_sym_normalized.csv"
        )
    )
    if len(model_3d_paths) != 5:
        raise ValueError(f"Expected five corrected 3D best-seed matrices, found {len(model_3d_paths)}.")
    model_3d_arrays = [pd.read_csv(path, index_col=0).loc[names, names].values for path in model_3d_paths]
    model_3d = pd.DataFrame(np.mean(model_3d_arrays, axis=0), index=names, columns=names)
    return measured, model_2d, model_3d


def get_ratio_binned_rmae(pair_table, measured, model_2d, model_3d, bin_count, animal, dataset_key):
    """Return existing- and non-link RMAE on shared pooled-ratio bins.

    Quantiles of all area-pair ratios define one shared set of bin edges. Both
    link categories use those exact bins and the pooled mean ratio as their
    shared x coordinate. No randomness is used.
    """
    if not isinstance(bin_count, int) or bin_count < 2:
        raise ValueError("bin_count must be an integer greater than one.")
    required = {"source_area", "target_area", "geodesic_to_euclidean_ratio"}
    missing = required.difference(pair_table.columns)
    if missing:
        raise ValueError(f"pair_table is missing required columns: {sorted(missing)}")
    relative_error_2d = uf.data_model_relative_error(
        measured, model_2d, animal, dataset_key, diagonal=False
    )
    relative_error_3d = uf.data_model_relative_error(
        measured, model_3d, animal, dataset_key, diagonal=False
    )
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
    bin_id = np.searchsorted(edges, ratio, side="right") - 1
    bin_id = np.clip(bin_id, 0, bin_count - 1)
    pooled_ratio_mean = np.asarray([
        np.mean(ratio[bin_id == local_bin]) for local_bin in range(bin_count)
    ])
    category_masks = uf.get_weight_category_mask(animal, dataset_key, diagonal=False)
    categories = [("existing", "all"), ("non-link", "non")]
    rows = []
    for category_label, category_key in categories:
        category = category_masks[category_key].values[source_index, target_index]
        for local_bin in range(bin_count):
            selected = category & (bin_id == local_bin)
            pair_count = int(np.sum(selected))
            rows.append({
                "link_category": category_label,
                "bin": local_bin + 1,
                "ratio_left": float(edges[local_bin]),
                "ratio_mean": float(pooled_ratio_mean[local_bin]),
                "ratio_right": float(edges[local_bin + 1]),
                "pair_count": pair_count,
                "rmae_2d": float(np.mean(error_2d[selected])) if pair_count else np.nan,
                "rmae_3d": float(np.mean(error_3d[selected])) if pair_count else np.nan,
            })
    return pd.DataFrame(rows)
