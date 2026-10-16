import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
from human_geodesic_analysis import analysis as shared_analysis
from human_geodesic_analysis import plotting as shared_plotting
from macaque_geodesic_analysis import analysis
from macaque_geodesic_analysis import geodesic
from utils_function import utils_function as uf
from voxel_edr_3d import model as voxel_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "macaque_geodesic_analysis"
CACHE_DIR = OUTPUT_DIR / "preprocessing"
VOXEL_CACHE_DIR = PROJECT_ROOT / "results" / "voxel_edr_3d_macaque_markov40" / "preprocessing"
GRAPH_PATH = CACHE_DIR / "no_corner_cutting_graph.npz"
SAMPLES_PER_AREA = 10
SAMPLE_SEED = 20261017
BATCH_SIZE = 4
BIN_COUNT = 20


def get_graph(catalog):
    if GRAPH_PATH.exists():
        return scipy.sparse.load_npz(GRAPH_PATH)
    print("Building no-corner-cutting macaque cortical voxel graph")
    graph = geodesic.get_no_corner_cutting_graph(
        catalog["grid_indices"], catalog["voxel_map"], catalog["metadata"]["spacing_mm"]
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    scipy.sparse.save_npz(GRAPH_PATH, graph)
    return graph


def main():
    catalog = voxel_model.load_displacement_catalog(VOXEL_CACHE_DIR)
    graph = get_graph(catalog)
    info = uf.AnimalInfo("macaque", "markov40")
    area_names = list(catalog["metadata"]["area_names"])
    area_lookup = {name: index for index, name in enumerate(area_names)}
    sampled_area_ids = np.asarray([area_lookup[name] for name in info.target_name], dtype=np.int64)
    largest_mask, component_sizes = geodesic.get_largest_component_mask(graph)
    source_indices, source_area_ids = geodesic.get_sampled_source_indices(
        np.asarray(catalog["area_ids"]), sampled_area_ids, largest_mask,
        SAMPLES_PER_AREA, SAMPLE_SEED
    )
    print(f"Computing distances from {len(source_indices)} sampled source voxels")
    geodesic_mean, euclidean_mean, component_sizes = geodesic.get_sampled_area_distance_matrices(
        graph, catalog["coordinates"], catalog["area_ids"], sampled_area_ids,
        source_indices, source_area_ids, BATCH_SIZE
    )
    ratio = geodesic_mean / euclidean_mean
    row_ids = np.asarray([area_lookup[name] for name in info.source_name], dtype=np.int64)
    ratio_comparison = ratio[np.ix_(row_ids, np.arange(len(sampled_area_ids)))]
    row_names = list(info.source_name)
    column_names = list(info.target_name)
    pair_table = analysis.get_ratio_pair_table(ratio_comparison, row_names, column_names)
    measured, model_2d, model_3d = analysis.load_macaque_matrices(PROJECT_ROOT)
    binned = shared_analysis.get_ratio_binned_rmae(
        pair_table, measured, model_2d, model_3d, BIN_COUNT, "macaque", "markov40"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    binned.to_csv(OUTPUT_DIR / "shortcut_ratio_binned_rmae_by_link_status.csv", index=False)
    pd.DataFrame(ratio_comparison, index=row_names, columns=column_names).to_csv(
        OUTPUT_DIR / "geodesic_to_euclidean_ratio.csv"
    )
    np.save(CACHE_DIR / "sampled_source_indices.npy", source_indices, allow_pickle=False)
    shared_plotting.plot_ratio_binned_rmae(
        binned, figures_dir / "shortcut_ratio_binned_rmae_by_link_status"
    )
    masks = uf.get_weight_category_mask("macaque", "markov40", diagonal=False)
    relative_error_2d = uf.data_model_relative_error(
        measured, model_2d, "macaque", "markov40", diagonal=False
    )
    relative_error_3d = uf.data_model_relative_error(
        measured, model_3d, "macaque", "markov40", diagonal=False
    )
    summary = {
        "corr_2d": float(uf.data_model_correlation(measured, model_2d, "macaque", "markov40", diagonal=False)),
        "corr_3d": float(uf.data_model_correlation(measured, model_3d, "macaque", "markov40", diagonal=False)),
        "sample_seed": SAMPLE_SEED,
        "samples_per_area": SAMPLES_PER_AREA,
        "sampled_area_count": len(sampled_area_ids),
        "graph_undirected_edge_count": int(graph.nnz // 2),
        "graph_component_count": len(component_sizes),
        "largest_component_voxel_count": int(np.max(component_sizes)),
    }
    for category, mask_key in [("existing", "all"), ("non_link", "non")]:
        mask = masks[mask_key]
        summary[f"rmae_2d_{category}"] = float(np.nansum(relative_error_2d.values * mask.values) / np.sum(mask.values))
        summary[f"rmae_3d_{category}"] = float(np.nansum(relative_error_3d.values * mask.values) / np.sum(mask.values))
    non_link_bins = binned[binned["link_category"] == "non-link"]
    non_link_delta = non_link_bins["rmae_3d"] - non_link_bins["rmae_2d"]
    summary["non_link_ratio_vs_rmae_3d_spearman"] = float(
        non_link_bins["ratio_mean"].corr(non_link_bins["rmae_3d"], method="spearman")
    )
    summary["non_link_ratio_vs_3d_minus_2d_spearman"] = float(
        non_link_bins["ratio_mean"].corr(non_link_delta, method="spearman")
    )
    summary["non_link_lowest_ratio_bin_3d_minus_2d"] = float(non_link_delta.iloc[0])
    summary["non_link_highest_ratio_bin_3d_minus_2d"] = float(non_link_delta.iloc[-1])
    summary["interpretation"] = "high_ratio_non_link_degradation_supported_but_not_sufficient_to_explain_global_correlation_gap"
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
    print(f"Graph edges: {graph.nnz // 2}; components: {len(component_sizes)}; largest: {int(np.max(component_sizes))}")
    for category, table in binned.groupby("link_category", sort=False):
        print(f"{category}: {int(table['pair_count'].min())}-{int(table['pair_count'].max())} pairs per bin")


if __name__ == "__main__":
    main()
