from pathlib import Path

from human_geodesic_analysis import analysis
from human_geodesic_analysis import plotting


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "human_geodesic_analysis"
CACHE_DIR = OUTPUT_DIR / "preprocessing"
BIN_COUNT = 10


def main():
    pair_table = analysis.load_geodesic_pair_table(CACHE_DIR)
    measured, model_2d, model_3d = analysis.load_human_matrices(PROJECT_ROOT)
    binned_rmae = analysis.get_ratio_binned_rmae(
        pair_table, measured, model_2d, model_3d, BIN_COUNT, "human", "lariviere"
    )
    binned_rmae.to_csv(OUTPUT_DIR / "shortcut_ratio_binned_rmae_by_link_status.csv", index=False)
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plotting.plot_ratio_binned_rmae(
        binned_rmae, figures_dir / "shortcut_ratio_binned_rmae_by_link_status"
    )
    for category, table in binned_rmae.groupby("link_category", sort=False):
        occupied = table["pair_count"] > 0
        print(f"{category}: {int(occupied.sum())}/{BIN_COUNT} occupied bins, {int(table['pair_count'].sum())} pairs")


if __name__ == "__main__":
    main()
