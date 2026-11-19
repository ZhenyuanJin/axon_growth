"""Generate a horizontal four-species target-correlation figure from saved tables."""

import pandas as pd

from target_conditioned_angle_edr import parameters
from target_conditioned_angle_edr import plotting


def main():
    comparisons = {}
    for animal, key in parameters.DATASETS:
        if key == "V4":
            continue
        table = parameters.OUTPUT_DIR / f"{animal}_{key}" / "tables" / "optimal_target_comparison.csv"
        comparisons[animal] = pd.read_csv(table, dtype={"target": str})
    output_dir = parameters.OUTPUT_DIR / "figures"
    plotting.plot_cross_species_target_correlations(comparisons, output_dir)
    print(f"Saved {output_dir / 'cross_species_target_correlations.png'}", flush=True)
    print(f"Saved {output_dir / 'cross_species_target_correlations.pdf'}", flush=True)


if __name__ == "__main__":
    main()
