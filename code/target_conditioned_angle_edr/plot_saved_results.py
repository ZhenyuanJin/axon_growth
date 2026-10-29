import pandas as pd

from target_conditioned_angle_edr import model
from target_conditioned_angle_edr import parameters
from target_conditioned_angle_edr import plotting


def main():
    for animal, key in parameters.DATASETS:
        output_dir = parameters.OUTPUT_DIR / f"{animal}_{key}"
        metadata_path = output_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Incomplete dataset: {output_dir}")
        context = model.get_context(animal, key)
        tables = output_dir / "tables"
        scan = pd.read_csv(tables / "lambda_scan_summary.csv")
        optimal = pd.read_csv(tables / "optimal_models.csv")
        comparison = pd.read_csv(tables / "optimal_target_comparison.csv", dtype={"target": str})
        plotting.plot_all_angle_distributions(context["angles"], output_dir / "angle_distributions",
                                               parameters.ANGLE_KDE_BANDWIDTH_DEGREES)
        plotting.plot_correlation_improvement(scan, optimal, comparison, output_dir / "figures", context["info"].SC_type)
        plotting.plot_metric_scans(scan, output_dir / "figures", context["info"].SC_type)


if __name__ == "__main__":
    main()
