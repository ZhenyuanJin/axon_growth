import json
from pathlib import Path

import pandas as pd

from experiments.marmoset_target_conditioned_angle_edr import run_experiment as experiment
from utils_function import utils_function as uf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "results" / "experiments" / "marmoset_target_conditioned_angle_edr"


def main():
    scan_metrics = pd.read_csv(OUTPUT_DIR / "lambda_scan_metrics.csv")
    target_metrics = pd.read_csv(OUTPUT_DIR / "target_metrics.csv")
    optimal_rows = experiment.get_optimal_rows(scan_metrics)
    metric_optima = experiment.get_metric_optima(scan_metrics)
    target_comparison = experiment.get_target_comparison(optimal_rows, target_metrics)
    animal_info = uf.AnimalInfo(experiment.ANIMAL, experiment.DATASET_KEY)
    SC_data = animal_info.SC_data.loc[animal_info.source_name, animal_info.target_name]
    experiment.plot_method_overview(SC_data, animal_info, OUTPUT_DIR)
    experiment.plot_correlation_improvement(scan_metrics, optimal_rows, target_comparison, OUTPUT_DIR)
    optimal_rows.to_csv(OUTPUT_DIR / "optimal_models.csv", index=False)
    metric_optima.to_csv(OUTPUT_DIR / "metric_optima.csv", index=False)
    target_comparison.to_csv(OUTPUT_DIR / "optimal_target_comparison.csv", index=False)

    metadata_path = OUTPUT_DIR / "metadata.json"
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    metadata["optimal_models"] = optimal_rows.to_dict(orient="records")
    metadata["metric_optima"] = metric_optima.to_dict(orient="records")
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(metric_optima.to_string(index=False))
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
