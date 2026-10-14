import json
from collections import defaultdict
from pathlib import Path

from utils_function import utils_function as uf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEARCH_DIR = PROJECT_ROOT / "results" / "parameter_search"
OUTPUT_FILE = PROJECT_ROOT / "results" / "optimal_parameters.json"


def get_optimal_parameters():
    grouped = defaultdict(list)
    for manifest_path in sorted(SEARCH_DIR.glob("**/manifest.json")):
        if manifest_path.parent.name in {"simulation", "connectivity"}:
            continue
        model = uf.ConnectionGenerationModel()
        model.set_timedir(manifest_path.parent)
        key = f"{model.animal}_{model.dataset_key}"
        grouped[key].append({
            "Lambda": model.get_unique_Lambda(),
            "seed": model.the_seed,
            "correlation": model.get_data_model_correlation(),
        })
    optimal = {}
    for key, candidates in grouped.items():
        best = max(candidates, key=lambda item: item["correlation"])
        optimal[key] = best
    return optimal


def main():
    optimal = get_optimal_parameters()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(optimal, file, indent=2, sort_keys=True)
    print(json.dumps(optimal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
