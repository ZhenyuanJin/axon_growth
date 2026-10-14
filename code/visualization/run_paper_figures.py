from pathlib import Path

from simulation.run_connection_generation import DATASETS, OUTPUT_DIR, get_run_config
from utils_function import utils_function as uf
from utils_function.model_runtime import get_config_hash
from visualization.paper_figures import PaperFig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"


def get_run_dir(animal, dataset_key):
    config = get_run_config(animal, dataset_key)
    dataset_name = f"{animal}_{dataset_key}"
    return OUTPUT_DIR / "connection_generation" / dataset_name / get_config_hash(config)


def main():
    run_dir_by_dataset = {
        f"{animal}_{dataset_key}": get_run_dir(animal, dataset_key)
        for animal, dataset_key in DATASETS
    }
    missing = [str(path) for path in run_dir_by_dataset.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run make connection-generation before plotting. Missing:\n"
            + "\n".join(missing)
        )
    model_run_dirs = [
        run_dir_by_dataset["mouse_gamanut"],
        run_dir_by_dataset["marmoset_majka"],
        run_dir_by_dataset["macaque_markov40"],
        run_dir_by_dataset["human_lariviere"],
    ]
    figures = PaperFig(
        model_timedir_list=[str(path) for path in model_run_dirs],
        marmoset_V4_timedir=str(run_dir_by_dataset["marmoset_V4"]),
        output_dir=FIGURE_DIR,
    )
    figures.plot_all()
    figures.table_1()
    print(f"Figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
