from pathlib import Path

from voxel_edr_3d.run_parameter_search import main

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "voxel_edr_3d_macaque_markov40"


if __name__ == "__main__":
    main(animal="macaque", dataset_key="markov40", output_dir=OUTPUT_DIR)
