# Connection Generation Across Species

This repository contains deterministic connection-generation models and the analysis code used to reproduce the project figures.

## Setup

Python 3.9 is the validated runtime. Create a virtual environment and install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

All commands below are run from the repository root. The project does not need to be installed as a package.

## Main workflow

```bash
python code/run_connection_generation.py
python code/run_paper_figures.py
python code/validate_figures.py
```

The first command runs the primary model for mouse, marmoset, macaque, human, and the marmoset V4 dataset. Results are written under `results/models/`. The second command creates the 16 main and supplementary figures under `results/figures/` in PDF and PNG formats.

Parameter searches and their summary are run with:

```bash
python code/run_parameter_search.py
python code/run_result_analysis.py
```

## Code layout

- `code/utils_function/utils_function.py`: data loading, model calculations, analyses, and domain plots.
- `code/utils_function/cf/`: project-local I/O, statistics, parallel, plotting, and style helpers.
- `code/run_*.py`: supported entry points for generation, parameter search, result analysis, and figure creation.
- `code/data_analysis/`, `code/result_analysis/`, and `code/simulation/`: detailed Python workflows converted from the former notebooks.

## Data and outputs

Processed inputs are stored in `data/`; source datasets are stored in `raw_data/`. Generated outputs are placed in `results/`, which is ignored by Git. Local visual references are expected at `results/reference_figs/`.

Each model run contains a configuration manifest, NumPy simulation arrays, CSV connection matrices, scalar metrics, and optional check figures. Output directories use a deterministic configuration hash rather than a timestamp.

The comparison models retain their previous calculations but are not used to generate the paper figures. The repository contains no notebook execution path.
