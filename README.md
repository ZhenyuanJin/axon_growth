# Cross-Species Connection Generation

This repository contains deterministic connection-generation models and the analysis code used to reproduce the project figures.

## Setup

Python 3.9 is the validated runtime. Create a virtual environment and install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

All commands below are run from the repository root. The project does not need to be installed as a package.

## Main workflow

```bash
make connection-generation
make paper-figures
```

The first command runs the primary model for mouse, marmoset, macaque, human, and the marmoset V4 dataset. Results are written under `results/models/`. The second command creates the 16 main and supplementary figures under `results/figures/` in PDF and PNG formats.

Parameter searches and their summary are run with:

```bash
make parameter-search
make result-analysis
```

Set a specific interpreter when needed, for example `make test PYTHON=/path/to/python`.

## Code layout

- `code/utils_function/utils_function.py`: data loading, model calculations, and analyses.
- `code/utils_function/cf/`: project-local I/O, statistics, parallel, plotting, and style helpers.
- `code/simulation/`: connection-generation and parameter-search entry points, plus retained comparison workflows.
- `code/result_analysis/`: parameter-search result analysis.
- `code/visualization/`: functional plotting helpers, the `PaperFig` orchestrator, and figure generation.
- `code/data_analysis/`: additional data-analysis workflows converted from the former notebooks.
- `Makefile`: supported repository-root commands; it exposes local code without installing a package.

All workflow data and output paths are resolved from each script's location rather than the current working directory.

## Data and outputs

Processed inputs are stored in `data/`; source datasets are stored in `raw_data/`. Generated outputs are placed in `results/`, which is ignored by Git.

Each model run contains a configuration manifest, NumPy simulation arrays, CSV connection matrices, scalar metrics, and optional check figures. Output directories use a deterministic configuration hash rather than a timestamp.

The comparison models retain their previous calculations but are not used to generate the paper figures. The repository contains no notebook execution path.
