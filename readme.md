Create and activate a Python virtual environment using your preferred environment manager, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run each notebook from its own directory so that its relative utility path resolves correctly:

1. Run `code/simulation/optimal_results.ipynb`.
2. Copy the first four printed model directories into `model_timedir_list` in `code/visualization/paper_fig.ipynb`.
3. Copy the printed marmoset V4 directory into `marmoset_V4_timedir`.
4. Run `code/visualization/paper_fig.ipynb` to reproduce the main figures, supplementary figures, and table.

The optimal parameters are defined in `code/utils_function/utils_function.py`. The marmoset V4 model intentionally uses the marmoset Majka parameter.
