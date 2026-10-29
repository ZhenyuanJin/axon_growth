# Repeated target-conditioned angle and EDR experiment

Run from the repository root with the validated Python 3.9 environment and the
dependencies pinned in `requirements.txt`. The parameter file is
`parameters.py`.

The modules follow the workflow:

- `sampling.py`: deterministic seeds, component weights, geometry lookup, and connection samples.
- `model.py`: task execution and evaluation through the project SC conversion.
- `analysis.py`: repeated metric summaries, optimum selection, and paired comparisons.
- `reporting.py`: shared table, matrix, metadata, and provenance exports for generation and reevaluation.
- `plotting.py`: figures using project-local defaults.
- `run_experiment.py`, `evaluate_results.py`, `plot_saved_results.py`, and
  `validate_results.py`: repository-root entry points.

```bash
export PYTHONPATH=code
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
python code/target_conditioned_angle_edr/run_experiment.py
```

The five datasets are mouse Gamanut, marmoset Majka, macaque Markov40, human
Lariviere, and marmoset V4. Each target requests 20,000 connection samples, with
up to ten attempts per target point. Unresolved points are explicitly recorded.
Each Lambda has five independent repeats. Both direction methods share the
target points and the distance of every original sample ID at every attempt.
The repeat seed depends only on dataset, repeat ID, and the base seed; it never
depends on workers, completion order, or the current scan grid.

Only direction sampling differs between the methods:

- `random`: uniform direction;
- `SC_angle_kde`: target-specific von Mises mixture with 15-degree approximate
  bandwidth, centered at source-component centroid minus target-component
  centroid, weighted by empirical `SC[source, target]` times the source and
  target component area fractions. Pair weights conserve each area's weight.

Target points are sampled uniformly within an area, allocating its components
by area. Self-connections and source points outside the annotated cortex are
rejected. Overlapping source polygons use dictionary insertion order to select
the first match and require membership in the whole cortex. Polygon validity
and overlaps are recorded in `geometry_audit.csv`.

Every method generates actual source/target coordinates and split-area IDs.
Those IDs go through
`uf.get_connectivity_from_connection_samples`: split-level counts, area-level
aggregation, self-connection handling, and every SC representation are produced
by the project pipeline. Evaluation uses `FLNe` for tracer datasets and
`CCMe_sym_normalized` for human, selected with `uf.SC_type_to_SC_dict_key`.
Human is generated target-first; its count matrix is symmetrized and globally
normalized by the same conversion pipeline.

Correlation aligns the source and target labels of the two matrices and excludes
self-connections where the source and target names match. Overall, log, and
target correlations and relative errors evaluate each repeat independently. Tables report arithmetic
means, sample SD (`ddof=1`), and SEM. The optimum for each method maximizes the
mean overall correlation, then target comparisons pair repeat IDs at those
fixed optima. Mean SC matrices are saved for inspection.
Other metric optima and paired raw differences are saved as well.

Initial Lambda values scale the marmoset grid `0.75:0.125:3.5` by each
dataset's existing reference Lambda. When either mean-correlation optimum lies
at a scan edge, the common grid is extended, up to six rounds; unresolved edges
are explicitly flagged. Other metric optima are restricted to the tested grid.

Outputs live in `results/experiments/target_conditioned_angle_edr/`:

- `<dataset>/angle_distributions/`: KDE-only PNG/PDF for every measured target,
  plus numeric densities. Angles are empirical and do not require repeated runs.
- `<dataset>/figures/`: correlation comparison and other metric scans, with
  means and sample SD; target points also carry SD error bars.
- `<dataset>/tables/`: task table, all repeats, summaries, paired differences,
  empirical matrix, low-acceptance records, and geometry checks.
- `<dataset>/runs/<task_id>/<method>/`: coordinates and integer split IDs in
  `connection_samples.npz`, plus every SC representation in
  `connectivity/`. `area_names.json` decodes split IDs. `sample_ids` identify the
  original target-point index within the target area.
- `<dataset>/optimal_matrices/`: all five optimum SCs and their separate mean.
- `provenance/`: source snapshots, SHA256 hashes, versions, and configuration.
- `cross_dataset_optimal_models.csv`: dataset-level summary.

Each paired task writes only its own directory. Completion markers allow
interrupted scans to resume; changed task configurations are rejected. The
main process collects tasks in sorted ID order. Regenerate figures without
simulating:

```bash
python code/target_conditioned_angle_edr/plot_saved_results.py
python code/target_conditioned_angle_edr/evaluate_results.py
python code/target_conditioned_angle_edr/validate_results.py
python -m pytest -q tests/test_target_conditioned_angle_edr.py
```

Interpretation:

- The empirical matrix supplies the angle mixture and evaluates the output;
  this is an in-sample reconstruction comparison. Five repeats measure sampling
  variability, not generalization uncertainty.
- Low acceptance means fewer than 20,000 valid samples. Requested, accepted,
  unresolved, self-rejected, and outside-rejected counts are all retained.
