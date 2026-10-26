# Target-conditioned angle and EDR benchmark

See `RESULTS.md` for the concise experiment record and numerical results.

This isolated experiment compares two target-first connection-generation models
for the marmoset Majka dataset:

- `random`: source direction is uniform on the circle, matching the main code;
- `SC_angle_kde`: source direction is drawn from a target-specific circular kernel
  mixture whose centers are target-to-source midpoint angles and whose weights
  are empirical FLNe values.

For named components such as `V2--1` and `V2--2`, each source/target
component pair receives `FLNe * source_area_fraction * target_area_fraction`.
Fractions are normalized within each area; an unsplit area has fraction exactly
one, so its FLNe receives no area scaling. Pair weights sum to the original FLNe.

Both direction methods generate the target point first, use
`length_method='exponential'`, draw distance from the same EDR with a scanned
`Lambda`, and then place the source point. Target points, distance seeds, `na`,
`max_iteration`, and evaluation code are shared between the methods.
Self-connections and points outside annotated source areas are rejected.

The circular kernel is a von Mises distribution with a 15-degree small-angle
standard-deviation approximation. This is an in-sample benchmark by design: the
same empirical FLNe matrix supplies the directional distribution and evaluates
the generated matrix.

Run from the repository root:

```bash
PYTHONPATH=code python code/experiments/marmoset_target_conditioned_angle_edr/run_experiment.py
```

Outputs are written to
`results/experiments/marmoset_target_conditioned_angle_edr/`.

Check component-weight conservation and unsplit-area invariance:

```bash
PYTHONPATH=code python code/experiments/marmoset_target_conditioned_angle_edr/test_component_weights.py
```

Regenerate tables and figures from an existing completed scan without rerunning
connection generation:

```bash
PYTHONPATH=code python code/experiments/marmoset_target_conditioned_angle_edr/plot_saved_results.py
```
