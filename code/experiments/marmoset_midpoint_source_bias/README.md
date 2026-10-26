# Marmoset midpoint source-bias experiment

This experiment represents every cortical area by the centroid of its largest
2D flat-map component. For each marmoset target area, it measures whether the
empirical incoming connection weights are directionally balanced around that
target midpoint.

The experiment is isolated from the primary workflows. It imports project data
loaders and plotting helpers but does not modify or call the connection-generation
pipeline.

Candidate targets must have boundary clearance at least as large as the median
clearance among all empirical targets. They are ranked by the mean percentile
of:

1. the empirical directional resultant length; and
2. the difference between the empirical direction vector and an area-weighted,
   distance-dependent midpoint baseline using the fitted marmoset Lambda.

Self-connections are excluded from all direction calculations.

Run from the repository root:

```bash
PYTHONPATH=code python code/experiments/marmoset_midpoint_source_bias/run_experiment.py
```

Outputs are written to `results/experiments/marmoset_midpoint_source_bias/`.
