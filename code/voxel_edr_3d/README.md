# Isolated 3D EDR connection-generation models

This isolated workflow samples directed connections between all distinct voxels
in the Majka marmoset atlas. For a pair at distance `d`, its unnormalized
weight is `exp(-d / Lambda) / N(d)`, where `N(d)` is the number of ordered
voxel pairs at the same exact grid distance.

Run from the repository root:

```bash
PYTHONPATH=code python code/voxel_edr_3d/run_parameter_search.py
PYTHONPATH=code python code/voxel_edr_3d/run_macaque_parameter_search.py
```

The workflow first builds an exact implicit displacement catalog under
`results/voxel_edr_3d/preprocessing/`, then performs the coarse and fine lambda
scans. The selected parameters, connection matrices, raw voxel pairs for the
best lambda, and validation figures are saved under `results/voxel_edr_3d/`.

## Validated Majka result

The completed 20-point coarse scan and 11-point fine scan selected
`Lambda = 0.7219740003313116 mm`. Across five 8,000,000-connection samples, the
mean Pearson correlation with measured FLNe was `0.7747718297842224` with a
standard deviation of `0.00035595069784068775`. Machine-readable values are in
`results/voxel_edr_3d/optimal_parameters.json`.

For macaque Markov40, the completed scan selected
`Lambda = 1.7188026860728818 mm`. Across five 8,000,000-connection samples, the
mean Pearson correlation was `0.6837347174120414` with a standard deviation of
`0.0006081089582068091`. These outputs are stored under
`results/voxel_edr_3d_macaque_markov40/`.
