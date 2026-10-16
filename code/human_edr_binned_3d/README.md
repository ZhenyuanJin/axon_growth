# Corrected human binned 3D EDR

This isolated workflow implements a fixed-width continuous-distance EDR for
human Lariviere surface vertices. It does not modify or overwrite the existing
connection-generation workflows.

For each nonempty 0.1 mm distance bin, the total target mass is the exact
integral of the exponential density over the bin. That mass is divided equally
among all ordered non-self vertex pairs in the bin. Thus geometric pair density
does not change the target distance distribution.

Run from the repository root:

```bash
PYTHONPATH=code python code/human_edr_binned_3d/run_parameter_search.py
```

Outputs are saved under `results/human_edr_binned_3d/`.

## Validated result

The completed 20-point coarse scan and 11-point fine scan selected
`Lambda = 13.221286142255256 mm`. Across five 8,000,000-connection samples,
the mean Pearson correlation was `0.6530640453652067` with a standard deviation
of `0.0005804709280996761`. Machine-readable values are stored in
`results/human_edr_binned_3d/optimal_parameters.json`.
