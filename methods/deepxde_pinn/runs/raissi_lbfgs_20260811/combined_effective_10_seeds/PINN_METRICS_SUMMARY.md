# PINN Metrics Summary

## Configuration

- Method: standard Raissi-style PINN
- Network: 6 hidden layers, 128 neurons per layer, `tanh`
- Training: fixed points, equal loss-component weights, no Adam, one full-batch
  L-BFGS call with at most 1,000 iterations
- Runs: 10 valid seeds per equation
- Statistics: mean +/- sample standard deviation
- Accuracy metrics: relative L2 error and maximum pointwise error

## Aggregate Results

| Equation | Valid runs | Relative L2 error | Maximum pointwise error |
|---|---:|---:|---:|
| 1D Poisson (Boundary) | 10 | 3.726645e+01 +/- 2.723103e+01 | 9.749287e+00 +/- 7.132510e+00 |
| 1D Poisson (Steep) | 10 | 2.076111e+00 +/- 1.283031e+00 | 2.966808e+00 +/- 1.579812e+00 |
| 2D Poisson (Peak) | 10 | 1.395887e-02 +/- 2.268726e-03 | 1.125535e-02 +/- 4.263781e-03 |
| 2D Poisson (Steep) | 10 | 4.177518e-02 +/- 1.093627e-02 | 1.699930e-01 +/- 6.235782e-02 |
| 1D Helmholtz | 10 | 9.999996e-01 +/- 1.909508e-06 | 1.000015e+00 +/- 8.669230e-06 |
| 2D Helmholtz | 10 | 1.913615e-01 +/- 9.676438e-02 | 6.307557e-01 +/- 2.063908e-01 |
| Lid-driven Cavity | 10 | 3.478358e-01 +/- 2.739655e-02 | 6.383648e-01 +/- 4.657538e-02 |
| L-shaped Poisson | 10 | 1.328838e-01 +/- 3.998597e-02 | 8.797030e-02 +/- 1.257879e-02 |
| Advection-Diffusion | 10 | 2.715258e-01 +/- 2.373141e-03 | 5.274558e-01 +/- 3.087199e-03 |
| Burgers | 10 | 4.643369e-02 +/- 2.875759e-02 | 4.412386e-01 +/- 3.057048e-01 |
| Allen-Cahn | 10 | 9.911936e-01 +/- 1.364435e-03 | 1.008610e+00 +/- 6.729034e-03 |
| Diffusion-Reaction | 10 | 2.731571e-03 +/- 1.052873e-03 | 1.183547e-02 +/- 5.388714e-03 |
| Discontinuous Advection | 10 | 6.784173e-02 +/- 2.038310e-02 | 8.493749e-01 +/- 1.419234e-01 |

## Seed Note

All equations except 1D Poisson (Steep) use seeds 0 through 9. For 1D Poisson
(Steep), Seed 4 diverged deterministically in L-BFGS on both A100 and RTX 4090,
and replacement Seed 10 also diverged. The 10 finite runs therefore use seeds
0, 1, 2, 3, 5, 6, 7, 8, 9, and 11. The failed runs remain archived and were not
silently converted into finite metrics.

Per-seed values are available in `per_seed_metrics.md` and
`per_seed_metrics.csv` in this directory.
