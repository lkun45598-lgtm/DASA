# L-shaped Poisson: Raissi PINN Training-Time Summary

## Experimental Setup

- Method: standard PINN
- Network: 6 hidden layers, 128 neurons per layer, `tanh`
- Initialization: Xavier normal
- Interior points: 10,000 fixed LHS points
- Boundary points: 128 points per boundary segment
- Loss weighting: PDE and boundary components are equally weighted
- Optimizer: no Adam; one full-batch L-BFGS call with `max_iter=1000`
- Hardware: NVIDIA GeForce RTX 4090 (GPU 3)
- Reference points: 109,610
- Seeds: 0, 1, and 2

## Aggregate Results

Values are mean +/- sample standard deviation over three seeds.

| Training time (s) | Relative L2 error | Maximum pointwise error |
|---:|---:|---:|
| 6.612697e+01 +/- 3.719903e+00 | 3.234626e-01 +/- 1.025468e-01 | 1.001637e-01 +/- 2.032220e-02 |

## Per-Seed Results

| Seed | Training time (s) | Relative L2 error | Maximum pointwise error |
|---:|---:|---:|---:|
| 0 | 6.974296e+01 | 4.219058e-01 | 1.193740e-01 |
| 1 | 6.632675e+01 | 2.172538e-01 | 7.888746e-02 |
| 2 | 6.231120e+01 | 3.312282e-01 | 1.022296e-01 |

## Timing Scope

`training_total_s` is the complete optimizer time recorded by the experiment.
Because this profile has no Adam stage, it is also the time for the single
full-batch L-BFGS call. Model evaluation and result serialization are excluded.

## Comparability Note

These runs shared GPU 3 with another PINN worker. The values therefore describe
the observed runtime of this run, but they are not an isolated hardware benchmark.
They must not be directly combined with the previous A100 DASA timing results to
claim a PINN-versus-DASA runtime ratio. A strict cost comparison requires rerunning
both methods on the same dedicated GPU under their declared training protocols.
