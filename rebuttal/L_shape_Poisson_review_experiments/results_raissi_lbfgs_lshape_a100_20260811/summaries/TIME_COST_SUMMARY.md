# L-shaped Poisson: Raissi PINN vs DASA on A100

## Experimental Setup

- Hardware: NVIDIA A100-SXM4-80GB
- Seeds per method: 3 (0, 1, and 2)
- Interior candidates: 10,000
- Boundary points: 128 per boundary segment
- Reference points: 109,610
- Reported accuracy metrics: relative L2 error and maximum pointwise error
- PINN: standard 6x128 tanh MLP, fixed points, equal loss weights, no Adam,
  one full-batch L-BFGS call with `max_iter=1000`
- DASA: 6x128 modified MLP with a 5x10 weighting subnet, using the complete
  declared DASA Adam and L-BFGS training protocol

## Aggregate Results

Values are mean +/- sample standard deviation over three seeds.

| Method | Trainable parameters | Full training time (s) | Relative L2 error | Maximum pointwise error |
|---|---:|---:|---:|---:|
| Raissi PINN | 83,073 | 5.339876e+01 +/- 1.749068e+00 | 2.872746e-01 +/- 4.960774e-02 | 9.266253e-02 +/- 9.527306e-03 |
| DASA | 84,321 | 1.557376e+03 +/- 6.486532e+01 | 8.484838e-03 +/- 4.738401e-03 | 6.860370e-03 +/- 8.006973e-04 |

## Per-Seed Results

| Seed | PINN time (s) | DASA time (s) | PINN relative L2 | DASA relative L2 | PINN maximum error | DASA maximum error |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.226613e+01 | 1.501983e+03 | 3.219952e-01 | 1.055048e-02 | 9.965587e-02 | 5.946656e-03 |
| 1 | 5.541322e+01 | 1.541413e+03 | 2.304582e-01 | 1.183976e-02 | 8.181131e-02 | 7.194907e-03 |
| 2 | 5.251695e+01 | 1.628733e+03 | 3.093705e-01 | 3.064273e-03 | 9.652042e-02 | 7.439545e-03 |

## DASA Relative to Raissi PINN

| Quantity | Result |
|---|---:|
| Trainable parameter increase | 1.502300e+00% |
| End-to-end training-time ratio | 2.916503e+01x |
| End-to-end training-time increase | 2.816503e+03% |
| Relative L2 error reduction | 9.704644e+01% |
| Maximum pointwise error reduction | 9.259639e+01% |

## Interpretation

On this L-shaped Poisson problem, DASA reduces the mean relative L2 error by
9.704644e+01% and the mean maximum pointwise error by 9.259639e+01% relative to
the original Raissi-style PINN baseline. The mean DASA end-to-end training time
is 2.916503e+01 times the mean PINN training time.

The time ratio compares the complete declared training procedures: the PINN uses
only one L-BFGS call, while DASA uses its full multi-stage Adam and L-BFGS
protocol. It therefore measures end-to-end method cost and must not be described
as the isolated overhead of the DASA weighting subnet.

## Data Sources

- Raissi PINN: `results_raissi_lbfgs_lshape_a100_20260811`
- DASA: `results_cost_unitmean_full_seed0/cost_dasa`
