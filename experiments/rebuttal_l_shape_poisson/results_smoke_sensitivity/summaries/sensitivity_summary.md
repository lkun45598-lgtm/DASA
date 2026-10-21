# DASA sensitivity on L-shaped Poisson

All errors are measured immediately after the adversarial/main stage, before unweighted fine-tuning and L-BFGS.

| Axis | Variant | Seeds | Subnet depth×width | Backbone LR | Subnet LR | Subnet:backbone updates | Params | Relative L2 | ΔL2 vs baseline | Max error | Total time (min) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | baseline_5x10_lr1e-3_ratio1to1 | 1 | 5x10 | 1.0e-03 | 1.0e-03 | 1:1 | 84321 | 6.6436 ± 0 | +0.00% | 0.59021 ± 0 | 0.01097 |
