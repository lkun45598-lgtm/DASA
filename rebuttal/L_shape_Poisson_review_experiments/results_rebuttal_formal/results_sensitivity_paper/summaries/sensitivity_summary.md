# DASA sensitivity on L-shaped Poisson

All errors are measured immediately after the adversarial/main stage, before unweighted fine-tuning and L-BFGS.

| Axis | Variant | Seeds | Subnet depth×width | Backbone LR | Subnet LR | Subnet:backbone updates | Params | Relative L2 | ΔL2 vs baseline | Max error | Total time (min) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| architecture | arch_large_7x20 | 3 | 7x20 | 1.0e-03 | 1.0e-03 | 1:1 | 86441 | 0.039884 ± 0.00872 | +5.38% | 0.027754 ± 0.00368 | 12.506 |
| architecture | arch_small_3x5 | 3 | 3x5 | 1.0e-03 | 1.0e-03 | 1:1 | 83921 | 0.038958 ± 0.0076 | +2.93% | 0.02738 ± 0.00326 | 12.049 |
| learning_rate | lr_both_high | 3 | 5x10 | 2.0e-03 | 2.0e-03 | 1:1 | 84321 | 0.03531 ± 0.00819 | -6.71% | 0.025794 ± 0.0035 | 12.23 |
| learning_rate | lr_both_low | 3 | 5x10 | 5.0e-04 | 5.0e-04 | 1:1 | 84321 | 0.0426 ± 0.00153 | +12.56% | 0.02893 ± 0.000568 | 12.374 |
| learning_rate | lr_subnet_high | 3 | 5x10 | 1.0e-03 | 2.0e-03 | 1:1 | 84321 | 0.037273 ± 0.00616 | -1.52% | 0.026723 ± 0.00268 | 12.113 |
| learning_rate | lr_subnet_low | 3 | 5x10 | 1.0e-03 | 5.0e-04 | 1:1 | 84321 | 0.036611 ± 0.00638 | -3.27% | 0.026448 ± 0.00273 | 12.444 |
| update_frequency | freq_dense_5to1 | 3 | 5x10 | 1.0e-03 | 1.0e-03 | 5:1 | 84321 | 0.040987 ± 0.00915 | +8.29% | 0.028158 ± 0.00383 | 28.76 |
| update_frequency | freq_sparse_1to5 | 3 | 5x10 | 1.0e-03 | 1.0e-03 | 1:5 | 84321 | 0.03756 ± 0.00832 | -0.76% | 0.026706 ± 0.00353 | 12.2 |
| baseline | baseline_5x10_lr1e-3_ratio1to1 | 3 | 5x10 | 1.0e-03 | 1.0e-03 | 1:1 | 84321 | 0.037848 ± 0.00689 | +0.00% | 0.026913 ± 0.00297 | 12.012 |
