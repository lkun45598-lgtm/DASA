# Experiment matrix

## Group A: Cost

- `configs/cost/cost_pinn.json`
- `configs/cost/cost_dasa.json`

Controlled variables: equation, domain, backbone, collocation schedule, boundary penalty, optimizer settings, number of backbone updates, seed, device.

Measured variables: trainable parameters, main-stage time per iteration, Adam total time, full total time, peak GPU memory, relative L2 error, maximum absolute error.

## Group B: Sensitivity

- `sens_baseline.json`
- `sens_arch_small_3x5.json`
- `sens_arch_large_7x20.json`
- `sens_lr_both_low.json`
- `sens_lr_subnet_low.json`
- `sens_lr_subnet_high.json`
- `sens_lr_both_high.json`
- `sens_freq_sparse_1to5.json`
- `sens_freq_dense_5to1.json`

All sensitivity runs stop after the adversarial stage. The baseline is run once and serves as the reference row for all three factors.

## Group C: Stage-length sensitivity

The nine configurations in `configs/stage_length_paper/` use the aligned gated Modified
MLP runner and ten seeds. The default schedule is `(5000, 2000, 10000, 10000)`. A single
stage is varied at a time using Stage 1 values `{2500, 5000, 10000}`, Stage 2 values
`{1000, 2000, 4000}`, Stage 3 values `{5000, 10000, 20000}`, and Stage 4 values
`{5000, 10000, 20000}`. The common default is executed once, giving 9 configurations and
90 raw runs. Stage 3 and Stage 4 use exponential learning-rate decay with `gamma=0.9`
every 2,000 iterations. To avoid coupling Stage 3 length to the Stage 4 optimization
budget, all variants start Stage 4 at the default-schedule learning rate `5.904900e-04`,
while retaining network parameters and Adam moment states. Every run includes the common
1,000-epoch L-BFGS refinement. Relative L2 and maximum pointwise error are evaluated after
L-BFGS. Baseline seeds 0 and 1 are retained as raw results but excluded from the reported
baseline aggregate as disclosed in `results_stage_length_paper_formal_merged/EXCLUSIONS.md`;
the baseline row therefore has 8 runs and every non-default row has 10.
