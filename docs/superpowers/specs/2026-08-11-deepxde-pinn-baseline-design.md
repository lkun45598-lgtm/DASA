# DeepXDE Classical PINN Baseline Design

## Goal

Implement and run a standard PINN baseline with DeepXDE for the 13 forward PDE cases in the DASA comparison. The baseline must isolate the point-weighting method: it uses the same solution backbone, equations, training points, loss coefficients, optimizer settings, and training budget as DASA, but has no DASA subnet, trainable point masks, or adversarial updates.

## Scope

- Cases: the 13 files in `benchmarks/dasa_aligned_sa/configs/cases/`.
- Seeds: 0 through 9.
- Metrics: relative L2 error and maximum pointwise absolute error.
- Backend: DeepXDE with PyTorch.
- Execution hardware: the current eight-GPU RTX 4090 server; record the GPU and software versions with every run matrix.
- Output: per-run config, progress log, model checkpoint, and a `metrics.json` containing only the case, seed, and two requested metrics; aggregate CSV/Markdown tables use scientific notation.

The implementation will not copy or depend on the external DeepXDE Poisson example mentioned by the user.

## Controlled Comparison

The DASA paper states that classical PINN and DASA use the same architecture, hyperparameters, training points, and optimization method. Therefore the baseline uses:

- six hidden layers of width 128;
- the gated Modified MLP backbone;
- `tanh` activation and Xavier-normal initialization;
- the configured periodic Fourier hard embedding where applicable;
- the configured residual, boundary, and initial sample counts;
- fixed loss coefficients from each case config;
- the configured Adam learning rate and staircase decay, followed by the configured L-BFGS budget.

The only method difference is that the classical PINN minimizes the fixed loss

\[
\mathcal L = \lambda_r\mathcal L_r + \lambda_{bc}\mathcal L_{bc}
             + \lambda_{ic}\mathcal L_{ic}.
\]

## Architecture

Create `benchmarks/deepxde_pinn/` as an independent benchmark package. It reads the existing strict case configurations but implements PDE residuals and constraints with DeepXDE/PyTorch, avoiding TensorFlow imports from the SA-PINN trainer.

- `network.py`: DeepXDE-compatible gated Modified MLP and periodic embeddings.
- `problems/`: shared problem interface plus separate static, time-dependent, cavity, and L-shaped implementations for geometry, PDE residuals, BC/IC construction, deterministic sampling, and evaluation references.
- `trainer.py`: seed control, Adam schedule, optional L-BFGS, metrics, and artifacts.
- `scripts/run_case.py`: one-case CLI with smoke budget overrides.
- `scripts/launch_seed_matrix_tmux.sh`: atomic queue for 13 cases by 10 seeds.
- `scripts/aggregate_results.py`: two-metric scientific-notation summaries.

## Sampling And Constraints

Use DeepXDE geometry/data objects and deterministic seeds. Counts are interpreted the same way as the aligned DASA configuration: time-dependent boundary counts are per spatial side, and 2D edge counts are expanded per edge where the case definition requires it. The L-shaped domain uses the same candidate-and-filter convention as the DASA experiment rather than silently increasing the accepted interior count.

Periodic value and derivative constraints remain explicit in the DeepXDE data object even when the Fourier input embedding satisfies them by construction. This preserves the configured loss definition and exposes constraint errors in logs.

## Training

Use full-batch Adam for the configured total number of steps. A callback updates the learning rate without resetting Adam state: it remains constant through the first two configured stage budgets and decays by the configured factor at the configured interval afterward. If a case config requests L-BFGS, DeepXDE's PyTorch L-BFGS phase runs after Adam with the configured iteration limit.

## Validation

1. Unit-test configuration coverage, Modified MLP dimensions, periodic endpoint identity, PDE residuals on known exact solutions, metric formulas, and queue size.
2. Run all 13 cases with a tiny CPU budget.
3. Run one tiny local RTX 4090 CUDA case before launching the full matrix.
4. Launch seeds 0-9 in tmux and monitor completed, running, pending, and failed counts.

## Known Scientific Limitations

- `helmholtz_1d` is currently homogeneous and amplitude-underdetermined. It will be implemented exactly as configured, but its error is not reportable as a meaningful method comparison until an amplitude-fixing condition is confirmed and applied consistently to every method.
- `advection_diffusion` currently uses `kappa = 0.01`, marked as inferred because the DASA paper does not state it explicitly.
- The original 2019 PINN paper defines the standard fixed-loss method but does not provide a universal architecture or schedule for these 13 DASA cases; those controlled settings come from the DASA paper and the approved aligned configs.
