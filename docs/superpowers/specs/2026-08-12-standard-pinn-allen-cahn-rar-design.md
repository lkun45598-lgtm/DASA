# Standard PINN Allen-Cahn RAR Design

## Goal

Run ten standard PINN Allen-Cahn seeds from `ac方程.rar` without changing any
SA-PINN implementation or result.

## Experiment Definition

The RAR implementation is the source of truth:

- Equation: `u_t - 1e-4 u_xx + 5 u^3 - 5 u = 0` on `[0,1] x [-1,1]`.
- Initial condition: `u(0,x) = x^2 cos(pi x)`.
- Periodicity is hard-enforced by the order-8 feature map
  `[t, cos(k pi x), sin(k pi x)]`, `k=1,...,8`.
- Network: ordinary fully connected `6 x 128 tanh` FNN in float64, without
  modified-MLP gates, SA masks, or a DASA subnetwork.
- Sampling: 25,600 fixed LHS residual points and 512 fixed LHS initial points.
  There are no explicit boundary points because the Fourier map enforces both
  periodic value and derivative conditions.
- Loss weights: residual `1`, initial condition `100`.
- Optimization: 300,000 Adam iterations with initial learning rate `1e-3`,
  step decay `0.9` every 2,000 steps, and no L-BFGS.
- Truth: the bundled float64 periodic Fourier pseudospectral reference for
  diffusivity `1e-4`, not `Allen-Cahn/AC.mat`.
- Metrics: relative L2 error and maximum pointwise error only.

## Isolation And Execution

Extract only the source, test, launcher, and reference files required by this
experiment into the standalone `ac方程/` directory. Preserve the RAR solver
logic. Extend its launcher only to accept eight GPUs and default to seeds
`0..9`. Launch it inside a new tmux session on `10.16.46.215`; each seed keeps
independent checkpoints and artifacts so interrupted training can resume.

## Validation

Before the full run, verify the equation coefficients, initial field, reference
metadata, order-8 feature dimension, plain-FNN architecture, sampling counts,
loss weights, and 300,000-step budget. Report all ten finite seeds and the
population mean plus standard deviation in scientific notation. Do not exclude
or replace a finite result merely because its error is outside `1e-2`.

