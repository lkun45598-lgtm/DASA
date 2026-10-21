# Wave-equation rebuttal experiments

This directory reproduces the three rebuttal tables on the wave equation:

1. PINN versus DASA computational cost (3 seeds).
2. DASA sub-network architecture, learning-rate, and update-ratio sensitivity (3 seeds).
3. Four-stage length sensitivity (10 seeds per schedule).

The experiment follows the supplied `wave equation(subnet).py` implementation:

- PDE: `u_tt - 4*u_xx = 0` on `(t, x) in [0, 1]^2`.
- Initial data: `u(0,x) = sin(pi*x) + 0.5*sin(4*pi*x)` and `u_t(0,x) = 0`.
- Boundary data: `u(t,0) = u(t,1) = 0`.
- Exact solution: `sin(pi*x)cos(2*pi*t) + 0.5*sin(4*pi*x)cos(8*pi*t)`.
- Backbone: the source's shared-weight 6 x 128 ST-MFF network.
- DASA sub-network: 5 x 10 by default.
- Mini-batches: 256 PDE, 128 initial, and 64 points per boundary side, resampled every 100 updates.
- Full L-BFGS batch: 25,600 PDE, 512 initial, and 256 points per boundary side.
- Default stages: `(5000, 2000, 50000, 50000)`, followed by the source's L-BFGS refinement.

The source sampled initial-condition coordinates from `[-1, 1]` while its PDE, boundary,
and evaluation grids use `[0, 1]`. The controlled runner uses `[0, 1]` consistently. The
DASA weights are normalized to batch mean one, matching Eq. (8), rather than retaining the
extra `+1` in the archived script.

PINN and DASA use exactly the same ST-MFF backbone and backbone update schedule. For the
stage-length table, each stage is varied by `0.5x` and `2x`; Stage 4 always starts at the
learning rate reached after the default 50,000-step Stage 3.

Run all experiments in tmux:

```bash
SESSION=wave_rebuttal_formal \
  bash run_tmux.sh start /absolute/path/to/results 0 1 2 3 4 5 6 7
```

Inspect progress and aggregate completed results:

```bash
bash run_tmux.sh status /absolute/path/to/results
bash run_tmux.sh aggregate /absolute/path/to/results
```

The final tables are written to `summaries/TABLES.md`; all per-seed values are retained in
`summaries/all_runs.csv`.
