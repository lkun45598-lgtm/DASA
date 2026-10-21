# Stage-length sensitivity experiment

Goal: address Reviewer 2's concern that the iteration counts of the four DASA stages affect the result but were not justified.

Scope: reuse the aligned DASA runner in `L_SHAPED_METHODS_FOR_GPT/refactored_dasa_pinn/src/lshape_experiment.py`; add one-factor-at-a-time configurations for Stage 1--4, retaining the L-shaped Poisson PDE, 6x128 gated Modified MLP, 5x10 weighting subnet, deterministic LHS sampling, Adam learning rates, and three seeds. L-BFGS remains disabled.

The default schedule is `(5000, 2000, 10000, 10000)`. Each sensitivity axis has low/default/high values while all other stage lengths remain fixed. The aggregate reports only relative L2 error and maximum pointwise error measured after Stage 4.

Verification: run a tiny CPU smoke configuration, validate JSON/CSV/Markdown generation, then launch all 27 formal runs through the A100 tmux launcher.
