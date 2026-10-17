# Raissi-Style PINN Rerun Design

## Goal

Rerun the L-shaped Poisson rebuttal baseline and all 13 DASA-aligned forward PDE
baselines using the classical PINN training interpretation approved by the user.
Existing outputs remain untouched.

## Effective PINN Definition

- Vanilla fully connected MLP with six hidden layers of width 128.
- `tanh` activation, Xavier-normal weights, trainable biases.
- No Modified MLP gates, Fourier embedding, adaptive masks, weighting subnet, or
  adversarial updates.
- Fixed collocation, boundary, and initial points for the full training run.
- Every PDE, boundary, initial, observation, and gauge MSE component has weight 1.
- No Adam phase.
- One full-batch L-BFGS optimization capped at 1,000 total iterations.
- Report only relative L2 error and maximum pointwise error as accuracy metrics.

The Raissi paper has no L-shaped Poisson configuration and does not prescribe one
universal network size. The 6x128 capacity therefore follows the user's previously
approved common comparison capacity, while the loss and optimization procedure follow
the paper's classical forward-PINN formulation.

## Experiment Scope

1. Run the standalone L-shaped rebuttal experiment for seeds 0, 1, and 2 using
   10,000 LHS candidates, 128 boundary points per segment, equal PDE/boundary
   weights, and a single L-BFGS call with `max_iter=1000`.
2. Run all 13 DeepXDE cases for seeds 0 through 9. Preserve each equation's approved
   domain, coefficients, reference solution, sampling counts, and sampling layout;
   only replace the optimizer/loss-weight profile with the definition above.
3. Split the 10-seed matrix across the local 8x RTX 4090 server and the remote 8x
   A100 server. Use separate result roots and merge them only after completion.
4. Allen-Cahn runs have exclusive GPU assignment: one Allen-Cahn seed per physical
   GPU with no second process on that card. Other PDEs may use two workers per GPU.
5. Launch every worker in a background `tmux` session and preserve enough run metadata
   to identify the host and GPU used by each result.

## Implementation Boundary

- Add one standalone L-shaped JSON configuration.
- Add an explicit Raissi training profile to the DeepXDE runner and matrix launcher.
- Record the effective profile and loss weights in each resolved configuration.
- Do not edit DASA, SA-PINN, PDE definitions, reference data, or existing result roots.

## Verification

- Perform one reduced single-case invocation to verify that the resolved run records
  `adam_steps=0`, `lbfgs_steps=1`, and unit loss weights without starting a long run.
- After launch, verify the new tmux sessions, process command lines, output roots, and
  GPU utilization.
- Aggregate the 10-seed matrix only after all 130 runs finish.

## Risks

- Some equations may converge poorly with L-BFGS-only training; that outcome is part of
  this baseline and must not be hidden by adding Adam or adaptive weighting.
- The previous interrupted A100 outputs are retained but must not be mixed with the new
  result roots.
