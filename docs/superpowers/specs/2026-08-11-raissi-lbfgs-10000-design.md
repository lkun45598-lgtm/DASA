# Raissi PINN L-BFGS 10,000-Step Rerun Design

## Objective

Rerun all 13 classical vanilla PINN equations for seeds 0 through 9 with at
most 10,000 full-batch L-BFGS iterations and no Adam training.

## Scope

- Preserve the existing `raissi_lbfgs` default of 1,000 iterations.
- Allow an explicit L-BFGS iteration override while using that profile.
- Pass the override through the existing tmux seed-matrix launcher.
- Write all 130 new runs to a new directory on the A100 server.
- Keep the 6x128 tanh MLP, fixed samples, equal loss-component weights, PDE
  definitions, references, and two requested metrics unchanged.

## Verification

A focused trainer test must show that the Raissi profile records an explicit
L-BFGS override while retaining zero Adam steps and equal loss weights. The
launcher must pass `bash -n`, and the remote queue manifest must record 13
cases, 10 seeds, and a 10,000-step L-BFGS budget before training starts.

