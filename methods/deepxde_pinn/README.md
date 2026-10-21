# DeepXDE Classical PINN Baseline

This package runs the classical fixed-loss PINN baseline for the 13 DASA-aligned
forward PDE cases. Following Raissi, Perdikaris, and Karniadakis (JCP 2019), it
uses a standard fully connected network with six 128-neuron hidden layers,
`tanh` activations, Xavier-normal initialization, and normalized raw coordinate
inputs. It does not use gated Modified MLP branches, Fourier hard embeddings,
SA-PINN point masks, or DASA adversarial subnetworks. Periodic conditions are
enforced by the corresponding boundary losses.

The only reported metrics are relative L2 error and maximum pointwise error.

Run one case:

```bash
DDE_BACKEND=pytorch MPLCONFIGDIR=/tmp/matplotlib-deepxde \
/home/lz/miniconda3/envs/pytorch/bin/python \
  -m methods.deepxde_pinn.scripts.run_case \
  --case burgers --seed 0 --output methods/deepxde_pinn/runs/example
```

Launch seeds 0-9 on selected GPUs:

```bash
bash methods/deepxde_pinn/scripts/launch_seed_matrix_tmux.sh \
  methods/deepxde_pinn/runs/seeds_0_9 0 1 2 3 4 5 6 7
```

Use `--resume` before the run directory to requeue stale `.running` tasks after
the tmux session has stopped.
