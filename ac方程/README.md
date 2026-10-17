# Original/Plain FNN PINN Baselines

This project implements DeepXDE PINN baselines for the PDE experiments in the DASA-PINNs
paper using the ordinary fully connected network family associated with the original PINNs
formulation.

## Interpretation

This is an architecture-control project:

- PDEs and experimental settings follow DASA-PINNs and the confirmed project settings.
- The backbone is an ordinary `tanh` FNN with 6 hidden layers and width 128.
- There are no modified-MLP encoders or gates.
- There is no DASA sub-network, pointwise adaptive weighting, or adversarial training.

This is not a literal reproduction of every script in `260811_PINNs-master`. For example,
that repository solves Allen-Cahn with a discrete-time IRK model, whereas this project solves
the complete DASA Allen-Cahn space-time problem with a continuous PINN and reports global
space-time errors.

Source locations:

```text
Original PINNs repository:
/data/ccy/260809_补实验_pinn/260811_PINNs-master

DASA-PINNs paper:
/data/ccy/260809_补实验_pinn/2. DASA_PINNs (Preprint submitted to Neurocomputing).pdf
```

## Layout

```text
260811_original_pinn_baselines/
|-- plain_pinn_baselines/
|   |-- common.py
|   |-- poisson_1d.py
|   |-- helmholtz_1d.py
|   |-- poisson_2d.py
|   |-- helmholtz_2d.py
|   |-- lid_driven_cavity.py
|   |-- burgers_1d.py
|   |-- allen_cahn_1d.py
|   |-- diffusion_reaction_1d.py
|   |-- advection_diffusion_1d.py
|   |-- advection_discontinuous_1d.py
|   `-- poisson_l_shaped.py
|-- reference_data/
|-- tests/
|-- run_*.sh
|-- EXPERIMENT_SETUP.md
`-- README.md
```

The vendored DeepXDE checkout is loaded from the parent project:

```text
/data/ccy/260809_补实验_pinn/deepxde
```

## Network

The shared `PlainFNN` subclasses DeepXDE's standard PyTorch `FNN` without overriding its
forward pass. For scalar two-coordinate problems its layer list is:

```text
[2, 128, 128, 128, 128, 128, 128, 1]
```

Each `metrics.json` records:

```text
type = plain_fnn
version = plain_fnn_original_pinn
hidden_layer_count = 6
width = 128
parameter_count = actual parameter count
gated_modified_mlp = false
```

## Running

Enter the project directory:

```bash
cd /data/ccy/260809_补实验_pinn/260811_original_pinn_baselines
```

Run an experiment with its launcher:

```bash
./run_poisson_1d.sh
./run_helmholtz_1d.sh
./run_poisson_2d.sh
./run_helmholtz_2d.sh
./run_lid_driven_cavity.sh
./run_burgers_1d.sh
./run_allen_cahn_1d.sh
./run_diffusion_reaction_1d.sh
./run_advection_diffusion_1d.sh
./run_advection_discontinuous_1d.sh
./run_poisson_l_shaped.sh
```

The launchers accept environment overrides such as:

```bash
SEEDS="2040 2041 2042" \
CUDA_DEVICE=0 \
OUTPUT_DIR=runs/manual/poisson_2d \
./run_poisson_2d.sh
```

`run_poisson_1d.sh` and `run_poisson_2d.sh` run both of their cases by default. Pass a case
name to run only one:

```bash
./run_poisson_1d.sh boundary-layer
./run_poisson_1d.sh steep
./run_poisson_2d.sh peak
./run_poisson_2d.sh steep
```

Allen-Cahn and Burgers distribute seeds across the GPU IDs in `CUDA_DEVICES`. For example:

```bash
CUDA_DEVICES="0 1 2" SEEDS="2024 2025 2026" ./run_burgers_1d.sh
```

Each seed is an independent single-GPU training process. Run these launchers separately
from other jobs that use the same GPUs.

Allen-Cahn saves a restart checkpoint every 10000 Adam steps and resumes automatically
from the latest valid checkpoint for each seed. `ADAM_ITERATIONS` is the total target,
not the number of additional steps after resuming. For example, a checkpoint at step
120000 with `ADAM_ITERATIONS=300000` runs 180000 additional Adam steps.

```bash
# Default: resume when a checkpoint exists, otherwise start from step 0.
RESUME_MODE=auto CHECKPOINT_EVERY=10000 ./run_allen_cahn_1d.sh

# Require every selected seed to have a checkpoint.
RESUME_MODE=required ./run_allen_cahn_1d.sh

# Start a fresh run. Use a new output directory to protect existing checkpoints.
RESUME_MODE=never \
OUTPUT_DIR=runs/multiseed_plain_fnn/allen_cahn_1d_fresh \
./run_allen_cahn_1d.sh
```

Checkpoints are isolated at `seed_<seed>/checkpoints/adam_step_<step>.pt`. They include
the network, Adam optimizer, learning-rate scheduler, RNG states, completed step, and a
configuration signature. A mismatched seed, network, equation, point count, or optimizer
configuration is rejected instead of being restored silently.

## Outputs

Default launcher outputs are isolated under:

```text
runs/multiseed_plain_fnn/<experiment>/seed_<seed>/<equation>/<case>/
```

Each completed run writes:

```text
metrics.json
predictions.npz
```

The two required metrics are:

```text
l2_relative_error = ||y_true - y_pred||_2 / ||y_true||_2
max_pointwise_error = max(abs(y_true - y_pred))
```

Reference fields are copied into this project so runs do not depend on result files from the
other baseline directory. Detailed assumptions are recorded in `EXPERIMENT_SETUP.md` and
in every `metrics.json`.

## Verification

Run the structural tests with:

```bash
PYTHONPATH=. DDE_BACKEND=pytorch python -m unittest discover -s tests -v
```
