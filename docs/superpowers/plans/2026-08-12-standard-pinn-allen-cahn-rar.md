# Standard PINN Allen-Cahn RAR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run ten standard PINN Allen-Cahn seeds from the user-provided RAR with the correct diffusivity-`1e-4` spectral reference.

**Architecture:** Keep the RAR experiment standalone under `ac方程/` and preserve its solver. Make only launcher and aggregation changes needed for eight-GPU ten-seed execution; do not modify SA-PINN code.

**Tech Stack:** Python 3.10, DeepXDE PyTorch backend, NumPy, SciPy, tmux, eight A100 GPUs.

---

### Task 1: Materialize And Verify The RAR Experiment

**Files:**
- Create from archive: `ac方程/plain_pinn_baselines/*.py`
- Create from archive: `ac方程/reference_data/allen_cahn_diffusivity_0.0001_spectral_reference.npz`
- Create from archive: `ac方程/generate_allen_cahn_reference.py`
- Create from archive: `ac方程/tests/*.py`

- [ ] Extract only the standard PINN experiment files required for Allen-Cahn.
- [ ] Run the RAR structural tests with `DDE_BACKEND=pytorch`.
- [ ] Inspect the bundled reference and assert diffusivity `1e-4`, reaction `5`, shape `(101, 201)`, and exact initial field.

### Task 2: Configure Ten Seeds On Eight GPUs

**Files:**
- Modify: `ac方程/run_allen_cahn_1d.sh`
- Create: `ac方程/aggregate_allen_cahn.py`

- [ ] Change launcher defaults to GPUs `0..7`, seeds `0..9`, and allow at most eight workers.
- [ ] Keep `ADAM_ITERATIONS=300000`, `LBFGS_ITERATIONS=0`, checkpointing, and resume behavior unchanged.
- [ ] Add an aggregator that reads completed metrics, emits only relative L2 and maximum pointwise values, and computes population mean and standard deviation.
- [ ] Run shell syntax and aggregation smoke checks.

### Task 3: Deploy And Run

**Files:**
- Synchronize: `ac方程/` to `/home/lz/SA-PINNs/ac方程/` on `10.16.46.215`
- Generate remotely: `ac方程/runs/standard_pinn_ac_10seeds_20260812/`

- [ ] Verify the remote Python environment sees PyTorch CUDA and the RAR source/reference.
- [ ] Start a new tmux session with ten seeds distributed across GPUs `0..7`.
- [ ] Verify eight active seed processes and two queued seeds with no failures.
- [ ] Wait for all ten metrics files, aggregate both metrics, and report every seed without filtering finite runs.

No Git commit is included because this project is an untracked child of a large read-only parent repository; explicit test and artifact checkpoints replace commits.

