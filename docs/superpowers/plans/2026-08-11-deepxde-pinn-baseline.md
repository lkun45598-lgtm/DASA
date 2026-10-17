# DeepXDE Classical PINN Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a DeepXDE/PyTorch classical PINN baseline for all 13 DASA forward PDE cases over seeds 0-9.

**Architecture:** A new `benchmarks/deepxde_pinn` package reads the approved DASA-aligned case configs, implements DeepXDE problem objects without importing the TensorFlow trainer, and uses a DeepXDE-compatible gated Modified MLP. One-case execution writes deterministic artifacts; shell launchers run smoke and 130-run tmux matrices.

**Tech Stack:** Python 3.10, DeepXDE 1.15.0, PyTorch 2.5.1/CUDA 12.1, NumPy, SciPy, pytest, tmux, eight local RTX 4090 GPUs.

---

### Task 1: Dependency And Package Skeleton

**Files:**
- Create: `benchmarks/deepxde_pinn/__init__.py`
- Create: `benchmarks/deepxde_pinn/requirements.txt`
- Create: `benchmarks/deepxde_pinn/tests/__init__.py`
- Create: `benchmarks/deepxde_pinn/tests/test_environment.py`

- [ ] **Step 1: Write the failing environment test**

```python
def test_deepxde_uses_pytorch_backend():
    import deepxde as dde
    assert dde.backend.backend_name == "pytorch"
```

- [ ] **Step 2: Verify the test fails because DeepXDE is absent**

Run: `DDE_BACKEND=pytorch /home/lz/miniconda3/envs/pytorch/bin/python -m pytest benchmarks/deepxde_pinn/tests/test_environment.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'deepxde'`.

- [ ] **Step 3: Pin and install the dependency**

`requirements.txt` contains `deepxde==1.15.0`. Install it into the existing PyTorch environment without modifying PyTorch:

```bash
/home/lz/miniconda3/envs/pytorch/bin/python -m pip install --no-deps deepxde==1.15.0
```

- [ ] **Step 4: Verify backend selection**

Run the Step 2 command again. Expected: PASS and backend `pytorch`.

### Task 2: Metrics And Gated Modified MLP

**Files:**
- Create: `benchmarks/deepxde_pinn/metrics.py`
- Create: `benchmarks/deepxde_pinn/network.py`
- Create: `benchmarks/deepxde_pinn/tests/test_metrics.py`
- Create: `benchmarks/deepxde_pinn/tests/test_network.py`

- [ ] **Step 1: Write failing metric and network tests**

Tests assert the exact requested formulas, a six-layer width-128 network, scalar and cavity output shapes, and endpoint identity for `periodic_unit` and `periodic_pi` embeddings.

```python
np.testing.assert_allclose(relative_l2(y, p), np.linalg.norm(y - p) / np.linalg.norm(y))
assert maximum_pointwise(y, p) == np.max(np.abs(y - p))
assert torch.allclose(model(torch.tensor([[0.4, -1.0]])), model(torch.tensor([[0.4, 1.0]])), atol=1e-6)
```

- [ ] **Step 2: Run tests and verify missing-module failures**

Run: `DDE_BACKEND=pytorch /home/lz/miniconda3/envs/pytorch/bin/python -m pytest benchmarks/deepxde_pinn/tests/test_metrics.py benchmarks/deepxde_pinn/tests/test_network.py -q`

- [ ] **Step 3: Implement minimal metrics and network**

`ModifiedMLP` subclasses DeepXDE's PyTorch `NN`, normalizes inputs, applies the configured Fourier embedding, builds U/V encoders plus six gated hidden transformations, and initializes every linear kernel with Xavier normal.

- [ ] **Step 4: Run the tests and verify PASS**

Expected: all metric/network tests pass with no backend fallback.

### Task 3: Shared Problem Interface And Static PDEs

**Files:**
- Create: `benchmarks/deepxde_pinn/problems/__init__.py`
- Create: `benchmarks/deepxde_pinn/problems/base.py`
- Create: `benchmarks/deepxde_pinn/problems/static.py`
- Create: `benchmarks/deepxde_pinn/tests/test_static_problems.py`

- [ ] **Step 1: Write exact-solution residual tests**

Cover `poisson_1d_boundary`, `poisson_1d_steep`, `helmholtz_1d`, `poisson_2d_peak`, `poisson_2d_steep`, and `helmholtz_2d`. Tests evaluate each DeepXDE residual on an analytic Torch model and require maximum residual below `1e-4` at interior points.

- [ ] **Step 2: Verify failures before implementation**

Run: `DDE_BACKEND=pytorch /home/lz/miniconda3/envs/pytorch/bin/python -m pytest benchmarks/deepxde_pinn/tests/test_static_problems.py -q`.

- [ ] **Step 3: Implement static geometry, residuals, BCs, and evaluation grids**

Use the existing strict case configs. Expand per-edge boundary counts where required, preserve uniform/LHS sampling, and keep the underdetermined homogeneous 1D Helmholtz case unchanged and explicitly flagged in resolved metadata.

- [ ] **Step 4: Verify all static residual and count tests pass**

### Task 4: Time-Dependent PDEs

**Files:**
- Create: `benchmarks/deepxde_pinn/problems/time.py`
- Create: `benchmarks/deepxde_pinn/tests/test_time_problems.py`

- [ ] **Step 1: Write failing tests for five time PDEs**

Cover Burgers, Allen-Cahn, diffusion-reaction, advection-diffusion, and discontinuous advection. Assert PDE formulas, IC values, per-side boundary count interpretation, periodic value/derivative constraints, and reference-data shapes.

- [ ] **Step 2: Verify failures before implementation**

Run: `DDE_BACKEND=pytorch /home/lz/miniconda3/envs/pytorch/bin/python -m pytest benchmarks/deepxde_pinn/tests/test_time_problems.py -q`.

- [ ] **Step 3: Implement DeepXDE `GeometryXTime`, PDE callbacks, BC/IC objects, and evaluators**

Burgers and Allen-Cahn load the existing MAT references. ADE uses `kappa=0.01` and records its inferred provenance. Fourier embeddings come only from the approved config.

- [ ] **Step 4: Run tests and verify PASS**

### Task 5: Cavity And L-Shaped Poisson

**Files:**
- Create: `benchmarks/deepxde_pinn/problems/cavity.py`
- Create: `benchmarks/deepxde_pinn/problems/lshape.py`
- Create: `benchmarks/deepxde_pinn/tests/test_irregular_problems.py`

- [ ] **Step 1: Write failing tests for residual dimensions, boundary construction, and references**

The cavity residual has momentum-x, momentum-y, and continuity components plus one pressure gauge constraint. L-shape candidate sampling deletes the upper-right quadrant and uses eight boundary segments.

- [ ] **Step 2: Verify failures before implementation**

Run: `DDE_BACKEND=pytorch /home/lz/miniconda3/envs/pytorch/bin/python -m pytest benchmarks/deepxde_pinn/tests/test_irregular_problems.py -q`.

- [ ] **Step 3: Implement both problems**

Use the vendored cavity reference and generated L-shaped NPZ reference already used by the aligned suite. Do not generate a new reference during training.

- [ ] **Step 4: Run tests and verify PASS**

### Task 6: Trainer, CLI, And Artifacts

**Files:**
- Create: `benchmarks/deepxde_pinn/artifacts.py`
- Create: `benchmarks/deepxde_pinn/trainer.py`
- Create: `benchmarks/deepxde_pinn/scripts/__init__.py`
- Create: `benchmarks/deepxde_pinn/scripts/run_case.py`
- Create: `benchmarks/deepxde_pinn/tests/test_trainer.py`

- [ ] **Step 1: Write a failing one-step training test**

Use a small static case and assert `config_resolved.json`, `progress.jsonl`, checkpoint, and `metrics.json`. Assert the metrics file has exactly `case`, `seed`, `l2_relative_error`, and `maximum_pointwise_error`.

- [ ] **Step 2: Verify failure before implementation**

Run: `DDE_BACKEND=pytorch /home/lz/miniconda3/envs/pytorch/bin/python -m pytest benchmarks/deepxde_pinn/tests/test_trainer.py -q`.

- [ ] **Step 3: Implement deterministic training**

Set DeepXDE/PyTorch/NumPy seeds, train full-batch Adam, update the staircase learning rate through a callback without resetting Adam state, optionally run DeepXDE L-BFGS, evaluate in batches, and write failures with tracebacks.

- [ ] **Step 4: Verify the one-step run and full unit suite**

Run: `DDE_BACKEND=pytorch /home/lz/miniconda3/envs/pytorch/bin/python -m pytest benchmarks/deepxde_pinn/tests -q`.

### Task 7: Aggregation And Tmux Orchestration

**Files:**
- Create: `benchmarks/deepxde_pinn/scripts/aggregate_results.py`
- Create: `benchmarks/deepxde_pinn/scripts/launch_seed_matrix_tmux.sh`
- Create: `benchmarks/deepxde_pinn/scripts/run_smoke_matrix.sh`
- Create: `benchmarks/deepxde_pinn/tests/test_orchestration.py`
- Create: `benchmarks/deepxde_pinn/README.md`

- [ ] **Step 1: Write failing queue and aggregation tests**

Assert 130 queue entries for 13 cases by 10 seeds, atomic claim states, explicit GPU lists, resume behavior, and scientific notation in CSV/Markdown summaries.

- [ ] **Step 2: Verify failures before implementation**

Run: `DDE_BACKEND=pytorch /home/lz/miniconda3/envs/pytorch/bin/python -m pytest benchmarks/deepxde_pinn/tests/test_orchestration.py -q`.

- [ ] **Step 3: Implement the minimal launchers and aggregator**

Use one worker per GPU, independent logs, `.pending/.running/.done/.failed` queue states, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and a finalizer that aggregates only after the queue drains.

- [ ] **Step 4: Verify shell syntax and tests**

Run: `bash -n benchmarks/deepxde_pinn/scripts/*.sh` and the orchestration test.

### Task 8: End-To-End Validation And Full Launch On The Local RTX 4090 Server

**Files:**
- Runtime outputs only under `benchmarks/deepxde_pinn/runs/` locally and the mirrored remote path.

- [ ] **Step 1: Run all 13 cases with tiny CPU budgets**

Expected: 13/13 metrics files, zero failures, finite two-metric summary.

- [ ] **Step 2: Run one tiny local RTX 4090 case**

Expected: DeepXDE reports PyTorch backend, the intended GPU is used, and both metrics are finite.

- [ ] **Step 3: Refresh local RTX 4090 occupancy and select idle GPUs**

Do not share a GPU with another training process. Record `nvidia-smi`, DeepXDE, PyTorch, and CUDA versions in the run root before launch.

- [ ] **Step 4: Launch the 130 runs in local tmux**

Use all genuinely idle GPUs, report the session name, output directory, completed/running/pending/failed counts, and an ETA based on observed iterations.

## Execution Note

No automatic git commits are included because `/data1/user/lz/SA-PINNs` is an untracked child of a large dirty parent repository; committing would mix unrelated workspace state. All edits remain confined to `benchmarks/deepxde_pinn/` and the two design/plan documents.
