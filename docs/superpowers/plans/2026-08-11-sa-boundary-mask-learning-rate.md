# SA-PINN Boundary-Layer Mask Learning Rate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independently configurable SA-mask Adam learning rate and use `1e-5` only for the 1D Poisson boundary-layer benchmark.

**Architecture:** Extend the strict optimizer configuration with an optional `mask_learning_rate` that defaults to the solution-network learning rate. Keep the existing network learning-rate schedule and apply its multiplicative decay factor to the independent mask base rate.

**Tech Stack:** Python 3.10, TensorFlow 2.15.1, standard-library `unittest`, tmux, A100 GPUs.

---

### Task 1: Specify Independent Mask Learning Rate

**Files:**
- Modify: `benchmarks/dasa_aligned_sa/tests/test_config.py`
- Modify: `benchmarks/dasa_aligned_sa/tests/test_trainer.py`

- [ ] **Step 1: Write failing configuration assertions**

Add assertions that `poisson_1d_boundary` resolves `mask_learning_rate == 1e-5` and a case without an override resolves `mask_learning_rate == learning_rate`.

```python
boundary = load_case_config("poisson_1d_boundary")
self.assertEqual(boundary.optimizer.mask_learning_rate, 1.0e-5)
helmholtz = load_case_config("helmholtz_1d")
self.assertEqual(
    helmholtz.optimizer.mask_learning_rate,
    helmholtz.optimizer.learning_rate,
)
```

- [ ] **Step 2: Write a failing decay-ratio assertion**

Import `mask_learning_rate_for_step` beside the existing network helper and require the initial ratio to remain constant after decay.

```python
config = load_case_config("poisson_1d_boundary").optimizer
for step in (0, 3500, 41500):
    self.assertAlmostEqual(
        mask_learning_rate_for_step(config, step)
        / learning_rate_for_step(config, step),
        0.01,
    )
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
.conda-env/bin/python -m unittest \
  benchmarks.dasa_aligned_sa.tests.test_config.ConfigurationTests.test_resolves_the_approved_coefficients \
  benchmarks.dasa_aligned_sa.tests.test_trainer -v
```

Expected: failure because `OptimizerConfig.mask_learning_rate` and `mask_learning_rate_for_step` do not exist.

### Task 2: Implement Independent Mask Learning Rate

**Files:**
- Modify: `benchmarks/dasa_aligned_sa/dasa_sa/config.py`
- Modify: `benchmarks/dasa_aligned_sa/dasa_sa/trainer.py`
- Modify: `benchmarks/dasa_aligned_sa/configs/cases/poisson_1d_boundary.yaml`

- [ ] **Step 1: Extend strict optimizer loading compatibly**

Add `mask_learning_rate: float` to `OptimizerConfig`. Accept `mask_learning_rate` as the only optional optimizer key and resolve an omitted value to `learning_rate`.

```python
mask_learning_rate=float(
    optimizer_data.get("mask_learning_rate", optimizer_data["learning_rate"])
)
```

Reject any other unknown optimizer key and require both learning rates to be positive.

- [ ] **Step 2: Add the boundary-layer override**

Set only the boundary-layer optimizer field:

```json
"mask_learning_rate": 0.00001
```

Do not edit the other 12 case files.

- [ ] **Step 3: Preserve the decay ratio in the trainer**

Implement:

```python
def mask_learning_rate_for_step(config: OptimizerConfig, step: int) -> float:
    network_rate = learning_rate_for_step(config, step)
    return network_rate * config.mask_learning_rate / config.learning_rate
```

Initialize `mask_optimizer` with `config.optimizer.mask_learning_rate` and assign `mask_learning_rate_for_step(...)` inside the Adam loop.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Task 1 Step 3. Expected: all focused tests pass.

- [ ] **Step 5: Run the relevant suite**

Run:

```bash
.conda-env/bin/python -m unittest \
  benchmarks.dasa_aligned_sa.tests.test_config \
  benchmarks.dasa_aligned_sa.tests.test_trainer \
  benchmarks.dasa_aligned_sa.tests.test_sa_weights \
  benchmarks.dasa_aligned_sa.tests.test_static_problems -v
```

Expected: all tests pass with no failures.

### Task 3: Deploy And Run Ten Seeds

**Files:**
- Synchronize the three modified production files to `/home/lz/SA-PINNs` on `10.16.46.215`
- Generate: `benchmarks/dasa_aligned_sa/runs/poisson_1d_boundary_mask_lr_1e-5_a100_20260811/`

- [ ] **Step 1: Verify remote resolved values**

Load `poisson_1d_boundary` and `helmholtz_1d` in the remote configured environment. Expected mask learning rates: `1e-5` and `1e-3` respectively.

- [ ] **Step 2: Launch ten seeds in tmux**

Use the existing seed-matrix launcher with case `poisson_1d_boundary`, seeds `0..9`, and GPUs `0..7`. Use a new session and run root; preserve all previous runs.

- [ ] **Step 3: Verify all workers start**

Expected initial state: eight `.running` tasks, two `.pending` tasks, and no failure artifacts.

- [ ] **Step 4: Aggregate final metrics**

After `10/10` completion, compute population mean and standard deviation for only `l2_relative_error` and `maximum_pointwise_error`. Report every seed and the aggregate in scientific notation without silently excluding finite runs.

No Git commit is included because this workspace's existing benchmark plan explicitly replaces commits with test checkpoints.
