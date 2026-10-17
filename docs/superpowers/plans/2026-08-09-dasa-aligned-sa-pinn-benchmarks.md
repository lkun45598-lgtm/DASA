# DASA-Aligned SA-PINNs Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify a modular SA-PINNs suite for all 13 DASA-aligned forward PDE benchmarks, including independent truth generation and eight-GPU tmux launch tooling.

**Architecture:** A shared TensorFlow training core owns the modified MLP, direct SA masks, Adam minimax updates, L-BFGS, metrics, and artifact handling. Small problem classes own equations, sampling, boundary/initial constraints, and evaluation truth; numerical-reference solvers are isolated from training. JSON-syntax `.yaml` files hold case configurations so the existing environment needs no YAML dependency.

**Tech Stack:** Python 3.10, TensorFlow 2.15.1, NumPy 1.26.4, SciPy 1.11.4, pyDOE, standard-library `unittest`, Bash, tmux.

**Repository constraint:** `/data1/user/lz/SA-PINNs` is an untracked directory inside a large parent repository whose Git metadata is read-only. The usual per-task commits are replaced by explicit test checkpoints; do not stage or modify unrelated parent-repository files.

---

### Task 1: Package Skeleton And Configuration Contract

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/dasa_aligned_sa/__init__.py`
- Create: `benchmarks/dasa_aligned_sa/README.md`
- Create: `benchmarks/dasa_aligned_sa/dasa_sa/__init__.py`
- Create: `benchmarks/dasa_aligned_sa/scripts/__init__.py`
- Create: `benchmarks/dasa_aligned_sa/tests/__init__.py`
- Create: `benchmarks/dasa_aligned_sa/dasa_sa/config.py`
- Create: `benchmarks/dasa_aligned_sa/configs/cases/*.yaml`
- Create: `benchmarks/dasa_aligned_sa/tests/test_config.py`

- [ ] **Step 1: Write failing configuration tests**

Test that `list_case_ids()` returns exactly the 13 approved IDs, every file resolves to a `CaseConfig`, every network is six-by-128, only approved provenance labels occur, and inferred `kappa`, `q`, and Helmholtz frequencies equal the design values.

```python
self.assertEqual(len(list_case_ids()), 13)
self.assertEqual(config.network.hidden_layers, 6)
self.assertEqual(config.network.width, 128)
self.assertEqual(load_case_config("advection_diffusion").parameters["kappa"], 0.01)
```

- [ ] **Step 2: Run the tests and verify import failure**

Run: `.conda-env/bin/python -m unittest benchmarks.dasa_aligned_sa.tests.test_config -v`

Expected: FAIL because `dasa_sa.config` does not exist.

- [ ] **Step 3: Implement frozen dataclasses and strict loading**

Define `NetworkConfig`, `OptimizerConfig`, `SamplingConfig`, and `CaseConfig`. Parse JSON syntax from `.yaml` with `json.load`, reject missing/unknown keys, validate positive counts, validate stage-budget sums, and resolve paths relative to the suite root.

```python
@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    family: str
    parameters: dict[str, float]
    sampling: SamplingConfig
    network: NetworkConfig
    optimizer: OptimizerConfig
    provenance: dict[str, str]
```

- [ ] **Step 4: Add all 13 resolved case files**

Encode the exact sampling and optimization matrix from the approved design. Each file includes `case_id`, family, domain, parameters, network, Fourier mode, fixed loss weights, sampling counts/method, four source stages, summed Adam steps, L-BFGS steps, and field-level provenance.

- [ ] **Step 5: Run configuration tests**

Run: `.conda-env/bin/python -m unittest benchmarks.dasa_aligned_sa.tests.test_config -v`

Expected: 13 configurations load and all tests PASS.

### Task 2: Metrics And Deterministic Evaluation Coordinates

**Files:**
- Create: `benchmarks/dasa_aligned_sa/dasa_sa/metrics.py`
- Create: `benchmarks/dasa_aligned_sa/dasa_sa/grids.py`
- Create: `benchmarks/dasa_aligned_sa/tests/test_metrics.py`
- Create: `benchmarks/dasa_aligned_sa/tests/test_grids.py`

- [ ] **Step 1: Write failing metric tests**

Use hand-computed arrays to enforce the user's exact formulas, float64 conversion, shape validation, and no extra metrics.

```python
y_true = np.array([3.0, 4.0])
y_pred = np.array([0.0, 4.0])
self.assertAlmostEqual(l2_relative_error(y_true, y_pred), 3.0 / 5.0)
self.assertEqual(maximum_pointwise_error(y_true, y_pred), 3.0)
```

- [ ] **Step 2: Verify the tests fail, then implement the two functions**

Run: `.conda-env/bin/python -m unittest benchmarks.dasa_aligned_sa.tests.test_metrics -v`

Implementation must expose only `l2_relative_error`, `maximum_pointwise_error`, and `compute_metrics`; `compute_metrics` returns exactly those two named values.

- [ ] **Step 3: Write failing grid tests**

Assert 10,001 static-1D points, `401x401` regular-2D points, `512x201` time points, cavity corner exclusion, and L-shaped strict-quadrant exclusion with both axial boundaries retained.

- [ ] **Step 4: Implement grid factories and rerun tests**

Run: `.conda-env/bin/python -m unittest benchmarks.dasa_aligned_sa.tests.test_metrics benchmarks.dasa_aligned_sa.tests.test_grids -v`

Expected: PASS.

### Task 3: Modified MLP And Direct SA Masks

**Files:**
- Create: `benchmarks/dasa_aligned_sa/dasa_sa/networks.py`
- Create: `benchmarks/dasa_aligned_sa/dasa_sa/sa_weights.py`
- Create: `benchmarks/dasa_aligned_sa/tests/test_networks.py`
- Create: `benchmarks/dasa_aligned_sa/tests/test_sa_weights.py`

- [ ] **Step 1: Write failing network tests**

Check output shapes for scalar and `(u,v,p)` models, exactly six backbone hidden
transformations of width 128 (one initial projection plus five gated mixing layers),
deterministic Xavier-normal initialization, finite first/second derivatives, and endpoint
equality for both Fourier embeddings.

- [ ] **Step 2: Implement `ModifiedMLP` and Fourier mappings**

Use two encoder branches, one initial hidden projection, and five gated mixing
transformations. Normalize raw coordinates first for ordinary cases; preserve time and map
the spatial coordinate to the configured sine/cosine pair for Fourier cases.

```python
z = tf.math.tanh(layer(h))
h = (1.0 - z) * u_encoder + z * v_encoder
```

- [ ] **Step 3: Write a failing minimax-direction test**

For `loss = mean((mask * residual)**2)`, verify one network descent step lowers the loss with fixed masks and one mask ascent step increases it with a fixed residual.

- [ ] **Step 4: Implement seeded direct masks and minimax application**

Create one `tf.Variable` per requested point group using `Uniform(0,1)`. Apply ordinary gradients to model variables and negated gradients to mask variables through independent Adam optimizers.

- [ ] **Step 5: Run network and mask tests**

Run: `.conda-env/bin/python -m unittest benchmarks.dasa_aligned_sa.tests.test_networks benchmarks.dasa_aligned_sa.tests.test_sa_weights -v`

Expected: PASS with TensorFlow CPU or GPU.

### Task 4: Problem Interface And Analytic Static PDEs

**Files:**
- Create: `benchmarks/dasa_aligned_sa/dasa_sa/problems/__init__.py`
- Create: `benchmarks/dasa_aligned_sa/dasa_sa/problems/base.py`
- Create: `benchmarks/dasa_aligned_sa/dasa_sa/problems/static_1d.py`
- Create: `benchmarks/dasa_aligned_sa/dasa_sa/problems/static_2d.py`
- Create: `benchmarks/dasa_aligned_sa/tests/test_static_problems.py`

- [ ] **Step 1: Write failing manufactured-residual tests**

For both 1D Poisson cases, 1D Helmholtz, both regular 2D Poisson cases, and 2D Helmholtz, evaluate the exact TensorFlow solution at interior points and require the governing residual to be below `1e-4` in float64 test computation. Test every endpoint/edge callback independently.

- [ ] **Step 2: Define the problem contract**

```python
class PDEProblem(ABC):
    def sample_training(self, seed: int) -> TrainingData: ...
    def residual(self, model, residual_points) -> tf.Tensor: ...
    def constraint_losses(self, model, data, masks) -> dict[str, tf.Tensor]: ...
    def evaluation_data(self) -> tuple[np.ndarray, np.ndarray]: ...
```

`TrainingData` keeps residual, boundary, initial, and pressure-gauge arrays distinct.

- [ ] **Step 3: Implement the seven analytic static cases**

Use seeded `scipy.stats.qmc.LatinHypercube` for stated random samples, structured residual
points for 2D Helmholtz, exact boundary labels, restored `y^a` in the peak solution,
`q(x)=1`, and `(a1,a2,k)=(1,4,1)`.

- [ ] **Step 4: Run static problem tests**

Run: `.conda-env/bin/python -m unittest benchmarks.dasa_aligned_sa.tests.test_static_problems -v`

Expected: all manufactured residual and sampling-count tests PASS.

### Task 5: Analytic Time PDEs

**Files:**
- Create: `benchmarks/dasa_aligned_sa/dasa_sa/problems/time_1d.py`
- Create: `benchmarks/dasa_aligned_sa/tests/test_time_problems.py`

- [ ] **Step 1: Write failing equation and condition tests**

Cover Burgers, Allen-Cahn, diffusion-reaction, advection-diffusion, and discontinuous advection. Analytic-residual checks apply to diffusion-reaction, advection-diffusion series points away from `t=0`, and discontinuous advection points away from jumps. Check all initial/boundary callbacks and exact sample counts.

- [ ] **Step 2: Implement all five time problems**

Use nested gradient tapes for `u_t`, `u_x`, and `u_xx`; use no Fourier mapping for Burgers or discontinuous advection. Implement the 1,000-term advection-diffusion series with stable scaling and exact zero endpoint assignment.

- [ ] **Step 3: Verify time-problem tests**

Run: `.conda-env/bin/python -m unittest benchmarks.dasa_aligned_sa.tests.test_time_problems -v`

Expected: PASS, with finite series values across the complete evaluation grid.

### Task 6: Cavity And L-Shaped Problems

**Files:**
- Create: `benchmarks/dasa_aligned_sa/dasa_sa/problems/cavity.py`
- Create: `benchmarks/dasa_aligned_sa/dasa_sa/problems/lshape.py`
- Create: `benchmarks/dasa_aligned_sa/tests/test_irregular_problems.py`

- [ ] **Step 1: Write failing geometry and residual tests**

Verify cavity returns three residual components, has 2,500 residual points and 512 noncorner boundary points, applies the lid only on the open top edge, and includes one pressure gauge. Verify L-shaped residual samples never enter the removed quadrant and its 1,024 boundary points are balanced over eight unit segments.

- [ ] **Step 2: Implement cavity derivatives and loss inputs**

Compute `(r_u,r_v,r_c)` with nested tapes and apply one point mask to the three-component residual vector. Keep the pressure gauge separate with coefficient one.

- [ ] **Step 3: Implement L-shaped sampling and operator**

Use rejection LHS for 10,000 valid residual points, eight deterministic 128-point boundary segments, and `-Delta(u)-1`.

- [ ] **Step 4: Run irregular-problem tests**

Run: `.conda-env/bin/python -m unittest benchmarks.dasa_aligned_sa.tests.test_irregular_problems -v`

Expected: PASS.

### Task 7: Numerical Truth Data

**Files:**
- Create: `benchmarks/dasa_aligned_sa/dasa_sa/reference_solvers.py`
- Create: `benchmarks/dasa_aligned_sa/dasa_sa/references.py`
- Create: `benchmarks/dasa_aligned_sa/scripts/generate_references.py`
- Create: `benchmarks/dasa_aligned_sa/tests/test_references.py`
- Add: `benchmarks/dasa_aligned_sa/data/cavity_reference_u.csv`
- Generate: `benchmarks/dasa_aligned_sa/references/lshape_poisson.npz`
- Generate: `benchmarks/dasa_aligned_sa/references/lshape_poisson.json`

- [ ] **Step 1: Write failing manifest and convergence tests**

Create a temporary reference, require equation parameters, solver settings, all grid levels,
both convergence values, generator SHA-256, file SHA-256, and rejection of tampered files.

- [ ] **Step 2: Implement deterministic reference storage**

Write `.npz` via a temporary path and atomic rename, then serialize the manifest and verify
the digest on every load. Never regenerate truth from a training command.

- [ ] **Step 3: Load upstream datasets and implement the L-shaped solver**

Load Burgers and Allen-Cahn directly from the `.mat` files shipped with SA-PINN. Load cavity
horizontal velocity from the `100x100` CSV used by DASA reference [19]. Implement only the
sparse five-point L-shaped Poisson solver with explicit grid levels.

- [ ] **Step 4: Run small-grid solver tests**

Run: `.conda-env/bin/python -m unittest benchmarks.dasa_aligned_sa.tests.test_references -v`

Expected: upstream data shapes, coordinate order, boundary values, manifest verification,
and L-shaped coarse-grid convergence direction PASS.

- [ ] **Step 5: Generate full references and enforce convergence gates**

Run: `.conda-env/bin/python -m benchmarks.dasa_aligned_sa.scripts.generate_references --all`

Expected: one verified L-shaped `.npz` and manifest; the command exits nonzero if either
required fine-grid convergence metric exceeds the approved threshold.

### Task 8: Shared Trainer, L-BFGS, Checkpoints, And Run CLI

**Files:**
- Create: `benchmarks/dasa_aligned_sa/dasa_sa/lbfgs.py`
- Create: `benchmarks/dasa_aligned_sa/dasa_sa/trainer.py`
- Create: `benchmarks/dasa_aligned_sa/dasa_sa/artifacts.py`
- Create: `benchmarks/dasa_aligned_sa/scripts/run_case.py`
- Create: `benchmarks/dasa_aligned_sa/tests/test_trainer.py`
- Create: `benchmarks/dasa_aligned_sa/tests/test_run_case.py`

- [ ] **Step 1: Write failing optimizer and artifact tests**

Optimize a scalar quadratic through the L-BFGS wrapper, verify masks remain frozen during
L-BFGS, verify mapped learning-rate boundaries, and assert result JSON has exactly `case`,
`seed`, `l2_relative_error`, and `maximum_pointwise_error`.

- [ ] **Step 2: Implement flatten/assign and SciPy L-BFGS bridge**

Flatten all model trainable variables, assign a candidate vector, compute TensorFlow loss and
gradient, and call `scipy.optimize.minimize(method="L-BFGS-B", options={"maxiter": n})`.

- [ ] **Step 3: Implement compiled Adam minimax training**

Create fixed full-batch tensors, direct masks, two Adam optimizers, the constant-then-decay
schedule, finite checks, periodic JSONL progress, and model-only L-BFGS handoff.

- [ ] **Step 4: Implement atomic artifacts and CLI overrides**

`run_case.py` accepts `--case`, `--seed`, `--output`, `--adam-steps`, and `--lbfgs-steps`.
Budget overrides exist only for tests/smoke runs and are written into resolved config; full
launch scripts pass no overrides.

- [ ] **Step 5: Run trainer and CLI tests**

Run: `.conda-env/bin/python -m unittest benchmarks.dasa_aligned_sa.tests.test_trainer benchmarks.dasa_aligned_sa.tests.test_run_case -v`

Expected: PASS and a tiny analytic case writes finite values for exactly the two metrics.

### Task 9: Aggregation, Tmux Launch, And Monitoring

**Files:**
- Create: `benchmarks/dasa_aligned_sa/scripts/aggregate_results.py`
- Create: `benchmarks/dasa_aligned_sa/scripts/monitor_runs.py`
- Create: `benchmarks/dasa_aligned_sa/scripts/launch_full_tmux.sh`
- Create: `benchmarks/dasa_aligned_sa/tests/test_orchestration.py`

- [ ] **Step 1: Write failing orchestration tests**

Parse the launch script and assert 13 distinct case windows, the exact GPU mapping, one
monitor window, no sequential wait/queue, isolated output directories, memory-growth setup,
and no Adam/L-BFGS override flags.

- [ ] **Step 2: Implement result aggregation**

Read only completed `metrics.json` files and emit CSV/Markdown columns `case`, `seed`,
`l2_relative_error`, and `maximum_pointwise_error`. Missing/failed cases are reported outside
the metric table.

- [ ] **Step 3: Implement monitor and detached launch script**

Create `sa_dasa_full_seed0`, one window per case, and start all training commands immediately.
Use the approved GPU mapping and `TF_FORCE_GPU_ALLOW_GROWTH=true`; redirect each window to its
own `train.log`.

- [ ] **Step 4: Run orchestration tests**

Run: `.conda-env/bin/python -m unittest benchmarks.dasa_aligned_sa.tests.test_orchestration -v`

Expected: PASS without starting tmux.

### Task 10: Full Verification And GPU Smoke Matrix

**Files:**
- Modify: `benchmarks/dasa_aligned_sa/README.md`
- Create: `benchmarks/dasa_aligned_sa/scripts/run_smoke_matrix.sh`

- [ ] **Step 1: Run the complete unit-test suite**

Run: `.conda-env/bin/python -m unittest discover -s benchmarks/dasa_aligned_sa/tests -v`

Expected: all tests PASS.

- [ ] **Step 2: Compile all new Python files**

Run: `.conda-env/bin/python -m compileall -q benchmarks/dasa_aligned_sa`

Expected: exit code 0.

- [ ] **Step 3: Run all 13 cases with smoke budgets on available GPUs**

Run: `bash benchmarks/dasa_aligned_sa/scripts/run_smoke_matrix.sh`

Each case uses `--adam-steps 2 --lbfgs-steps 1`; expected output is 13 isolated run
directories with finite prediction arrays and exactly two finite metrics.

- [ ] **Step 4: Validate full launch without starting training**

Run: `bash -n benchmarks/dasa_aligned_sa/scripts/launch_full_tmux.sh`

Run: `.conda-env/bin/python -m benchmarks.dasa_aligned_sa.scripts.monitor_runs --root benchmarks/dasa_aligned_sa/runs/smoke`

Expected: shell syntax passes and monitor reports all 13 smoke cases.

- [ ] **Step 5: Document exact operational commands**

README must include reference generation, one-case smoke/full runs, full tmux launch,
attach/detach, monitor, aggregation, artifact layout, and the rule that repeated seeds wait
for seed-0 review.
