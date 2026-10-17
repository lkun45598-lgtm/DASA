# DASA-Aligned SA-PINNs Benchmark Design

## 1. Goal

Build a new SA-PINNs benchmark suite that solves the forward PDE cases from the DASA-PINNs
paper with DASA-aligned equations, coefficients, domains, training-point counts, network size,
and optimization budgets. The optimization method remains SA-PINNs: direct trainable
pointwise masks are maximized while the solution network is minimized. No DASA weight
prediction subnetwork or DASA bilevel-gradient approximation is implemented.

The first experiment phase is one full-budget run of all 13 cases with seed 0. All cases are
launched concurrently in a detached tmux session across eight GPUs. Repeated-seed runs are
not launched until the first-run error magnitudes have been reviewed.

## 2. Scope And Non-Goals

The new implementation lives under `benchmarks/dasa_aligned_sa/`. Existing code under
`Allen-Cahn/`, `Burgers/`, `Helmholtz/`, and `SA-PINNs-master/` remains unchanged and serves
as the reference for the original SA-PINNs minimax behavior.

The suite covers these 13 forward cases:

1. 1D Poisson with a boundary-layer solution.
2. 1D Poisson with a steep solution.
3. 1D Helmholtz.
4. 2D Poisson with a peak solution.
5. 2D Poisson with a steep solution.
6. 2D Helmholtz.
7. Lid-driven cavity at `Re = 100`.
8. 1D Burgers.
9. 1D Allen-Cahn.
10. 1D diffusion-reaction.
11. 1D advection-diffusion.
12. 2D Poisson on an L-shaped domain.
13. 1D advection with a discontinuous initial condition.

The following are out of scope:

- Implementing DASA-PINNs or its weight-prediction subnetwork.
- Inverse Allen-Cahn and inverse Navier-Stokes experiments.
- Adding PINN, DASA-PINN, VPINN, or other methods to this repository.
- Changing or deleting the original SA-PINNs examples.
- Reporting RMSE, MAE, residual error, or additional experiment metrics.
- Automatically starting repeated-seed runs after the first full run.

## 3. Source And Provenance Policy

Every resolved configuration value is stored with one of these provenance labels:

- `paper`: explicitly stated in the DASA-PINNs paper.
- `user`: explicitly standardized by the user for this comparison.
- `sa_reference`: retained from the original SA-PINNs implementation because it is part of
  the comparison method.
- `inferred`: absent or ambiguous in the paper and completed using the closest
  same-dimensional benchmark or cited/reference implementation.

The resolved configuration and provenance map are written into every run directory. An
inferred value must never be serialized as though it came from the paper.

The governing-equation evidence is on paper pages 17-30. General activation,
initialization, and optimizer settings are on pages 14-15. The SA-PINNs comparison and
modified losses are on pages 50-57.

## 4. Common Model And SA-PINNs Method

### 4.1 Solution Network

Every case uses one solution network with:

- Six hidden layers.
- 128 neurons per hidden layer.
- `tanh` activation.
- Xavier-normal kernel initialization and zero biases.
- A linear output layer.
- `float32` network computation and `float64` truth/metric computation.

The six-by-128 standard is explicit in the paper for most cases and was explicitly approved
by the user for Burgers, forward Allen-Cahn, and L-shaped Poisson where the per-case text is
incomplete.

The backbone is the modified MLP used by the DASA paper. Given normalized or embedded input
`X`, it forms two encoder branches

```text
U = tanh(X W_u + b_u)
V = tanh(X W_v + b_v)
H_1 = tanh(X W_1 + b_1)
Z_l = tanh(H_l W_l + b_l)
H_(l+1) = (1 - Z_l) * U + Z_l * V
```

and maps the final hidden state through a linear output layer. The cavity output has three
components `(u, v, p)`; all other cases have one component `u`.

Inputs are affinely normalized to `[-1, 1]` per coordinate before entering the model. For
the three cases where the paper specifies Fourier embedding, the spatial coordinate is
replaced by a first-harmonic periodic embedding before the modified MLP:

- Allen-Cahn and advection-diffusion on `x in [-1, 1]`:
  `(t, cos(pi*x), sin(pi*x))`.
- Diffusion-reaction on `x in [-pi, pi]`:
  `(t, cos(x), sin(x))`.

The paper does not state the number or distribution of Fourier modes. The first-harmonic
embedding is therefore `inferred`; it is the minimal deterministic embedding that enforces
the stated periodic endpoint identification without introducing random Fourier parameters.

### 4.2 Direct Trainable Point Masks

The comparison method retains the original SA-PINNs mechanism:

- One trainable scalar mask is attached to each PDE collocation point.
- One trainable scalar mask is attached to each initial-condition point for time-dependent
  equations.
- Boundary losses remain ordinary weighted MSE terms, matching the existing SA-PINNs
  Burgers, Allen-Cahn, and Helmholtz patterns.
- The solution-network parameters descend the weighted loss.
- The point masks ascend the same loss with an independent Adam optimizer.
- Masks are direct variables, not outputs of another network. They are not min-max scaled or
  normalized as in DASA.
- During L-BFGS, learned masks are frozen and only solution-network parameters are optimized.

For scalar equations, a collocation mask multiplies the scalar residual before squaring. For
the cavity, one scalar collocation mask applies to all three residual components at that
point, and the three squared components are summed. This preserves pointwise rather than
equation-component attention.

Loss-term coefficients follow the DASA appendix:

- `w_r = 1` for PDE residuals.
- `w_b = 100` for boundary constraints.
- `w_i = 100` for initial constraints.

All mask variables use seeded `Uniform(0, 1)` initialization. No clipping or hidden
normalization is applied because that would change the SA-PINNs method.

### 4.3 Optimizer-Budget Mapping

DASA has four stages because it owns both a backbone and a weight-prediction subnetwork.
SA-PINNs has direct point masks instead. To preserve both the DASA compute budget and the
SA-PINNs simultaneous minimax method, the DASA stage counts are summed into one SA-PINNs
Adam minimax budget:

```text
SA Adam steps = N_S1 + N_S2 + N_S3 + N_S4
```

Every Adam step updates the solution network by descent and all applicable point masks by
ascent. This avoids introducing DASA's network-only/subnetwork-only pretraining into the
comparison method. If the paper specifies an L-BFGS phase, it follows Adam with frozen masks.

The solution-network and mask optimizers both start at learning rate `1e-3`. The rate stays
constant for the first `N_S1 + N_S2` mapped steps, then is multiplied by `0.9` every 2,000
steps over the mapped `N_S3 + N_S4` portion. This preserves the paper's rule that decay
starts with adversarial training while still using joint SA-PINNs updates throughout.
Full-batch optimization is used because the masks have a one-to-one relationship with fixed
training points.

## 5. Governing Equations

For manufactured-solution cases, the forcing term is generated from the stated exact
solution and governing operator. Boundary labels are always evaluated from that same exact
solution, including nonzero boundaries.

### 5.1 Static PDEs

#### 1D Poisson: Boundary Layer

```text
-u_xx = f(x),                           x in [-1, 1]
u_exact = 0.1 sin(5*pi*x) + exp((0.01 - (x + 1)) / 0.01)
u(-1) = u_exact(-1), u(1) = u_exact(1)
```

#### 1D Poisson: Steep Solution

```text
-u_xx = f(x),                           x in [-1, 1]
u_exact = 0.1 sin(8*pi*x) + tanh(80*x)
u(-1) = u_exact(-1), u(1) = u_exact(1)
```

#### 1D Helmholtz

```text
-u_xx + q(x) u = f(x),                  x in [0, 1]
u_exact = sin(20*pi*x)
u(0) = u(1) = 0
```

The paper leaves `q(x)` unspecified. This suite uses `q(x) = 1`, marked `inferred`, and
constructs `f(x) = ((20*pi)^2 + 1) sin(20*pi*x)`.

#### 2D Poisson: Peak Solution

```text
-Delta u = f(x, y),                     (x, y) in [0, 1]^2
u_exact = 2^(4*a) x^a y^a (1-x)^a (1-y)^a, a = 10
u on the boundary = u_exact
```

The printed DASA equation omits the `y^a` factor. That omission would produce values of
about `10^6` on `y=0`, which is inconsistent with both the standard peak benchmark and the
paper's reported `2.7e-4` maximum pointwise error. Restoring `y^a` is therefore an
`inferred` correction to a paper typesetting error, not an unmarked equation change.

#### 2D Poisson: Steep Solution

```text
-Delta u = f(x, y),                     (x, y) in [0, 1]^2
u_exact = (0.1 sin(2*pi*x) + tanh(10*x)) sin(2*pi*y)
u on the boundary = u_exact
```

#### 2D Helmholtz

```text
Delta u + k^2 u - q(x, y) = 0,          (x, y) in [-1, 1]^2
u = 0 on all four edges
u_exact = sin(a1*pi*x) sin(a2*pi*y)
q = -(a1*pi)^2 u_exact -(a2*pi)^2 u_exact + k^2 u_exact
```

The paper prints the symbolic parameters but not their numerical values. The suite uses
`a1 = 1`, `a2 = 4`, and `k = 1`, marked `inferred`, matching the bundled canonical SA-PINNs
Helmholtz implementation used for the paper's SA comparison.

#### Lid-Driven Cavity

```text
u_vec . grad(u_vec) + grad(p) - (1/Re) Delta(u_vec) = 0
div(u_vec) = 0
(x, y) in [0, 1]^2, Re = 100
u_vec(x, 1) = (1, 0)
u_vec(x, 0) = u_vec(0, y) = u_vec(1, y) = (0, 0)
```

The model predicts `(u, v, p)`. A single pressure gauge `p(0.5, 0.5) = 0` is added as an
`inferred` numerical constraint; it removes the arbitrary pressure constant without changing
the velocity solution, and its fixed loss coefficient is `1`. The moving-lid condition is
applied for `0 < x < 1`; the four isolated corners are omitted from boundary training because
the top and side conditions conflict there. The same four corners are omitted from the
cavity metric grid. Metrics are computed only for horizontal velocity `u`, matching the field
reported and displayed in the DASA paper.

#### L-Shaped Poisson

```text
-Delta u = 1,                           (x, y) in Omega
u = 0,                                  (x, y) on boundary(Omega)
Omega = [-1, 1]^2 minus [0, 1]^2
```

Residual points satisfying `x >= 0 and y >= 0` are excluded. The positive coordinate axes
are sampled separately as the internal re-entrant boundary. The two length-two outer edges
are split at zero, giving the eight unit line segments implied by the paper's `128x8`
boundary count.

### 5.2 Time-Dependent PDEs

#### Burgers

```text
u_t + u u_x - (0.01/pi) u_xx = 0,       t in [0, 1], x in [-1, 1]
u(0, x) = -sin(pi*x)
u(t, -1) = u(t, 1) = 0
```

#### Allen-Cahn

```text
u_t - 0.0001 u_xx + 5 u^3 - 5 u = 0,   t in [0, 1], x in [-1, 1]
u(0, x) = x^2 cos(pi*x)
u(t, -1) = u(t, 1)
u_x(t, -1) = u_x(t, 1)
```

The Fourier embedding enforces endpoint periodicity, so no separate boundary samples are
used for this case.

#### Diffusion-Reaction

```text
u_t = D u_xx + R(t, x),                 t in [0, 1], x in [-pi, pi]
D = 1
u(0, x) = sum(k=1..4) sin(k*x)/k + sin(8*x)/8
u(t, -pi) = u(t, pi) = 0
R = exp(-t) [3/2 sin(2*x) + 8/3 sin(3*x)
             + 15/4 sin(4*x) + 63/8 sin(8*x)]
u_exact = exp(-t) [sum(k=1..4) sin(k*x)/k + sin(8*x)/8]
```

#### Advection-Diffusion

```text
u_t + v u_x = kappa u_xx,               t in [0, 1], x in [-1, 1]
v = 1
kappa = 0.01
u(0, x) = -sin(pi*x)
u(t, -1) = u(t, 1) = 0
```

The DASA paper leaves `kappa` symbolic. Its cited source studies several diffusivities but
does not identify the paper's choice. `kappa = 0.01` is the approved reproducibility choice
and is marked `inferred`. The exact evaluator uses the cited infinite-series construction
with 1,000 terms and numerically stable exponential scaling.

#### Discontinuous Advection

```text
u_t + v u_x = kappa u_xx,               t in [0, 1], x in [-1.5, 1.5]
v = 1, kappa = 0
u(t, -1.5) = u(t, 1.5) = 0
u(0, x) = 1 for x in [-0.2, 0.2], else 0
u_exact(t, x) = u_0(x - t)
```

Evaluation follows the paper's translated pulse on the stated finite domain. Values exactly
on a jump are defined consistently as `1` because the initial interval is closed; these
measure-zero points are nevertheless fixed so maximum-error results are reproducible.

## 6. Sampling And Optimization Matrix

`N_r`, `N_bc`, and `N_ic` mean residual, total boundary, and initial-condition points.
`Adam` is the summed SA-PINNs minimax budget described in Section 4.3.

| Case | Sampling | DASA stages `(S1,S2,S3,S4)` | SA Adam | L-BFGS | Provenance notes |
|---|---:|---:|---:|---:|---|
| 1D Poisson boundary | `N_r=500`, 2 endpoints | `(1000,500,20000,20000)` | 41,500 | 0 | Paper |
| 1D Poisson steep | `N_r=500`, 2 endpoints | `(1000,500,20000,20000)` | 41,500 | 0 | Paper |
| 1D Helmholtz | `N_r=500`, 2 endpoints | `(1000,500,10000,10000)` | 21,500 | 0 | S1/S2 inherited from stated similar 1D Poisson strategy |
| 2D Poisson peak | `N_r=2500`, `N_bc=50` | `(1000,1000,10000,10000)` | 22,000 | 1,000 | S1/S2 paper global defaults |
| 2D Poisson steep | `N_r=2500`, `N_bc=50` | `(1000,1000,10000,10000)` | 22,000 | 1,000 | S1/S2 paper global defaults |
| 2D Helmholtz | `101x101` residual mesh, `N_bc=200` | `(1000,1000,10000,10000)` | 22,000 | 1,000 | Paper |
| Cavity Re=100 | `50x50` residual mesh, `N_bc=512` | `(5000,2000,10000,10000)` | 27,000 | 1,000 | Paper |
| Burgers | `N_r=25600`, `N_bc=256`, `N_ic=512` | `(1000,2000,10000,10000)` | 23,000 | 1,000 | Sampling/budget inferred from same-dimensional diffusion/advection family |
| Allen-Cahn | `N_r=25600`, `N_ic=512` | `(2000,2000,150000,150000)` | 304,000 | 0 | S2 and S4 inferred as matching the explicitly stated S1 and S3 strategy |
| Diffusion-reaction | `N_r=25600`, `N_bc=256`, `N_ic=512` | `(1000,2000,10000,10000)` | 23,000 | 1,000 | Paper |
| Advection-diffusion | `N_r=25600`, `N_bc=256`, `N_ic=512` | `(1000,2000,10000,10000)` | 23,000 | 1,000 | Paper |
| L-shaped Poisson | `N_r=10000`, `N_bc=1024` | `(5000,2000,10000,10000)` | 27,000 | 1,000 | Paper; network size user-standardized |
| Discontinuous advection | `N_r=25600`, `N_bc=256`, `N_ic=512` | `(5000,2000,10000,10000)` | 27,000 | 1,000 | Counts inferred from same-dimensional family; stages from paper |

LHS is used where the paper specifies LHS or generic physical-domain sampling. Structured
meshes are used where the paper explicitly gives `101x101` or `50x50`. Boundary counts are
totals, split as evenly as possible across boundary segments. The training set is generated
once per seed and stays fixed so each direct SA mask remains attached to one point.

## 7. Truth Data

### 7.1 Analytic Or Series Truth

Direct analytic truth is used for both 1D Poisson cases, 1D Helmholtz, both 2D Poisson
cases, 2D Helmholtz, diffusion-reaction, and discontinuous advection. Advection-diffusion
uses its 1,000-term analytical series.

### 7.2 Numerical Truth Sources

Burgers and Allen-Cahn use the numerical datasets already shipped with the original SA-PINN
implementation: `Burgers/burgers_shock.mat` and `Allen-Cahn/AC.mat`. Cavity uses the
`reference_u.csv` horizontal-velocity field from the official repository for DASA reference
[19], `PredictiveIntelligenceLab/GradientPathologiesPINNs` at commit `93e752b`.

Only L-shaped Poisson is generated locally. It uses a sparse five-point finite-difference
solve on the exact masked Cartesian domain at `101`, `201`, and `401` grid points. The
fine-versus-next-fine limits are relative L2 at most `5e-4` and maximum difference at most
`5e-3`; failure does not relax either threshold. Its `.npz` and JSON manifest record the
solver, grid sizes, convergence values, generator SHA-256, and reference SHA-256.

### 7.3 Fixed Evaluation Grids

Evaluation coordinates are deterministic and independent of training samples:

- 1D static cases: 10,001 uniformly spaced points.
- Regular 2D static cases: a `401x401` uniform mesh.
- Burgers: the original `256x100` SA-PINN data grid.
- Allen-Cahn: the original `512x201` SA-PINN data grid.
- Other time-dependent 1D cases: `512` spatial by `201` temporal points.
- Cavity: the `100x100` grid supplied by reference [19], with metrics on horizontal velocity
  `u` only.
- L-shaped Poisson: the closure of the valid L-shaped region on a `401x401` mesh over
  `[-1,1]^2`; the strict removed quadrant is excluded and its two axial boundary segments
  are retained.

Numerical references may be generated on finer grids than these evaluation grids.

## 8. Metrics And Result Schema

Only the following two experiment metrics are computed and placed in result tables:

```python
def l2_relative_error(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)


def maximum_pointwise_error(y_true, y_pred):
    return np.max(np.abs(y_true - y_pred))
```

Inputs are flattened in the same deterministic coordinate order and converted to `float64`
before evaluation. No epsilon, normalization by point count, interpolation after prediction,
or component aggregation is added. Cavity passes only the flattened horizontal-velocity
arrays to these functions.

The machine-readable result for one run contains:

```json
{
  "case": "case_id",
  "seed": 0,
  "l2_relative_error": 1.23e-4,
  "maximum_pointwise_error": 4.56e-4
}
```

The numbers above illustrate the schema only; they are not expected or prefilled results.

Run status, timing, checkpoints, and training losses are operational metadata, not benchmark
metrics, and are stored separately. The aggregate experiment table contains the case, seed,
and exactly the two requested metric columns.

## 9. First-Run Error Review

The first run is a numerical sanity pass, not an expectation that SA-PINNs must reproduce
DASA-PINNs accuracy. Relative-L2 results are compared with the SA-PINNs column in DASA Table
1 because those values are the closest method-matched reference:

| Case | Paper SA-PINNs relative L2 |
|---|---:|
| 1D Poisson boundary | `9.51e-2` |
| 1D Poisson steep | `9.96e-4` |
| 1D Helmholtz | `1.79e-2` |
| 2D Poisson peak | `1.49e-4` |
| 2D Poisson steep | `7.80e-5` |
| 2D Helmholtz | `8.00e-5` |
| Cavity Re=100 | `8.53e-2` |
| Burgers | `5.26e-4` |
| Allen-Cahn | `7.48e-2` |
| Diffusion-reaction | `1.96e-4` |
| Advection-diffusion | `2.89e-2` |
| L-shaped Poisson | `7.07e-3` |
| Discontinuous advection | `8.80e-2` |

A result is flagged for diagnosis if it is non-finite, if the prediction is effectively
constant where the truth is not, or if relative L2 is more than one order of magnitude worse
than the corresponding paper SA-PINNs value. A much better result is checked for train/test
leakage and truth-coordinate mistakes before acceptance. Maximum pointwise error is always
reported but is not rejected against a universal threshold because the paper does not give
that value consistently for all 13 SA-PINNs cases.

Repeated seeds begin only after all flagged cases have been diagnosed and the first-run
table has been reviewed.

## 10. Package Layout And Interfaces

```text
benchmarks/dasa_aligned_sa/
  README.md
  configs/
    cases/*.yaml
  dasa_sa/
    config.py
    networks.py
    sa_weights.py
    losses.py
    trainer.py
    metrics.py
    references.py
    problems/
      static_1d.py
      static_2d.py
      time_1d.py
      cavity.py
      lshape.py
  scripts/
    generate_references.py
    run_case.py
    launch_full_tmux.sh
    monitor_runs.py
    aggregate_results.py
  references/
  runs/
  tests/
```

Problem definitions own domains, equations, exact/reference evaluators, training-point
generation, and output selection for metrics. The trainer owns only the shared modified MLP,
SA minimax updates, learning-rate schedule, checkpointing, and L-BFGS handoff. This prevents
a 1D problem from borrowing 2D derivative logic and keeps each PDE independently testable.

`run_case.py --case CASE --seed SEED --output DIR` is the only training entry point. It
resolves one case configuration, validates reference hashes, enables TensorFlow GPU memory
growth, trains, evaluates, and atomically writes the final metric JSON.

## 11. GPU And Tmux Launch

The detached session is named `sa_dasa_full_seed0`. It has one window per experiment plus a
monitor window. All 13 training windows start immediately; cases sharing a GPU are separate
concurrent processes, not a sequential queue.

| GPU | Concurrent experiment windows |
|---:|---|
| 0 | Allen-Cahn |
| 1 | diffusion-reaction |
| 2 | advection-diffusion |
| 3 | cavity; 1D Poisson boundary-layer |
| 4 | L-shaped Poisson; 1D Poisson steep |
| 5 | 2D Helmholtz; 1D Helmholtz |
| 6 | Burgers; 2D Poisson peak |
| 7 | discontinuous advection; 2D Poisson steep |

Each window exports one physical GPU through `CUDA_VISIBLE_DEVICES`, enables memory growth,
and writes to an isolated run directory. The monitor reports process state, last progress
record, result-file presence, and GPU memory/utilization without modifying a run. A failed
case stays failed and visible; it is not automatically restarted with different parameters.

## 12. Validation And Failure Handling

Before the full tmux launch:

1. Unit-test every PDE residual against its manufactured exact solution where available.
2. Verify all initial and boundary-condition callbacks on endpoints, corners, and random
   samples.
3. Verify modified-MLP input/output shapes and first/second automatic derivatives for 1D,
   1D-time, 2D, and three-output cavity models.
4. Verify SA mask gradients ascend while network gradients descend on a small deterministic
   loss.
5. Verify both metric functions exactly against small NumPy arrays.
6. Verify the three upstream datasets, then generate and convergence-check the L-shaped
   reference and its SHA-256 manifest.
7. Run every case for a very small smoke budget on GPU and require finite loss, prediction,
   and the two metric values.
8. Validate tmux command generation and ensure all 13 windows have distinct output paths.

If a full run fails, the code preserves its resolved config, seed, log, last checkpoint, and
exception. Numerical instability is diagnosed without silently changing coefficients,
sampling, precision, network size, or budget. Any later deviation becomes a new explicitly
named configuration and cannot overwrite the aligned seed-0 run.

## 13. Success Criteria

The implementation is ready for the first full experiment only when:

- All 13 cases exactly match the equations and domain choices in this document.
- Every model has six hidden layers of width 128 and the configured modified MLP/Fourier
  behavior.
- Direct SA-PINNs masks, not a DASA subnetwork, perform the pointwise maximization.
- Only relative L2 error and maximum pointwise error appear in experiment results.
- Burgers, Allen-Cahn, and cavity use their upstream numerical datasets; L-shaped Poisson
  uses its convergence-checked generated reference.
- All tests and 13 GPU smoke runs pass.
- One detached tmux command starts all 13 full-budget seed-0 runs with the approved mapping.
- The aggregate seed-0 table is reviewed before any repeated-seed launch.
