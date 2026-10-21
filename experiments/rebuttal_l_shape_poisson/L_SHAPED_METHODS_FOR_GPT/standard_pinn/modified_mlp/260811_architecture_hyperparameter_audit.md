# Classical PINN architecture and hyperparameter audit

## 1. Audit objective

This audit checks whether each DeepXDE classical PINN baseline keeps the DASA-PINNs
backbone architecture and shared experimental settings while removing only the DASA-specific
sub-network, pointwise adaptive residual weights, and adversarial optimization.

Primary source:

```text
/data/ccy/260809_补实验_pinn/2. DASA_PINNs (Preprint submitted to Neurocomputing).pdf
```

No training code or existing result was changed during this audit.

## 2. Paper statements relevant to the baseline

The paper gives two closely related fairness statements:

- The discussion following Table 1 states that DASA-PINNs, classical PINNs, and SA-PINNs
  use the same number of training points and optimization method.
- The caption of Table 2 states that classical PINNs and DASA-PINNs use the same
  architecture and hyperparameters.

Therefore, the architecture-matched classical PINN should retain the DASA backbone,
training points, fixed loss-term coefficients, optimizer family, learning-rate schedule, and
applicable L-BFGS stage. It should remove only:

- the DASA sub-network;
- pointwise residual weights predicted by the sub-network;
- gradient-ascent and adversarial/min-max updates involving the sub-network.

The fixed coefficients such as `wr=1`, `wb=100`, and `wi=100` are part of the PDE loss
configuration and must not be removed as if they were DASA pointwise weights.

## 3. Shared backbone audit

All 11 current PDE scripts instantiate the corrected custom `ModifiedMLP` from
`pinn_baselines/common_modified_mlp_corrected.py`. None of them uses DeepXDE's default
ordinary `FNN`. The original `pinn_baselines/common.py` is retained unchanged as the legacy
implementation associated with historical results.

The current implementation contains:

- two input encoders `U` and `V`;
- one initial gating layer;
- `depth - 1` additional gating layers;
- one linear output layer;
- `tanh` activation;
- Xavier/Glorot normal initialization and zero biases.

The DASA paper refers to [21], and [21] prints the modified-MLP forward pass. Under its
`L=6` convention, the relevant sequence is:

```text
U = tanh(F Wu + bu)
V = tanh(F Wv + bv)
H1 = tanh(F W1 + b1)
for l = 1,...,5:
    Zl = tanh(Hl Wl + bl)
    H(l+1) = (1-Zl)*U + Zl*V
y = H6 Wout + bout
```

The legacy `common.py` instead immediately mixes the initial hidden state before entering
its five stored hidden layers:

```text
H1 = tanh(F W1 + b1)
H1 = H1*U + (1-H1)*V  # extra initial mixing
```

It then performs five further gated mixes. Consequently, the number and shapes of trainable
linear layers agree with a six-hidden-layer network, but the forward function is not exactly
the one printed in [21]. The extra mix introduces no parameters, yet it changes the model
class. The corrected implementation removes this pre-loop mix. The `Z*U + (1-Z)*V`
versus `(1-Z)*U + Z*V` ordering alone is not material because the two independent encoders
can be renamed; the extra pre-loop mixing is the actual discrepancy.

Runtime inspection confirmed these parameter counts, which are unchanged by the formula
correction:

```text
1D scalar input, scalar output                 83,457
2D scalar input, scalar output                 83,841
2D cavity input, three outputs                 84,099
Allen-Cahn Fourier order 8, feature_dim=17     89,601
```

## 4. Shared training audit

The current shared settings are:

```text
backend       = PyTorch through DeepXDE
activation    = tanh
initializer   = Xavier/Glorot normal
initial lr    = 0.001
lr schedule   = StepLR(gamma=0.9, step_size=2000)
precision     = float64
```

The paper explicitly gives `tanh`, Xavier-normal, initial learning rate `0.001`, and decay
rate `0.9` every 2000 Adam iterations. DeepXDE's PyTorch backend implements the current
schedule as a staircase `StepLR`, which is consistent with multiplying the learning rate by
0.9 every 2000 iterations.

The paper does not state the floating-point precision. `float64` is therefore a project-level
numerical choice, not a confirmed paper hyperparameter.

For experiments with `NS3=NS4=10000`, the current baseline maps those stages to one
20000-step standard Adam phase. This excludes both Stage 1 backbone pretraining and Stage 2
sub-network pretraining. Excluding Stage 2 is required for a classical PINN, but whether the
Stage 1 backbone updates should be added to the baseline budget is not explicitly stated.
The mapping is plausible and consistent with the near-equal costs in Table 2, but the paper
does not separately print the exact classical-PINN iteration count for every equation. The
PDF also does not make clear whether the Adam state and learning-rate schedule are continued
or restarted between Stages 3 and 4; the current baseline uses one continuous Adam phase.

## 5. Per-equation comparison

| Equation | Current backbone | Points and sampling | Current optimization | Audit status |
|---|---|---|---|---|
| 1D Poisson, boundary-layer/steep | Modified MLP 6x128 | `Nr=500` LHS, two endpoints | Adam 40000, no L-BFGS; `wr=1`, `wb=100` | Architecture, points, fixed weights, and mapped Adam budget agree with the paper. The endpoint loss uses the sum in Eq. 53. Test grid and float precision are not specified by the paper. |
| 1D Helmholtz | Modified MLP 6x128 | `Nr=500` LHS, two endpoints, plus one observation | Adam 20000, no L-BFGS; residual/boundary/observation weights `1/100/100` | **Not the strict printed paper problem.** The current `u(1/40)=1` observation fixes the non-unique eigenfunction amplitude but is absent from the paper. Results from this script cannot be reported as a direct reproduction of the Table 1 classical PINN without an explicit qualification. |
| 2D Poisson, peak/steep | Modified MLP 6x128 | `Nr=2500` uniform interior grid, `Nbc=50` total perimeter points | Adam 20000 + L-BFGS 1000; `wr=1`, `wb=100` | Aligned after the confirmed missing `y^a` correction and confirmed steep domain `[-1,1]^2`. Paper does not specify the error-test grid. |
| 2D Helmholtz | Modified MLP 6x128 | `101x101` residual mesh, 50 points per edge | Adam 20000 + L-BFGS 1000; `wr=1`, `wb=100` | Aligned with confirmed `a1=1`, `a2=4`, `k=1`. Paper does not specify the error-test grid or detailed L-BFGS tolerances. |
| Lid-driven cavity, Re=100 | Modified MLP 6x128, three outputs | `50x50` residual mesh, 128 points per edge | Adam 20000 + L-BFGS 1000; residual weight 1 and boundary weight 100 | Backbone, point counts, equations, and optimizer stages agree. The code excludes corners from boundary samples; the paper does not state its corner convention. The exact reference field and L2 definition remain unconfirmed. |
| 1D Burgers | Modified MLP 6x128 | `Nr=25600`, `Nbc=256` per side, `Nic=512`, fixed LHS | Adam 60000, no L-BFGS; all three loss weights 1 | Network size, point counts, and Adam budget are externally confirmed project settings. The paper subsection itself does not state the backbone, point counts, fixed loss weights, Fourier use, or L-BFGS setting. The current all-one losses and no-L-BFGS choice remain unverified against the authors' exact baseline. |
| 1D Allen-Cahn | Modified MLP 6x128 with Fourier order 8 | `Nr=25600`, `Nic=512`, fixed LHS; periodic BC hard-enforced | Adam 300000, no L-BFGS; `wr=1`, `wi=100` | The current temporary map is `[t,cos(k*pi*x),sin(k*pi*x)]`, `k=1,...,8`. Reference [21]'s Allen-Cahn setup uses `m=10`, while the DASA PDF omits the order; the DASA authors' actual order remains unconfirmed. |
| 1D Diffusion-Reaction | Modified MLP 6x128 with Fourier order 8 | `Nr=25600`, `Nbc=256` per side, `Nic=512`, fixed LHS | Adam 20000 + L-BFGS 1000; `wr=1`, `wb=wi=100` | Points, network size, losses, and optimizer stages agree. The paper states Fourier embedding but omits its explicit form and order; order 8 is a project inference based on the highest frequency in the solution. |
| 1D Advection-Diffusion | Modified MLP 6x128 with Fourier order 8 | `Nr=25600`, `Nbc=256` per side, `Nic=512`, fixed LHS | Adam 20000 + L-BFGS 1000; `wr=1`, `wb=wi=100` | Aligned with confirmed `kappa=0.01` and the confirmed Diffusion-Reaction Fourier convention. The paper does not state its Table 1 test grid. |
| 1D discontinuous Advection | Modified MLP 6x128 | `Nr=25600`, `Nbc=256` per side, `Nic=512`, fixed LHS | Adam 20000 + L-BFGS 1000; `wr=1`, `wb=wi=100` | Backbone, fixed losses, and optimizer stages agree with the paper; point counts are externally confirmed settings. The paper does not state its Table 1 test grid. |
| L-shaped Poisson | Modified MLP 6x128 | `Nr=10000` LHS, `128x8=1024` boundary points | Adam 20000 + L-BFGS 1000; `wr=1`, `wb=100` | Points, losses, and optimizer stages agree. The subsection does not separately state the backbone size, so 6x128 follows the shared-architecture interpretation. The local finite-difference reference is not the paper's adaptive FEALPy reference. |

## 6. Findings requiring action

### High priority

1. The current 1D Helmholtz launcher executes the version with the non-paper observation
   `u(1/40)=1`. Keep this as a well-posed diagnostic variant, but do not label its metrics as
   the strict Table 1 baseline. The original printed equation and the amplitude-fixed variant
   need separate experiment names and result directories.

2. Existing legacy result files contain descriptive metadata but no source-code hash, network
   parameter count, or saved model checkpoint. The original `common.py` is now retained
   unchanged, while new runs record `version=modified_mlp_corrected`, hidden/gate counts,
   the formula reference, and parameter count.

### Medium priority

3. Confirm the 1D Burgers fixed loss weights and L-BFGS setting. These are the largest
   remaining training-protocol uncertainties that can change optimization behavior rather
   than only the reported metric grid.

4. Confirm whether the classical-PINN Adam budget should include the paper's Stage 1
   backbone updates and whether the Stage 3/4 learning-rate schedule restarts. The current
   `NS3+NS4` continuous-phase mapping is an explicit convention, not a directly printed
   classical-PINN schedule.

5. The shared formula correction is complete. New official runs must use
   `common_modified_mlp_corrected.py` and the separate
   `runs/multiseed_modified_mlp_corrected` output root; historical runs remain under their
   original directories.

6. Confirm the Allen-Cahn Fourier order. Reference [21] uses `m=10`, while the current
   temporary project setting is `m=8`; the DASA PDF states only "Fourier feature
   embedding". Also treat the
   Diffusion-Reaction Fourier order, cavity corner convention, and L-BFGS tolerances as
   explicit assumptions until confirmed.

### Metric-only limitations

7. Unknown Table 1 test grids affect exact numerical comparability but do not prevent the
   project from computing a valid relative L2 error on its declared grid.

8. The Allen-Cahn generated spectral field, L-shaped finite-difference field, and cavity CSV
   field permit internal evaluation, but they are not all confirmed to be byte-for-byte or
   method-for-method identical to the paper's references.

## 7. Current conclusion

The current equation scripts now use a custom gated modified MLP whose forward pass follows
the formula printed in reference [21], rather than DeepXDE's default ordinary FNN. Historical
results were generated with the retained legacy implementation and must not be mixed with
new `modified_mlp_corrected` results.

Before rerunning all equations, the next code change should be narrow:

1. keep the corrected modified-MLP layer semantics and formula covered by structural tests;
2. separate strict-paper and amplitude-fixed 1D Helmholtz variants;
3. add reproducibility metadata without changing the mathematical experiments;
4. resolve or explicitly label the remaining per-equation assumptions above.
