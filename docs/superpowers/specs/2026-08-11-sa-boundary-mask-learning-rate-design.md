# SA-PINN Boundary-Layer Mask Learning Rate

## Goal

Stabilize the DASA-aligned SA-PINN solution of the 1D Poisson boundary-layer
benchmark while leaving its equation, sampling, network, loss coefficients, and
training budget unchanged.

The accepted success criterion is that the corrected seed-0 run reproduces the
validated probe behavior: relative L2 error near the paper's SA-PINN value of
`9.51e-02`, followed by a complete ten-seed run for the final statistics.

## Evidence

The existing configuration uses `1e-3` for both the solution-network Adam and
the direct SA-mask Adam. The completed seed-0 probes changed only the initial
mask learning rate:

| Initial mask learning rate | Relative L2 error | Maximum pointwise error |
|---:|---:|---:|
| `1e-4` | `7.265038e-01` | `1.605873e-01` |
| `1e-5` | `9.644182e-02` | `2.056142e-02` |
| `1e-6` | `1.046353e+00` | `2.234650e-01` |

The selected value is `1e-5`. It is an SA-specific optimizer setting not stated
in the DASA paper; it does not change a PDE coefficient.

## Design

`OptimizerConfig` gains `mask_learning_rate`. Case files may omit the field; in
that case the loader resolves it to the ordinary network `learning_rate`, which
preserves all existing behavior.

Only `poisson_1d_boundary.yaml` explicitly sets `mask_learning_rate` to `1e-5`.
The trainer initializes the mask Adam from this value. During training, its
learning rate follows the same multiplicative decay factor as the network Adam:

```text
mask_rate(step) = network_rate(step)
                  * mask_initial_rate / network_initial_rate
```

No mask normalization, clipping, hard boundary embedding, extra collocation
points, checkpoint selection, or equation scaling is introduced.

## Files

- `benchmarks/dasa_aligned_sa/dasa_sa/config.py`
- `benchmarks/dasa_aligned_sa/dasa_sa/trainer.py`
- `benchmarks/dasa_aligned_sa/configs/cases/poisson_1d_boundary.yaml`
- Focused assertions in the existing configuration/trainer tests

The remaining 12 case files and all PDE problem implementations remain
unchanged.

## Verification

1. A test first requires boundary-layer mask learning rate `1e-5`, default
   fallback for another case, and preservation of the model-to-mask learning
   rate ratio after decay.
2. The focused test must fail before implementation and pass afterward.
3. The modified production files are synchronized to A100.
4. Ten seeds run in tmux over GPUs 0-7.
5. Report only relative L2 error and maximum pointwise error in scientific
   notation, with all seeds included unless a failure is explicitly documented.

## Risks

The seed-0 probe does not guarantee low variance over ten seeds. Therefore the
final ten-seed distribution, not the probe, determines whether the change is
robust. The result must identify the tuned SA mask learning rate so the
comparison remains reproducible.
