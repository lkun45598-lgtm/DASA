# Classical PINN Backbone for the L-shaped Poisson Cost Comparison

## Goal

Replace the PINN baseline backbone in the L-shaped Poisson rebuttal experiment with the classical fully connected feed-forward PINN architecture described by Raissi, Perdikaris, and Karniadakis (JCP 2019), while leaving DASA's Modified MLP unchanged.

The new comparison must report trainable parameters, main-stage time per iteration, Adam time, end-to-end Adam plus L-BFGS time, relative L2 error, and maximum pointwise error over seeds 0, 1, and 2.

## Architecture Interpretation

The original PINN paper uses simple deep feed-forward neural networks with hyperbolic tangent activations and no gated Modified MLP encoders. Network depth and width vary by example. For this comparison, model capacity follows the existing DASA L-shaped Poisson setup requested by the user:

- input dimension: 2;
- six hidden layers;
- 128 neurons per hidden layer;
- `tanh` after every hidden affine layer;
- scalar output;
- Xavier-normal weight initialization;
- trainable biases;
- no encoder U/V branches, gating, Fourier features, dropout, or other regularization.

## Implementation

Add a `VanillaMLP` next to the existing `ModifiedMLP`. Select the backbone through an explicit `backbone.type` configuration value instead of coupling architecture selection to the method name:

- `cost_pinn.json`: `vanilla_mlp`;
- `cost_dasa.json`: `modified_mlp`.

Both methods continue to use the same L-shaped domain, collocation and boundary sampling protocol, loss coefficients, backbone update count, learning-rate schedule, and L-BFGS configuration. DASA continues to use the paper-consistent unit-mean pointwise weight scaling.

## Experiment Outputs

Run only the three new classical PINN seeds because the corrected DASA seeds are already complete. Store the new runs in a separate result root and combine them with copies of the existing DASA results for aggregation. Do not overwrite the earlier same-Modified-MLP controlled comparison.

The rebuttal material will retain two interpretations:

1. Classical PINN versus DASA: the user-requested method-level accuracy and runtime comparison. Runtime differences include both the Modified MLP and subnet.
2. Modified-MLP PINN versus DASA: the controlled measurement that isolates the incremental DASA subnet overhead.

The primary Comment 4 table will use the classical PINN comparison. Its prose must not attribute the entire runtime or parameter difference solely to the subnet.

## Verification

Add one focused architecture test that checks the vanilla network has six hidden affine layers, uses `tanh`, produces the expected output shape, and has no gated encoder branches. Keep the existing DASA unit-mean regression test.

Run the local focused test suite, then execute seeds 0, 1, and 2 with full Adam and L-BFGS training on A100 GPUs. Aggregate only after all three `metrics.json` files exist, and verify from raw metrics that the reported means and standard deviations match the generated table.

## Risks

- The method-level comparison is not an architecture-controlled estimate of subnet overhead. The rebuttal must distinguish it from the same-backbone controlled comparison.
- The classical PINN paper does not prescribe a single universal depth and width. Six by 128 is retained from the L-shaped DASA experiment to avoid changing capacity and architecture family simultaneously.
- Existing DASA and same-backbone PINN results must remain untouched for traceability.
