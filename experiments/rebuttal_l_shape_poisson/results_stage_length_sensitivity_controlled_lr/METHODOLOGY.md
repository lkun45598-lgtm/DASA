# Controlled stage-length sensitivity

- Default schedule: `(5000, 2000, 10000, 10000)`.
- Design: one factor at a time, three seeds per configuration.
- Backbone: 6x128 gated Modified MLP.
- Stage 4 starting learning rate: `2.82475249e-05` for every configuration.
- Network parameters and Adam moment states are retained at the Stage 3--4 transition.
- L-BFGS is disabled.
- Reported metrics: relative L2 error and maximum pointwise error after Stage 4.

The Stage 1, Stage 2, and Stage 4 variants use the original runs because their Stage 3
length is the default 10,000 steps and therefore already enters Stage 4 at the controlled
learning rate. The Stage 3 low/default/high rows use the dedicated reruns from
`../results_stage3_length_controlled_lr/`.
