# Superseded Stage 3 rows

This directory preserves the first stage-length run for traceability. Its Stage 3 variants
changed both the number of adversarial updates and the learning rate inherited by Stage 4,
because the scheduler was active only during Stage 3. Those Stage 3 rows must not be used
as one-factor-at-a-time sensitivity results.

Use `../results_stage_length_sensitivity_controlled_lr/` for the formal table. The formal
Stage 3 rerun fixes the Stage 4 starting learning rate at `2.82475249e-05` while retaining
the trained network parameters and Adam moment states.
