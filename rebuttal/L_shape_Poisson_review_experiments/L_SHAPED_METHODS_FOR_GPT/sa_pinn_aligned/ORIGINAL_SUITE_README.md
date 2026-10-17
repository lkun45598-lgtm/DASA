# DASA-Aligned SA-PINNs Benchmarks

This suite runs the original direct point-mask SA-PINNs method on the 13 forward PDE cases
aligned with the DASA-PINNs paper. It does not implement the DASA weight-prediction
subnetwork. All solution models use the approved six hidden layers by 128 neurons, `tanh`,
Xavier-normal initialization, and the configured modified MLP/Fourier input mapping.

## Environment

Run commands from the repository root with `.conda-env/bin/python`. No package beyond the
project's existing TensorFlow, NumPy, SciPy, and pyDOE environment is required. The `.yaml`
case files contain JSON syntax and are parsed by the Python standard library.

## Reference Data

Burgers and Allen-Cahn use the original SA-PINN datasets at
`Burgers/burgers_shock.mat` and `Allen-Cahn/AC.mat`. The cavity case uses the
`100x100` horizontal-velocity data from the official repository for reference [19] in the
DASA paper (`PredictiveIntelligenceLab/GradientPathologiesPINNs`, commit `93e752b`). Its
vendored SHA-256 is `1a93d210374c6e512ef2f7a9fee95307734b9c2eb77ad087c2e76db6b115de50`.

Only L-shaped Poisson requires a generated numerical reference:

```bash
.conda-env/bin/python -m benchmarks.dasa_aligned_sa.scripts.generate_references --all
```

The L-shaped reference manifest records three grid levels, both convergence comparisons,
solver parameters, generator hash, and reference SHA-256. Training rejects a missing or
modified L-shaped reference.

## Tests

```bash
.conda-env/bin/python -m unittest discover -s benchmarks/dasa_aligned_sa/tests -v
.conda-env/bin/python -m compileall -q benchmarks/dasa_aligned_sa
bash -n benchmarks/dasa_aligned_sa/scripts/launch_full_tmux.sh
```

After the L-shaped reference exists, run all 13 cases with two Adam steps and one L-BFGS
iteration:

```bash
bash benchmarks/dasa_aligned_sa/scripts/run_smoke_matrix.sh
```

The smoke script accepts an optional new output directory as its first argument. It refuses
to reuse an existing directory.

## One Case

Use budget overrides only for tests and smoke checks:

```bash
CUDA_VISIBLE_DEVICES=5 .conda-env/bin/python \
  -m benchmarks.dasa_aligned_sa.scripts.run_case \
  --case helmholtz_1d --seed 0 \
  --output benchmarks/dasa_aligned_sa/runs/helmholtz_1d_smoke \
  --adam-steps 2 --lbfgs-steps 1
```

For the configured full budget, omit both override flags.

## Full Seed-0 Run

The full launcher starts all 13 experiments immediately in separate tmux windows using the
approved eight-GPU mapping:

```bash
bash benchmarks/dasa_aligned_sa/scripts/launch_full_tmux.sh
tmux attach -t sa_dasa_full_seed0
```

Detach with tmux's standard `Ctrl-b d`. The launcher refuses to replace an existing session.
It accepts an optional new run root as its first argument.

Inspect status without attaching:

```bash
.conda-env/bin/python -m benchmarks.dasa_aligned_sa.scripts.monitor_runs \
  --root benchmarks/dasa_aligned_sa/runs/seed0
```

Aggregate completed results:

```bash
.conda-env/bin/python -m benchmarks.dasa_aligned_sa.scripts.aggregate_results \
  --root benchmarks/dasa_aligned_sa/runs/seed0
```

`results.csv` and `results.md` contain only `case`, `seed`, `l2_relative_error`, and
`maximum_pointwise_error`. Failures are reported separately. Do not launch repeated seeds
until the complete seed-0 metric table has been reviewed.

## Run Artifacts

Each case directory contains resolved configuration and provenance, progress JSONL, model
weights, the training log, and either `metrics.json` or `failure.json`. `metrics.json` has
exactly the case ID, seed, relative L2 error, and maximum pointwise error.
