## Table 1: Computational cost on the wave equation

| Method | Runs | Trainable parameters | Main-stage Adam time/iteration (ms) | Total Adam-stage time (min) | End-to-end training time (min) |
|---|---:|---:|---:|---:|---:|
| PINNs | 0 | pending | pending | pending | pending |
| DASA-PINNs | 0 | pending | pending | pending | pending |

## Table 2: DASA hyperparameter sensitivity on the wave equation

| Factor | Setting | Runs | Relative L2 error | Maximum pointwise error | Training time (min) |
|---|---|---:|---:|---:|---:|
| Sub-network architecture | 3 x 5 | 0 | pending | pending | pending |
| Sub-network architecture | 5 x 10 | 0 | pending | pending | pending |
| Sub-network architecture | 7 x 20 | 0 | pending | pending | pending |
| Learning rates (eta_b, eta_s) | (5e-4, 5e-4) | 0 | pending | pending | pending |
| Learning rates (eta_b, eta_s) | (1e-3, 5e-4) | 0 | pending | pending | pending |
| Learning rates (eta_b, eta_s) | (1e-3, 1e-3) | 0 | pending | pending | pending |
| Learning rates (eta_b, eta_s) | (1e-3, 2e-3) | 0 | pending | pending | pending |
| Learning rates (eta_b, eta_s) | (2e-3, 2e-3) | 0 | pending | pending | pending |
| Update ratio | 1:5 | 0 | pending | pending | pending |
| Update ratio | 1:1 | 0 | pending | pending | pending |
| Update ratio | 5:1 | 0 | pending | pending | pending |

## Table 3: Four-stage length sensitivity on the wave equation

| Varied stage | (N_S1, N_S2, N_S3, N_S4) | Runs | Relative L2 error | Maximum pointwise error |
|---|---:|---:|---:|---:|
| Stage 1 | (2500, 2000, 50000, 50000) | 5 | 3.379256e-05 +/- 3.724697e-06 | 9.566173e-05 +/- 2.640026e-05 |
| Baseline | (5000, 2000, 50000, 50000) | 5 | 4.042384e-05 +/- 4.544987e-06 | 9.093881e-05 +/- 1.327799e-05 |
| Stage 1 | (10000, 2000, 50000, 50000) | 5 | 3.574927e-05 +/- 8.141932e-06 | 7.140487e-05 +/- 1.438134e-05 |
| Stage 2 | (5000, 1000, 50000, 50000) | 5 | 3.893765e-05 +/- 5.177982e-06 | 8.301139e-05 +/- 1.323773e-05 |
| Stage 2 | (5000, 4000, 50000, 50000) | 5 | 3.811380e-05 +/- 1.297970e-05 | 9.680148e-05 +/- 3.271788e-05 |
| Stage 3 | (5000, 2000, 25000, 50000) | 5 | 4.065303e-05 +/- 4.873177e-06 | 8.906722e-05 +/- 1.079592e-05 |
| Stage 3 | (5000, 2000, 100000, 50000) | 0 | pending | pending |
| Stage 4 | (5000, 2000, 50000, 25000) | 0 | pending | pending |
| Stage 4 | (5000, 2000, 50000, 100000) | 0 | pending | pending |
