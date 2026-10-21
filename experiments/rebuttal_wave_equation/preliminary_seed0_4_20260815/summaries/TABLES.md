> **Superseded. Do not use this table.** This version incorrectly restricted every Stage-length configuration to seeds 0-4. Use `preliminary_stage4_seed0_4_20260815/summaries/TABLES.md` instead.

## Table 1: Computational cost on the wave equation

| Method | Runs | Trainable parameters | Main-stage Adam time/iteration (ms) | Total Adam-stage time (min) | End-to-end training time (min) |
|---|---:|---:|---:|---:|---:|
| PINNs | 3 | 99,329 | 109.027 +/- 0.736 | 190.669 +/- 1.131 | 252.809 +/- 23.502 |
| DASA-PINNs | 3 | 99,809 | 113.058 +/- 1.983 | 195.707 +/- 1.943 | 286.692 +/- 44.266 |

## Table 2: DASA hyperparameter sensitivity on the wave equation

| Factor | Setting | Runs | Relative L2 error | Maximum pointwise error | Training time (min) |
|---|---|---:|---:|---:|---:|
| Sub-network architecture | 3 x 5 | 3 | 4.510252e-03 +/- 1.894096e-03 | 9.056012e-03 +/- 6.338945e-03 | 105.417 +/- 1.889 |
| Sub-network architecture | 5 x 10 | 3 | 3.090577e-03 +/- 1.211081e-04 | 5.400315e-03 +/- 1.177591e-04 | 105.640 +/- 0.946 |
| Sub-network architecture | 7 x 20 | 3 | 3.304567e-03 +/- 3.283452e-04 | 5.490844e-03 +/- 9.533670e-04 | 107.796 +/- 1.103 |
| Learning rates (eta_b, eta_s) | (5e-4, 5e-4) | 3 | 3.061244e-03 +/- 8.393584e-04 | 5.678112e-03 +/- 2.254019e-03 | 106.924 +/- 2.224 |
| Learning rates (eta_b, eta_s) | (1e-3, 5e-4) | 3 | 7.109344e-03 +/- 2.579298e-03 | 1.366164e-02 +/- 6.173933e-03 | 107.153 +/- 1.860 |
| Learning rates (eta_b, eta_s) | (1e-3, 1e-3) | 3 | 3.090577e-03 +/- 1.211081e-04 | 5.400315e-03 +/- 1.177591e-04 | 105.640 +/- 0.946 |
| Learning rates (eta_b, eta_s) | (1e-3, 2e-3) | 3 | 4.504044e-03 +/- 2.377810e-03 | 9.304677e-03 +/- 5.106616e-03 | 106.837 +/- 1.817 |
| Learning rates (eta_b, eta_s) | (2e-3, 2e-3) | 3 | 2.143137e-02 +/- 1.171300e-02 | 4.134826e-02 +/- 1.907456e-02 | 107.403 +/- 1.976 |
| Update ratio | 1:5 | 3 | 8.959973e-03 +/- 1.074692e-02 | 1.478628e-02 +/- 1.446041e-02 | 104.458 +/- 0.127 |
| Update ratio | 1:1 | 3 | 3.090577e-03 +/- 1.211081e-04 | 5.400315e-03 +/- 1.177591e-04 | 105.640 +/- 0.946 |
| Update ratio | 5:1 | 3 | 7.509418e-02 +/- 3.821969e-02 | 1.386127e-01 +/- 6.468439e-02 | 317.522 +/- 1.178 |

## Table 3: Four-stage length sensitivity on the wave equation

| Varied stage | (N_S1, N_S2, N_S3, N_S4) | Runs | Relative L2 error | Maximum pointwise error |
|---|---:|---:|---:|---:|
| Stage 1 | (2500, 2000, 50000, 50000) | 5 | 3.569205e-05 +/- 6.384268e-06 | 1.028351e-04 +/- 2.796867e-05 |
| Baseline | (5000, 2000, 50000, 50000) | 5 | 3.832490e-05 +/- 6.501263e-06 | 8.896161e-05 +/- 2.026104e-05 |
| Stage 1 | (10000, 2000, 50000, 50000) | 5 | 3.698017e-05 +/- 6.861678e-06 | 7.608384e-05 +/- 1.463771e-05 |
| Stage 2 | (5000, 1000, 50000, 50000) | 5 | 3.690320e-05 +/- 6.842542e-06 | 8.827599e-05 +/- 2.001189e-05 |
| Stage 2 | (5000, 4000, 50000, 50000) | 5 | 3.191242e-05 +/- 8.549041e-06 | 1.016274e-04 +/- 3.886193e-05 |
| Stage 3 | (5000, 2000, 25000, 50000) | 5 | 3.683741e-05 +/- 6.785181e-06 | 8.118153e-05 +/- 1.293697e-05 |
| Stage 3 | (5000, 2000, 100000, 50000) | 5 | 3.281753e-05 +/- 2.707612e-06 | 7.562041e-05 +/- 7.783874e-06 |
| Stage 4 | (5000, 2000, 50000, 25000) | 5 | 3.345714e-05 +/- 3.783665e-06 | 8.003116e-05 +/- 8.292000e-06 |
| Stage 4 | (5000, 2000, 50000, 100000) | 4 | 3.324745e-05 +/- 2.412920e-06 | 8.144975e-05 +/- 1.351951e-05 |
