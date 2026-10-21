# Stage-length sensitivity: L-shaped Poisson

The four stage lengths are varied one factor at a time. All other training settings are fixed, including the Stage 4 starting learning rate (5.904900e-04). Network parameters and Adam moment states are retained at the Stage 3--4 transition. Errors are evaluated after the common L-BFGS refinement and reported as mean +/- sample standard deviation over completed seeds.

| Axis | Variant | (S1, S2, S3, S4) | Seeds | Relative L2 error | Maximum pointwise error |
|---|---|---:|---:|---:|---:|
| baseline | baseline_5000_2000_10000_10000 | (5000, 2000, 10000, 10000) | 8 | 3.496530e-03 +/- 1.671198e-03 | 5.075910e-03 +/- 1.444202e-03 |
| stage1_warmup | stage1_10000 | (10000, 2000, 10000, 10000) | 10 | 5.303326e-03 +/- 3.956646e-03 | 5.720407e-03 +/- 1.491437e-03 |
| stage1_warmup | stage1_2500 | (2500, 2000, 10000, 10000) | 10 | 6.343157e-03 +/- 4.738431e-03 | 6.082287e-03 +/- 2.172435e-03 |
| stage2_subnet_pretraining | stage2_1000 | (5000, 1000, 10000, 10000) | 10 | 5.326818e-03 +/- 3.825478e-03 | 5.588238e-03 +/- 1.225201e-03 |
| stage2_subnet_pretraining | stage2_4000 | (5000, 4000, 10000, 10000) | 10 | 5.161844e-03 +/- 3.997712e-03 | 5.704561e-03 +/- 1.498852e-03 |
| stage3_adversarial | stage3_20000 | (5000, 2000, 20000, 10000) | 10 | 4.410008e-03 +/- 3.032476e-03 | 5.342368e-03 +/- 1.230757e-03 |
| stage3_adversarial | stage3_5000 | (5000, 2000, 5000, 10000) | 10 | 5.044061e-03 +/- 3.461277e-03 | 5.501484e-03 +/- 1.418873e-03 |
| stage4_finetuning | stage4_20000 | (5000, 2000, 10000, 20000) | 10 | 4.772101e-03 +/- 2.952129e-03 | 5.392554e-03 +/- 1.359936e-03 |
| stage4_finetuning | stage4_5000 | (5000, 2000, 10000, 5000) | 10 | 5.856680e-03 +/- 4.419372e-03 | 5.897598e-03 +/- 1.804688e-03 |
