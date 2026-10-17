# Stage-length sensitivity: L-shaped Poisson

The four stage lengths are varied one factor at a time. All other training settings are fixed, including the Stage 4 starting learning rate (2.82475249e-05). Network parameters and Adam moment states are retained at the Stage 3--4 transition. Errors are evaluated after Stage 4 and reported as mean +/- sample standard deviation over completed seeds.

| Axis | Variant | (S1, S2, S3, S4) | Seeds | Relative L2 error | Maximum pointwise error |
|---|---|---:|---:|---:|---:|
| baseline | baseline_5000_2000_10000_10000 | (5000, 2000, 10000, 10000) | 3 | 1.422686e-02 +/- 6.382694e-03 | 1.522819e-02 +/- 3.611002e-03 |
| stage1_warmup | stage1_10000 | (10000, 2000, 10000, 10000) | 3 | 1.300091e-02 +/- 5.767970e-03 | 1.448132e-02 +/- 3.391215e-03 |
| stage1_warmup | stage1_2500 | (2500, 2000, 10000, 10000) | 3 | 1.474538e-02 +/- 6.523094e-03 | 1.543246e-02 +/- 3.776592e-03 |
| stage2_subnet_pretraining | stage2_1000 | (5000, 1000, 10000, 10000) | 3 | 1.442400e-02 +/- 6.622441e-03 | 1.532303e-02 +/- 3.710897e-03 |
| stage2_subnet_pretraining | stage2_4000 | (5000, 4000, 10000, 10000) | 3 | 1.442426e-02 +/- 6.127991e-03 | 1.535781e-02 +/- 3.429479e-03 |
| stage3_adversarial | stage3_20000 | (5000, 2000, 20000, 10000) | 3 | 1.394950e-02 +/- 5.884092e-03 | 1.508899e-02 +/- 3.337401e-03 |
| stage3_adversarial | stage3_5000 | (5000, 2000, 5000, 10000) | 3 | 1.488769e-02 +/- 6.950436e-03 | 1.558213e-02 +/- 3.880956e-03 |
| stage4_finetuning | stage4_20000 | (5000, 2000, 10000, 20000) | 3 | 1.124953e-02 +/- 2.405415e-03 | 1.351693e-02 +/- 1.458678e-03 |
| stage4_finetuning | stage4_5000 | (5000, 2000, 10000, 5000) | 3 | 1.125993e-02 +/- 4.996330e-03 | 1.340934e-02 +/- 3.184370e-03 |
