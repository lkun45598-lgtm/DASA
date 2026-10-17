# Stage-length sensitivity: L-shaped Poisson

The four stage lengths are varied one factor at a time. All other training settings are fixed. Errors are evaluated after Stage 4 and reported as mean +/- sample standard deviation over completed seeds.

| Axis | Variant | (S1, S2, S3, S4) | Seeds | Relative L2 error | Maximum pointwise error |
|---|---|---:|---:|---:|---:|
| baseline | baseline_5000_2000_10000_10000 | (5000, 2000, 10000, 10000) | 3 | 1.422686e-02 +/- 6.382694e-03 | 1.522819e-02 +/- 3.611002e-03 |
| stage3_adversarial | stage3_20000 | (5000, 2000, 20000, 10000) | 3 | 1.394950e-02 +/- 5.884092e-03 | 1.508899e-02 +/- 3.337401e-03 |
| stage3_adversarial | stage3_5000 | (5000, 2000, 5000, 10000) | 3 | 1.488769e-02 +/- 6.950436e-03 | 1.558213e-02 +/- 3.880956e-03 |
