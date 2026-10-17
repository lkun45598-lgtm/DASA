# Cost comparison

| Method | Seeds | Trainable params | Added subnet params | Param overhead | Main Adam time/step (ms) | Adam total (min) | Full training total (min) | Final relative L2 | Max error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DASA | 3 | 84321 | 480 | 0.573% | 46.873 ± 0.953 | 19.361 ± 0.27 | 25.956 ± 1.08 | 0.0084848 ± 0.00474 | 0.0068604 ± 0.000801 |
| PINN | 3 | 83841 | 0 | 0% | 43.266 ± 1.31 | 17.948 ± 0.405 | 24.006 ± 1.1 | 0.0095799 ± 0.00543 | 0.0074072 ± 0.000406 |
