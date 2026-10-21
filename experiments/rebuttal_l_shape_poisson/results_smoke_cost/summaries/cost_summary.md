# Cost comparison

| Method | Seeds | Trainable params | Added subnet params | Param overhead | Main Adam time/step (ms) | Adam total (min) | Full training total (min) | Final relative L2 | Max error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DASA | 1 | 84321 | 480 | 0.573% | 51.12 ± 0 | 0.0087981 ± 0 | 0.0087981 ± 0 | 4.7581 ± 0 | 0.49753 ± 0 |
| PINN | 1 | 83841 | 0 | 0% | 85.889 ± 0 | 0.011652 ± 0 | 0.011652 ± 0 | 4.8241 ± 0 | 0.49155 ± 0 |
