# Wave-equation rebuttal experiments

## Experimental setup

### Governing equation and reference solution

We consider the one-dimensional wave equation

$$
\frac{\partial^2 u}{\partial t^2}
-4\frac{\partial^2 u}{\partial x^2}=0,
\qquad (t,x)\in[0,1]\times[0,1],
$$

subject to the initial and boundary conditions

$$
\begin{aligned}
u(0,x) &= \sin(\pi x)+\frac{1}{2}\sin(4\pi x),\\
u_t(0,x) &= 0,\\
u(t,0) &= u(t,1)=0.
\end{aligned}
$$

The corresponding analytical solution used to evaluate the predictions is

$$
u(t,x)=\sin(\pi x)\cos(2\pi t)
+\frac{1}{2}\sin(4\pi x)\cos(8\pi t).
$$

### Network and data configuration

| Component | Configuration |
|---|---|
| Shared backbone | Six hidden layers with 128 neurons per layer and tanh activations |
| Backbone embedding | Spatio-temporal multi-scale Fourier features (ST-MFF) |
| Fourier scales | Low-frequency time: 1; high-frequency time: 10; space: 1 |
| Backbone parameters | 99,329 trainable parameters |
| DASA weighting sub-network | Five hidden layers with 10 neurons per layer and sigmoid activations |
| DASA sub-network parameters | 480 trainable parameters |
| Adam interior points | 256 Latin-hypercube points |
| Adam initial points | 128 points at $t=0$ |
| Adam boundary points | 64 points on each spatial boundary |
| Adam resampling | Collocation and condition points resampled every 100 iterations |
| L-BFGS interior points | 25,600 fixed Latin-hypercube points |
| L-BFGS initial points | 512 fixed points at $t=0$ |
| L-BFGS boundary points | 256 fixed points on each spatial boundary |
| Evaluation grid | $501\times501=251{,}001$ uniformly spaced points |

For DASA-PINNs, the weighting sub-network generates nonnegative pointwise
residual weights that are normalized to have unit mean within each batch. The
PINN and DASA-PINN models use the same ST-MFF backbone, sampling strategy,
condition losses, and evaluation grid; DASA-PINNs adds only the weighting
sub-network and its adversarial updates.

The training objective is

$$
\mathcal{L}
=\mathcal{L}_{r}
+10^4\mathcal{L}_{u_t(0,x)}
+10^4\left(
\mathcal{L}_{u(0,x)}
+\mathcal{L}_{u(t,0)}
+\mathcal{L}_{u(t,1)}
\right).
$$

For PINNs, $\mathcal{L}_{r}$ is the unweighted residual mean-squared error.
For DASA-PINNs, it is the mean-squared error of the pointwise weighted
residuals.

### Optimization protocol

Adam uses $\beta_1=0.9$, $\beta_2=0.999$, and an initial learning rate of
$10^{-3}$ for both the backbone and the weighting sub-network. The default
four-stage schedule is summarized below.

| Stage | Iterations | Updated variables | Objective and learning-rate schedule |
|---|---:|---|---|
| Stage 1 | 5,000 | Backbone | Unweighted PINN warm-up with Adam at $10^{-3}$ |
| Stage 2 | 2,000 | Weighting sub-network | Backbone frozen; adversarial weights pretrained by gradient ascent |
| Stage 3 | 50,000 | Backbone and weighting sub-network | Simultaneous backbone descent and sub-network ascent with a 1:1 update ratio |
| Stage 4 | 50,000 | Backbone | Unweighted backbone fine-tuning |
| Final refinement | 1,000 outer calls | Backbone | Full-batch L-BFGS |

During Stage 3, the backbone learning rate is multiplied by 0.9 every 1,000
iterations. Stage 4 starts from the controlled learning rate
$5.153775\times10^{-6}$. The final L-BFGS optimizer uses a learning rate of
0.8, at most 100 internal iterations per outer call, a history size of 100,
a gradient tolerance of $10^{-5}$, and a parameter-change tolerance of
$10^{-9}$.

### Comparison and sensitivity protocols

- **Computational cost (Table 1):** PINNs and DASA-PINNs use the same backbone,
  data, loss coefficients, backbone optimizer, and evaluation grid. PINNs has
  no weighting sub-network or Stage-2 sub-network pretraining. Timing results
  were collected on the same NVIDIA A100-SXM4-80GB platform.
- **Hyperparameter sensitivity (Table 2):** the sub-network architecture,
  backbone/sub-network learning rates, or sub-network-to-backbone update ratio
  is varied while all other settings remain fixed. These experiments evaluate
  the model immediately after Stage 3 and therefore do not include Stage 4 or
  L-BFGS. The update ratio is defined as sub-network updates to backbone
  updates.
- **Stage-length sensitivity (Table 3):** one stage length is varied at a time
  while the other three stages retain the default schedule. All configurations
  use the same final L-BFGS refinement. When the Stage-3 length is varied, the
  Stage-4 starting learning rate remains fixed at the default Stage-3 endpoint,
  preventing the Stage-3 learning-rate budget from confounding the Stage-4
  comparison.

### Random seeds and reproducibility

Independent runs use predefined sequential random seeds starting from zero;
the number of independent runs used for each reported configuration is
given in the **Runs** column of the corresponding table. Each seed controls the
Python, NumPy, and PyTorch random-number generators, including CUDA
initialization. It therefore determines the network initialization, random
Fourier frequencies, Latin-hypercube samples, and deterministic resampling
streams. CUDA seeds are set on all visible devices, cuDNN benchmarking is
disabled, and deterministic cuDNN execution is enabled. For a fixed seed,
the sampling stream is derived deterministically from the training stage and
resampling-block index.

### Evaluation metrics

The two reported accuracy metrics are

$$
\mathrm{Relative\ L2}
=\frac{\lVert u_{\mathrm{ref}}-u_{\mathrm{pred}}\rVert_2}
{\lVert u_{\mathrm{ref}}\rVert_2},
\qquad
\mathrm{Maximum\ pointwise\ error}
=\max_i\left|u_{\mathrm{ref},i}-u_{\mathrm{pred},i}\right|.
$$

All summary values are reported as the mean $\pm$ sample standard deviation
over independent random-seed runs.

## Table 1: Computational cost on the wave equation

| Method | Runs | Trainable parameters | Main-stage Adam time/iteration (ms) | Total Adam-stage time (min) | End-to-end training time (min) |
|---|---:|---:|---:|---:|---:|
| PINNs | 3 | 99,329 | 109.027 +/- 0.736 | 190.669 +/- 1.131 | 252.809 +/- 23.502 |
| DASA-PINNs | 3 | 99,809 | 113.058 +/- 1.983 | 195.707 +/- 1.943 | 286.692 +/- 44.266 |

![PINN and DASA-PINN end-to-end training time on the L-shaped Poisson and wave equations](../../../../rebuttal/L_shape_Poisson_review_experiments/results_cost_classical_pinn_vs_dasa_full/summaries/training_time_lshape_wave_pinn_vs_dasa.png)

**Figure: Stage-specific training time on the L-shaped Poisson and wave
equations.** Lines connect the mean time of S1, S2, S3, S4, and L-BFGS over
three seeds.
`Pre-train*` is a DASA-only weighting-subnetwork stage and is therefore not
applicable to PINN.
The panel annotation reports the complete training time. Timing includes all
configured Adam stages and the complete L-BFGS stage on an NVIDIA A100-SXM4-80GB.

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
| Stage 1 | (2500, 2000, 50000, 50000) | 10 | 3.651173e-05 +/- 5.984821e-06 | 9.615375e-05 +/- 2.418776e-05 |
| Baseline | (5000, 2000, 50000, 50000) | 10 | 4.035737e-05 +/- 8.402686e-06 | 9.625173e-05 +/- 2.228142e-05 |
| Stage 1 | (10000, 2000, 50000, 50000) | 10 | 3.808763e-05 +/- 9.673206e-06 | 7.852241e-05 +/- 1.861636e-05 |
| Stage 2 | (5000, 1000, 50000, 50000) | 10 | 3.715343e-05 +/- 7.932953e-06 | 8.536355e-05 +/- 2.041789e-05 |
| Stage 2 | (5000, 4000, 50000, 50000) | 10 | 3.660772e-05 +/- 1.059581e-05 | 9.814752e-05 +/- 3.108410e-05 |
| Stage 3 | (5000, 2000, 25000, 50000) | 10 | 3.668877e-05 +/- 6.709303e-06 | 8.199980e-05 +/- 1.247150e-05 |
| Stage 3 | (5000, 2000, 100000, 50000) | 10 | 3.512900e-05 +/- 4.241870e-06 | 7.694719e-05 +/- 8.494323e-06 |
| Stage 4 | (5000, 2000, 50000, 25000) | 10 | 3.371882e-05 +/- 4.754910e-06 | 7.932896e-05 +/- 1.134756e-05 |
| Stage 4 | (5000, 2000, 50000, 100000) | 5 | 3.234194e-05 +/- 2.909702e-06 | 7.722974e-05 +/- 1.503746e-05 |

## Result analysis

Table 1 shows that the DASA weighting sub-network introduces only 480
additional trainable parameters, corresponding to a 0.48% increase over the
shared backbone. Relative to PINNs, DASA-PINNs increases the main-stage Adam
time per iteration by approximately 3.7% and the total Adam-stage time by
approximately 2.6%. This confirms that the adaptive adversarial weighting
mechanism adds limited parameter and per-iteration overhead.

Table 2 indicates that the baseline $5\times10$ weighting sub-network is
competitive with both smaller and larger alternatives. Moderate learning
rates produce errors of the same order, whereas simultaneously increasing
both learning rates to $2\times10^{-3}$ degrades accuracy. The balanced 1:1
update ratio provides the most stable accuracy-cost trade-off; infrequent
sub-network updates increase the variability, while five sub-network updates
per backbone update substantially increase both error and training time.
These results support the baseline architecture, learning rates, and update
ratio without relying on aggressive hyperparameter tuning.

Table 3 shows that, across the tested four-stage schedules, the relative L2
error remains within $3.23\times10^{-5}$ to $4.04\times10^{-5}$, while the
maximum pointwise error remains within $7.69\times10^{-5}$ to
$9.81\times10^{-5}$. All stage-length configurations therefore retain the
same accuracy order without unstable degradation, demonstrating robustness to
moderate changes in the duration of each training stage. Increasing the
Stage-3 or Stage-4 budget can further reduce the mean error, but the
improvement is not proportional to the additional iterations. The default
schedule consequently provides a reasonable balance between predictive
accuracy and computational cost.
