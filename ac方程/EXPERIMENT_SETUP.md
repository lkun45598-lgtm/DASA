# Experiment setup

## Baseline definition

This project uses the ordinary fully connected PINN network family from the original PINNs
formulation, while matching the DASA-PINNs PDE experiments as closely as possible.

The shared network is:

```text
type            = plain fully connected neural network
hidden layers   = 6
width           = 128
activation      = tanh
initializer     = Xavier/Glorot normal
precision       = float64
```

It has no modified-MLP encoders or gates, no DASA sub-network, no pointwise adaptive loss
weights, and no adversarial optimization.

## Experiment table

| Experiment | Training points | Fixed loss weights | Optimization |
|---|---|---|---|
| 1D Poisson boundary-layer/steep | `Nr=500`, two endpoints, LHS | `wr=1`, `wb=100`; endpoint sum | Adam 40000 |
| 1D Helmholtz | `Nr=500`, two endpoints, LHS | `wr=1`, `wb=100` | Adam 20000 |
| 2D Poisson peak/steep | `Nr=2500` uniform, `Nbc=50` total | `wr=1`, `wb=100` | Adam 20000 + L-BFGS 1000 |
| 2D Helmholtz | `101x101` residual mesh, 50 points per edge | `wr=1`, `wb=100` | Adam 20000 + L-BFGS 1000 |
| Lid-driven cavity Re=100 | `50x50` residual mesh, 128 points per edge | residual `1`, boundary `100` | Adam 20000 + L-BFGS 1000 |
| 1D Burgers | `Nr=25600`, `Nbc=256` per side, `Nic=512`, LHS | residual/boundary/initial `1/1/1` | Adam 60000 |
| 1D Allen-Cahn | `Nr=25600`, `Nic=512`, LHS | `wr=1`, `wi=100` | Adam 300000 |
| 1D Diffusion-Reaction | `Nr=25600`, `Nbc=256` per side, `Nic=512`, LHS | `wr=1`, `wb=wi=100` | Adam 20000 + L-BFGS 1000 |
| 1D Advection-Diffusion | same time-PDE point counts, LHS | `wr=1`, `wb=wi=100` | Adam 20000 + L-BFGS 1000 |
| 1D discontinuous Advection | same time-PDE point counts, LHS | `wr=1`, `wb=wi=100` | Adam 20000 + L-BFGS 1000 |
| L-shaped Poisson | `Nr=10000`, `Nbc=128x8`, LHS | `wr=1`, `wb=100` | Adam 20000 + L-BFGS 1000 |

## Fourier inputs

- Allen-Cahn temporarily uses `[t, cos(k*pi*x), sin(k*pi*x)]`, `k=1,...,8`.
- Diffusion-Reaction uses `[t, cos(k*x), sin(k*x)]`, `k=1,...,8`.
- Advection-Diffusion uses the same order-8 convention confirmed for the current project.
- The backbone after these fixed input maps remains an ordinary FNN.

## Known qualifications

- The 1D Helmholtz implementation follows Section 3.1.2: `-u_xx + q*u = f` on `[0,1]`,
  `u=sin(20*pi*x)`, `q=400*pi^2`, and `f=800*pi^2*sin(20*pi*x)`. No interior observation
  is added.
- Allen-Cahn Fourier order 8 is temporary; the DASA PDF omits the order, while its cited
  reference [21] uses `m=10`.
- The L-shaped reference is a local finite-difference field rather than the paper's FEALPy
  adaptive FEM field.
- The exact Table 1 evaluation grids are not printed for several experiments.
- Burgers fixed losses and no-L-BFGS setting remain project conventions based on currently
  available information.
