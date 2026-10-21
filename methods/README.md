# Methods × 方程

"✓" 表示该方法目录下有对应方程的代码/配置。

| 方程 | sa_pinn | sa_pinn_aligned | dasa | plain_pinn | deepxde_pinn | gpinn | hp_vpinn |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1D Poisson boundary-layer | | ✓ | | ✓ `poisson_1d` | ✓ | ✓ | ✓ |
| 1D Poisson steep | | ✓ | | ✓ `poisson_1d` | ✓ | ✓ | |
| 1D Helmholtz | ✓ `helmholtz` | ✓ | | | ✓ | ✓ | |
| 2D Poisson peak / steep | | ✓ | | | ✓ | ✓ | |
| 2D Helmholtz | | ✓ | | | ✓ | ✓ | |
| Lid-driven cavity | | ✓ | | | ✓ | ✓ | |
| 1D Burgers | ✓ `burgers` | ✓ | | | ✓ | ✓ | |
| 1D Allen–Cahn | ✓ `allen_cahn` | ✓ | | ✓ `allen_cahn_1d` | ✓ | ✓ | |
| 1D Diffusion–reaction | | ✓ | | | ✓ | ✓ | |
| 1D Advection–diffusion | | ✓ | | | ✓ | ✓ | |
| 1D Advection discontinuous | | ✓ | | | ✓ | ✓ | |
| L-shaped Poisson | | ✓ | ✓ `l_shape_poisson` | | ✓ | ✓ | |

各方法的 13 个算例定义见:
`sa_pinn_aligned/configs/cases/*.yaml`、`deepxde_pinn/problems/`、`gpinn/gpinn_baselines/cases/`。
波动方程实验在 `experiments/rebuttal_wave_equation/`。

说明:`dasa/` 目前只有 L 形 Poisson 的 DASA 实现(原始 notebook 与可复现 runner);
`sa_pinn_aligned` 是**不含** DASA 权重预测子网络的直接点掩码 SA-PINN。
