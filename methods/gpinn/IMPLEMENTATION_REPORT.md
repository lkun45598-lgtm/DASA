# 实现报告

## 仓库审查

原目录不是 Git 工作树，因此无法使用 commit 基线比对；修改前已逐个读取文件。原正向脚本使用 DeepXDE，并在多数文件中显式导入 TensorFlow 后端。它们的问题包括：

- 网络为 3×20、3×32 或 4×64，不符合 6×128。
- `burgers.py` 和 `allen_cahn.py` 默认使用 RAR，不是纯 gPINN。
- `allen_cahn.py` 和 `usol_D_0.001_k_5.mat` 使用 0.001，与式 (20a) 的 0.0001 不一致。
- `Burgers.npz` 的 $x\in[-1,1]$、初值和边界值匹配式 (19)，但时间只覆盖 0 到 0.99，没有完整覆盖论文的 $[0,1]$。
- `poisson_2d.py` 提供了对称 peak 解的意图，可用于解决论文式 (13) 后第一个解排版丢失 $y^a$ 的问题。

原始 `Burgers.npz`、`usol_D_0.001_k_5.mat`、DeepXDE 脚本和无关逆问题文件已清理。当前只保留原生 PyTorch 实现。

## 算例对应

| 序号 | 算例 | 论文页/公式 | 代码 |
|---:|---|---|---|
| 1 | 1D Poisson boundary-layer | 第 17 页，(10), (11a) | `cases/elliptic.py::Poisson1DBoundaryLayer` |
| 2 | 1D Poisson steep | 第 17 页，(10), (11b) | `cases/elliptic.py::Poisson1DSteep` |
| 3 | 1D Helmholtz | 第 18 页，(12) | `cases/elliptic.py::Helmholtz1D` |
| 4 | 2D Poisson peak | 第 18–20 页，(13) 解 1 | `cases/elliptic.py::Poisson2DPeak` |
| 5 | 2D Poisson steep | 第 18–20 页，(13) 解 2 | `cases/elliptic.py::Poisson2DSteep` |
| 6 | 2D Helmholtz | 第 21 页，(14)–(16) | `cases/elliptic.py::Helmholtz2D` |
| 7 | Lid-driven cavity | 第 22–23 页，(17)–(18) | `cases/flow.py::LidDrivenCavity` |
| 8 | 1D Burgers | 第 24 页，(19) | `cases/time_dependent.py::Burgers1D` |
| 9 | 1D Allen–Cahn | 第 24–26 页，(20) | `cases/time_dependent.py::AllenCahn1D` |
| 10 | 1D Diffusion–reaction | 第 26–27 页，(21)–(23) | `cases/time_dependent.py::DiffusionReaction1D` |
| 11 | 1D Advection–diffusion | 第 28 页，(24) | `cases/time_dependent.py::AdvectionDiffusion1D` |
| 12 | L-shaped Poisson | 第 29 页，(25) | `cases/elliptic.py::PoissonLShape` |
| 13 | 1D Advection discontinuous | 第 30 页，(26)–(28) | `cases/time_dependent.py::AdvectionDiscontinuous1D` |

## 方程要点

1. 两个 1D Poisson 对 $-u_{xx}=f$ 及 $\partial_xr$ 建模；修正后以输出变换严格满足两端解析 Dirichlet 值，不再使用会与大尺度梯度残差竞争的软边界损失。训练仍为纯 gPINN，不含 RAR。
2. 1D Helmholtz 对 $-u_{xx}+q u=f$ 及 $\partial_xr$ 建模；$q=1$ 是显式可配置默认。
3. 两个 2D Poisson 对 $-\Delta u=f$、$\partial_xr$、$\partial_yr$ 建模，源项由解析解自动微分生成。
4. 2D Helmholtz 对 $\Delta u+k^2u-q=0$ 建模；$(a_1,a_2,k)=(1,4,1)$ 是显式可配置默认。
5. Cavity 同时建模两个动量方程和连续性，并对三个残差的 $x,y$ 导数增强；压力均值约束固定规范。
6. Burgers 使用 $\nu=0.01/\pi$，对残差的 $x,t$ 导数分别增强并求和；保留原仓库的1500个域内点、硬初边值输出变换、恒定学习率和 Adam→L-BFGS 训练顺序，仅将网络改为统一的6×128 float64。
7. Allen–Cahn 只使用 $10^{-4}$，包含初值、周期值和周期 $u_x$ 损失，对残差的 $x,t$ 导数增强。
8. Diffusion–reaction 的式 (22) 按加和解释，与式 (23) 代回一致。
9. Advection–diffusion 使用 $v=1,\kappa=0.01$ 可配置默认，指标参考为稳定计算的 1000 项像级数解。
10. L-shaped Poisson 使用精确 L 形采样掩码和 8 段边界。
11. 间断对流使用 $u_0(x-t)$ 参考；$\kappa=0$ 时不计算无意义的扩散项。

## 参考解

| 算例 | 方法 | 服务器细/粗分辨率 |
|---|---|---|
| Burgers | 有限差分 IMEX-BDF2 | 2049/1025 点，dt 0.00025/0.0005 |
| Allen–Cahn | Fourier 伪谱 ETDRK4 | 1024/512 点，dt 0.00025/0.0005 |
| L-shaped Poisson | 五点有限差分+CG | 401×401 / 201×201 |
| Cavity | 涡量–流函数 | 257×257 / 129×129 |

正式数值参考解由服务器流水线强制重新生成。实际网格差、Cavity 迭代步数和收敛判据以服务器生成的 `data/reference_data/<case>/reference.npz` 内 `metadata` 为准。

## 文件

新建：

- `gpinn_baselines/`：网络、自动微分、配置、采样、13 个问题类、训练、checkpoint、指标、绘图、参考解、验证、单算例与全算例入口。
- `run_server_pipeline.sh`：可配置 GPU/CPU、默认 10 个种子的 smoke → full 自动流水线。
- `experiments/poisson_1d_10seeds/`：两个 1D Poisson 的独立十种子 smoke → full 流水线、诊断和两行 XLSX 汇总说明。
- `gpinn_baselines/aggregate.py`：对全部 13 个方程执行跨种子均值和样本标准差统计，输出 CSV、Markdown 和单表 XLSX。
- `tests/test_baselines.py`：非训练单元测试。
- `gpinn_baselines/references.py`：服务器生成四个数值参考解及网格加密数据。
- `IMPLEMENTATION_REPORT.md`：本报告。

修改：

- `README.md`、`requirements.txt`。
- 2022 原仓库的 `src/`、根目录旧模块副本和 DeepXDE 单文件遗留已移除，统一以 `gpinn_baselines/` 为主线。

## 已验证

- Python 字节码编译：通过。
- Bash 脚本语法：通过。
- 单元测试：4/4 通过。
- 13 个算例导入和网络 6×128/float64 审计：通过。
- 解析解代回：自动微分残差最大值为 0 到 `4.55e-13`；Advection–diffusion 级数的差分代回残差为 `1.13e-7`。
- 初值、Dirichlet、Allen–Cahn 周期配对和 Cavity 速度边界：通过。
- Allen–Cahn 配置和残差系数 `1e-4`、旧 0.001 数据不被引用：通过。
- 数值参考解生成器的收敛与非有限值审计逻辑：已实现；正式数值由服务器流水线重新得到。

## 待服务器运行

根据用户明确指令，本机没有运行任何 PINN smoke 或 full 训练，因此没有伪造或预填 13 个训练误差。服务器命令为：

```bash
./run_server_pipeline.sh --gpus 0,1,2,3,4,5,6,7 --cpus 32 --seeds 0-9 --output-root results/gpinn_baselines_10seeds
```

该脚本会调度 13×10=130 个任务。只有全部 130 个 smoke 均通过，且每个种子的保存指标可从 NPZ 重算、结果无 NaN/Inf 时，才自动进入 full。完成后生成 130 行逐种子明细，以及 13 行 `mean ± std` 汇总；标准差采用样本标准差（`ddof=1`）。正式单表位于 `results/gpinn_baselines_10seeds/full/results_mean_std.xlsx`。

## 未解歧义

- 1D Helmholtz $q(x)$、2D Helmholtz $a_1,a_2,k$、Advection–diffusion $\kappa$ 在论文中均未唯一给出。
- Burgers 不设置边界/初值采样点，因为其初值和边界由原仓库输出变换严格满足。

所有默认和依据已写入 `config.py`、`config.json` 输出及 README，可使用 `--parameter` 或点数命令行参数覆盖。
