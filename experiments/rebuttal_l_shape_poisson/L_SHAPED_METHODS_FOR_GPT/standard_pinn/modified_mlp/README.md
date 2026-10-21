# Classical PINN Baselines for DASA-PINNs

本目录使用 **DeepXDE + PyTorch** 从零实现 DASA-PINNs 论文中各个 PDE 实验的
fixed-weight PINN baseline。这里的网络骨干明确使用 gated Modified MLP，
不是 DeepXDE 默认的 vanilla FNN。

论文路径：

```text
/data/ccy/260809_补实验_pinn/2. DASA_PINNs (Preprint submitted to Neurocomputing).pdf
```

实现遵循以下原则：

- PDE、计算区域、边界条件、初始条件和解析解优先依据论文。
- classical PINN 尽可能复用对应 DASA 实验的 backbone 和训练参数。
- 不使用 DASA sub-network、pointwise adaptive loss weighting 或 adversarial training。
- 不复制或迁移项目中 DeepXDE examples 的具体 PINN 源代码。
- 所有 baseline 使用 DeepXDE 的 PyTorch backend 和 `float64`。
- 默认 backbone 为自定义 gated modified MLP，而不是退化为普通 FNN。
- 每个正式实验分别保存预测场和指标元数据。

## 目录结构

```text
classical_pinn_baselines/
├── pinn_baselines/
│   ├── common_modified_mlp_corrected.py
│   ├── poisson_1d.py
│   ├── helmholtz_1d.py
│   ├── poisson_2d.py
│   ├── helmholtz_2d.py
│   ├── lid_driven_cavity.py
│   ├── burgers_1d.py
│   ├── allen_cahn_1d.py
│   ├── diffusion_reaction_1d.py
│   ├── advection_diffusion_1d.py
│   ├── advection_discontinuous_1d.py
│   └── poisson_l_shaped.py
├── reference_data/
├── runs/
├── generate_advection_diffusion_reference.py
├── generate_allen_cahn_reference.py
├── generate_l_shaped_poisson_reference.py
├── run_*.sh
├── PAPER_SETUP_QUESTIONS.md
└── 260810_Chaoye_未确定的实验设置.md
```

所有方程脚本统一使用 `pinn_baselines/common_modified_mlp_corrected.py`，其提供：

- 按引用 [21] 公式修正的 gated `ModifiedMLP`；
- Adam 和 L-BFGS 训练封装；
- 随机种子设置；
- 相对 L2 和最大逐点误差；
- `metrics.json`、`predictions.npz` 的统一保存逻辑。

## 环境要求

代码使用父目录内的 vendored DeepXDE：

```text
/data/ccy/260809_补实验_pinn/deepxde
```

建议环境包含：

```text
Python 3.10+
PyTorch with CUDA
NumPy
SciPy
mpmath
```

启动脚本会自动设置：

```bash
export DDE_BACKEND=pytorch
export PYTHONPATH=/data/ccy/260809_补实验_pinn/classical_pinn_baselines:$PYTHONPATH
```

进入项目目录：

```bash
cd /data/ccy/260809_补实验_pinn/classical_pinn_baselines
```

## 统一网络与训练约定

除非对应论文实验另有设置，当前 baseline 使用：

```text
backbone       = gated modified MLP (`modified_mlp_corrected`)
depth          = 6
width          = 128
activation     = tanh
initializer    = Xavier/Glorot normal
precision      = float64
initial_lr     = 0.001
lr_decay       = 0.9 every 2000 Adam iterations
sampling       = LHS or the paper-specified fixed mesh
```

时间 PDE 当前统一采用：

```text
Nr  = 25600
Nbc = 256 per side
Nic = 512
```

显式使用左右边界损失时，左右两侧共享同一组 LHS 时间坐标。

## 实验列表

| Experiment | Main setup | Default optimization | Launcher |
|---|---|---|---|
| 1D Poisson boundary-layer / steep | `Nr=500` | Adam 40000 | `run_poisson_1d.sh` |
| 1D Helmholtz | `Nr=500`, `q=-(20π)^2`, `f=0` | Adam 20000 | `run_helmholtz_1d.sh` |
| 2D Poisson peak / steep | `Nr=2500`, `Nbc=50` | Adam 20000 + L-BFGS 1000 | `run_poisson_2d.sh` |
| 2D Helmholtz | `101×101` residual mesh, 50 points/edge | Adam 20000 + L-BFGS 1000 | `run_helmholtz_2d.sh` |
| Lid-driven cavity Re=100 | `50×50` residual mesh, 128 points/edge | Adam 20000 + L-BFGS 1000 | `run_lid_driven_cavity.sh` |
| 1D Burgers | `Nr=25600`, `Nbc=256/side`, `Nic=512` | Adam 60000 | `run_burgers_1d.sh` |
| 1D Allen-Cahn | `Nr=25600`, `Nic=512`, periodic Fourier order 8 | Adam 300000 | `run_allen_cahn_1d.sh` |
| 1D Diffusion-Reaction | `Nr=25600`, `Nbc=256/side`, `Nic=512` | Adam 20000 + L-BFGS 1000 | `run_diffusion_reaction_1d.sh` |
| 1D Advection-Diffusion | `v=1`, `κ=0.01`, Fourier order 8 | Adam 20000 + L-BFGS 1000 | `run_advection_diffusion_1d.sh` |
| 1D discontinuous Advection | `v=1`, `κ=0` | Adam 20000 + L-BFGS 1000 | `run_advection_discontinuous_1d.sh` |
| L-shaped Poisson | `Nr=10000`, `128×8` boundary points | Adam 20000 + L-BFGS 1000 | `run_poisson_l_shaped.sh` |

表中的值是当前脚本默认值。论文明确项、作者补充确认项和当前推断项的详细区分见：

- `PAPER_SETUP_QUESTIONS.md`
- `260810_Chaoye_未确定的实验设置.md`

## 快速运行

### 1D Poisson

不带参数时同时运行 boundary-layer 和 steep：

```bash
./run_poisson_1d.sh
```

只运行一个 case：

```bash
./run_poisson_1d.sh boundary-layer
./run_poisson_1d.sh steep
```

将 40000 步结果保存到单独目录：

```bash
ADAM_ITERATIONS=40000 \
OUTPUT_DIR=runs/multiseed/poisson_1d_40000step \
./run_poisson_1d.sh all
```

### 2D Poisson

不带参数时同时运行 peak 和 steep：

```bash
./run_poisson_2d.sh
```

只运行一个 case：

```bash
./run_poisson_2d.sh peak
./run_poisson_2d.sh steep
```

### 其余实验

```bash
./run_helmholtz_1d.sh
./run_helmholtz_2d.sh
./run_lid_driven_cavity.sh
./run_burgers_1d.sh
./run_allen_cahn_1d.sh
./run_diffusion_reaction_1d.sh
./run_advection_diffusion_1d.sh
./run_advection_discontinuous_1d.sh
./run_poisson_l_shaped.sh
```

启动脚本默认 seed 为：

```text
2027 2028 2029 2030 2031 2032 2033
```

可以临时覆盖，不需要修改文件：

```bash
SEEDS="2040 2041 2042" ./run_burgers_1d.sh
```

## Allen-Cahn 四 GPU 并行

`run_allen_cahn_1d.sh` 使用“一个 seed 独占一张 GPU”的 worker 调度，最多同时使用四张 GPU。默认设备为：

```text
CUDA_DEVICES="0 1 2 3"
```

直接运行：

```bash
./run_allen_cahn_1d.sh
```

指定其他 GPU：

```bash
CUDA_DEVICES="4 5 6 7" ./run_allen_cahn_1d.sh
```

同一 GPU 上分配到的多个 seed 顺序执行，不同 GPU 的 worker 并行执行。该脚本不是单个 seed 的多 GPU 数据并行，因此不会改变模型训练定义。

## 覆盖参数

大多数启动脚本支持环境变量：

```text
CUDA_DEVICE
SEEDS
ADAM_ITERATIONS
LBFGS_ITERATIONS
DISPLAY_EVERY
OUTPUT_DIR
REFERENCE_PATH
```

示例：

```bash
CUDA_DEVICE=1 \
SEEDS="2024 2025 2026" \
ADAM_ITERATIONS=20000 \
LBFGS_ITERATIONS=1000 \
OUTPUT_DIR=runs/recheck/helmholtz_2d \
./run_helmholtz_2d.sh
```

`run_poisson_1d.sh` 和 `run_helmholtz_1d.sh` 当前直接在脚本内指定 GPU；如果要改变设备，需要修改对应脚本。Allen-Cahn 使用复数形式的 `CUDA_DEVICES`。

每个 Python 模块也可以直接运行，例如：

```bash
DDE_BACKEND=pytorch PYTHONPATH=. \
python -m pinn_baselines.burgers_1d \
    --seed 2024 \
    --adam-iterations 60000 \
    --output-dir runs/manual/seed_2024
```

使用以下命令查看单个模块的全部参数：

```bash
python -m pinn_baselines.burgers_1d --help
```

## 指标定义

所有标量解实验统一计算：

### Relative L2 error

```text
||y_true - y_pred||_2 / ||y_true||_2
```

对应代码：

```python
def l2_relative_error(y_true, y_pred):
    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)
```

### Max pointwise error

```text
max(abs(y_true - y_pred))
```

对应代码：

```python
def max_pointwise_error(y_true, y_pred):
    return np.max(np.abs(y_true - y_pred))
```

不同实验还会保存 PDE residual RMS/max、边界误差、初始条件误差或周期跳跃误差等诊断指标。

### Lid-driven cavity 指标

Lid-driven cavity 的主指标作用于标量速度模长：

```text
speed = sqrt(u^2 + v^2)
```

主相对 L2 为：

```text
||speed_true - speed_pred||_2 / ||speed_true||_2
```

同时保存 `u/v` 分量误差，以及原参考代码使用的速度矩阵 2-范数指标。

DeepXDE 训练日志中的 `Test metric []` 是正常现象：本项目在训练结束后自行计算上述指标，并写入 `metrics.json`，没有把它们注册到 DeepXDE 的实时 metric 列表。

## 结果保存

启动脚本通常采用以下结构：

```text
runs/multiseed_modified_mlp_corrected/<experiment>/seed_<seed>/<equation>/<case>/
├── metrics.json
└── predictions.npz
```

例如：

```text
runs/multiseed_modified_mlp_corrected/poisson_2d/seed_2027/poisson_2d/peak/metrics.json
runs/multiseed_modified_mlp_corrected/poisson_2d/seed_2027/poisson_2d/peak/predictions.npz
```

`metrics.json` 包含：

- `metrics`：L2、最大逐点误差和物理诊断指标；
- `metadata`：方程、区域、点数、loss 权重、网络、优化器、参考场和 seed；
- `created_at`：UTC 创建时间。

`predictions.npz` 通常包含：

```text
x
y_true
y_pred
reference_x / reference_t（适用时）
```

### 防止覆盖

不同 seed 写入不同的 `seed_<seed>` 目录，不会互相覆盖。以下条件全部相同时会覆盖旧的 `metrics.json` 和 `predictions.npz`：

- `OUTPUT_DIR` 相同；
- seed 相同；
- equation/case 相同。

训练步数和 loss 权重通常没有全部编码进目录名。因此使用不同超参数重跑同一个 seed 时，建议更换 `OUTPUT_DIR`：

```bash
SEEDS="2024 2025 2026" \
OUTPUT_DIR=runs/recheck/burgers_60000 \
./run_burgers_1d.sh
```

## 参考数据

| File | Experiment | Source and caveat |
|---|---|---|
| `reference_data/Burgers.npz` | 1D Burgers | 与方程和初边值一致的数值参考场；论文未明确指出使用同一文件 |
| `reference_data/Allen_Cahn.mat` | 1D Allen-Cahn legacy | DeepXDE 附带场对应 `d=0.001`，与论文 `d=0.0001` 不一致，不可用于当前指标 |
| `reference_data/allen_cahn_diffusivity_0.0001_spectral_reference.npz` | 1D Allen-Cahn | 本地周期 Fourier 伪谱 + SciPy DOP853 生成，与论文扩散系数一致 |
| `reference_data/lid_driven_cavity_re100_reference.npz` | Cavity Re=100 | 从项目 CSV 转换；尚未独立证明就是论文 [19] 的原始场 |
| `reference_data/advection_diffusion_kappa_0.01_series_reference.npz` | Advection-Diffusion | 1000 项解析级数，80 位精度求和后转为 float64 |
| `reference_data/l_shaped_poisson_fdm_reference.npz` | L-shaped Poisson | 本地五点有限差分场；不是论文 FEALPy 自适应 FEM 原场 |

### 重新生成 Advection-Diffusion 参考场

```bash
python generate_advection_diffusion_reference.py
```

默认设置：

```text
velocity          = 1
kappa             = 0.01
series terms      = 1000
decimal precision = 80
grid              = 256 × 101
```

### 重新生成 Allen-Cahn 参考场

```bash
python generate_allen_cahn_reference.py
```

默认使用 `d=0.0001`、6400 个周期 Fourier 内部网格点和 `201×101` 输出网格。时间积分步长上限根据最大扩散特征值自动设置。DeepXDE 附带的 `Allen_Cahn.mat` 使用 `d=0.001`，不能替代该参考场。

### 重新生成 L-shaped Poisson 参考场

```bash
python generate_l_shaped_poisson_reference.py
```

默认采用 `513×513` 求解网格，即 `h=1/256`，并下采样到间距 `1/128` 的评估点。对应启动脚本在参考文件缺失时也会自动生成。

## 重要实验说明

### 1D Helmholtz 非唯一性

当前确认的方程为：

```text
-u_xx - (20*pi)^2*u = 0
u(0) = u(1) = 0
```

只使用 PDE 和两端零边界时，`0` 和任意 `A*sin(20*pi*x)` 都是解，无法确定振幅。当前脚本额外加入：

```text
u(1/40) = 1
```

该条件使用权重 100 的软 PointSetBC。它不是 PDF 原文打印的条件。现有诊断表明权重 100 仍不足以克服高频 Helmholtz 的优化尺度问题，模型可能继续停留在近零解。因此，在内部观测误差接近 1 时，不应把该结果视为有效收敛结果。

### Allen-Cahn 训练预算

当前确认的 classical PINN 训练预算为 Adam 300000 步，无 L-BFGS。历史 `seed_2024/2025/2026` 结果只训练了 40000 步，并且使用了扩散系数错配的 DeepXDE `Allen_Cahn.mat`，不能作为当前论文方程的有效指标。当前启动脚本默认使用 300000 步和新生成的 `d=0.0001` 参考场。

### L-shaped Poisson 参考口径

当前 relative L2 和 max pointwise error 基于本地有限差分参考场。论文 Table 1 使用 FEALPy 自适应 FEM，因此二者可以比较数量级，但不保证完全同口径。

### 测试网格

论文没有为所有 Table 1 指标明确给出测试网格。当前脚本使用各自记录在 `metrics.json` 中的均匀网格或参考数据原始网格。指标在当前参考场上有效，但与论文数值完全复现仍需要相同参考场和相同测试点。

## 选择多 seed 最佳结果

当前填表约定是：

1. 在同一实验设置的多个 seed 中选择 `l2_relative_error` 最小者。
2. `max_pointwise_error` 必须取自同一个 seed，不能再从其他 seed 单独选择最小值。
3. 比较前检查 `metadata.optimizer`、参考场、case 和训练点设置是否一致。

不要混合以下结果：

- 不同训练步数；
- 不同 loss 权重；
- 不同参考场；
- Helmholtz 有/无内部观测；
- 历史诊断目录与正式 multiseed 目录。

## 设置记录

遇到论文缺失参数或正文/图示不一致时，先查阅：

```text
PAPER_SETUP_QUESTIONS.md
260810_Chaoye_未确定的实验设置.md
```

新增确认信息时，应同时更新：

1. 对应 Python 脚本中的常量或默认参数；
2. 对应 `.sh` 的默认训练设置；
3. 两份实验设置文档；
4. 未来运行生成的 `metrics.json` metadata；
5. 必要时使用新的 case 名或 `OUTPUT_DIR`，避免覆盖旧结果。
