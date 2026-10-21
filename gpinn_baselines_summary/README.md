# gPINN 汇总

本目录只保留修改后的 gPINN 代码、独立增强实验、测试和说明文档。主线实现为 Zhang 等论文中 13 个正向 PDE 算例的统一原生 PyTorch gPINN 基线，用于与 DASA-PINNs 公平对比；不包含历史训练结果、参考数据，也不包含 2022 仓库的 DeepXDE/TensorFlow 旧代码。

## 目录

```text
gpinn_baselines_summary/
├── gpinn_baselines/       # 13 个纯 gPINN 算例的统一主线
├── experiments/           # Allen–Cahn、Burgers、Helmholtz、Poisson 独立实验
├── tests/                 # 主线与独立实验测试
├── run_server_pipeline.sh # 13 算例多种子服务器入口
├── requirements.txt       # Python 依赖
└── IMPLEMENTATION_REPORT.md
```

当前目录不包含模型、参考数据、指标、日志、图片或统计表。以后实际运行训练时，程序会按需新建 `results/` 和 `data/`。

## 安装

Python 3.10 或更高版本：

```bash
cd gpinn_baselines_summary
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

网络对所有算例固定为 6 个隐藏层、每层 128 个神经元、`tanh`、Xavier-normal 初始化和 `float64`。稳态标量问题输出 `u`，Lid-driven cavity 输出 `(u,v,p)`。设备选择顺序为 CUDA → CPU；MPS 因不满足 float64 高阶自动微分要求会可靠回退到 CPU。

## 定义

对物理残差 $r_j$ 和独立变量 $z_k$，纯 gPINN 目标为

$$
L=w_r\operatorname{MSE}(r_j)+w_g\operatorname{MSE}(\partial_{z_k}r_j)+L_{bc}+L_{ic}+L_{periodic}+L_{gauge}.
$$

稳态问题对所有空间坐标增强；时空问题同时对 $x,t$ 增强；Navier–Stokes 对两个动量残差和连续性残差的 $x,y$ 导数全部增强。所有权重都位于 `gpinn_baselines/config.py`，也可用命令行覆盖。

相对 L2 和最大逐点误差分别为

$$
\frac{\lVert u_{pred}-u_{ref}\rVert_2}{\lVert u_{ref}\rVert_2},\qquad
\max_i |u_{pred,i}-u_{ref,i}|.
$$

Cavity 另外保存 `u`/`v` 分量指标、联合速度相对 L2 和最大逐点速度二范数误差。压力通过均值约束固定规范，不混入速度误差。

正式对比默认使用随机种子 `0-9`，即每个方程独立训练 10 次。每项指标报告算术均值和样本标准差

$$
\bar e=\frac{1}{10}\sum_{s=0}^{9}e_s,\qquad
s_e=\sqrt{\frac{1}{9}\sum_{s=0}^{9}(e_s-\bar e)^2}.
$$

因此汇总中的 `mean ± std` 使用 `ddof=1`，不是总体标准差。任何缺失、重复、未完成或包含 NaN/Inf 的种子结果都会使统计生成失败，不会用少于 10 次的结果冒充正式统计。

## 对应

| 算例 | 论文 | 实现类 | 默认点数 `Nr/Nbc/Nic` |
|---|---|---|---:|
| 1D Poisson boundary-layer | (10), (11a) | `Poisson1DBoundaryLayer` | 500/0/0（硬边界变换） |
| 1D Poisson steep | (10), (11b) | `Poisson1DSteep` | 500/0/0（硬边界变换） |
| 1D Helmholtz | (12) | `Helmholtz1D` | 500/2/0 |
| 2D Poisson peak | (13) 解 1 | `Poisson2DPeak` | 2500/50/0 |
| 2D Poisson steep | (13) 解 2 | `Poisson2DSteep` | 2500/50/0 |
| 2D Helmholtz | (14)–(16) | `Helmholtz2D` | 10201/200/0 |
| Lid-driven cavity, Re=100 | (17)–(18) | `LidDrivenCavity` | 2500/512/0 |
| 1D Burgers | (19) | `Burgers1D` | 1500/0/0（硬初边值变换） |
| 1D Allen–Cahn | (20) | `AllenCahn1D` | 25600/512/512 |
| 1D Diffusion–reaction | (21)–(23) | `DiffusionReaction1D` | 25600/256/512 |
| 1D Advection–diffusion | (24) | `AdvectionDiffusion1D` | 25600/256/512 |
| L-shaped Poisson | (25) | `PoissonLShape` | 10000/1024/0 |
| 1D Advection discontinuous | (26)–(28) | `AdvectionDiscontinuous1D` | 25600/256/512 |

PDE 定义在 `gpinn_baselines/cases/elliptic.py`、`time_dependent.py` 和 `flow.py`；公共网络、自动微分、采样、训练、指标、绘图、checkpoint 和参考解分别位于同名模块。`gpinn_baselines/` 是统一且唯一的 13 算例实现入口。

## 方程

- Poisson 与 Helmholtz 的解析解直接生成源项，验证器会将解析解代回自动微分残差。
- 两个 1D Poisson 保留论文的 500 个 LHS 域内点、20,000 次 Adam 和 0.01 梯度权重；按原始 gPINN Poisson 风格用输出变换严格满足两端 Dirichlet 值，并使用恒定 $10^{-3}$ 学习率。它们是纯 gPINN，不使用 RAR。
- 2D peak 的论文排版式缺少 $y^a$；图 3 的对称峰和原 `poisson_2d.py` 一致支持使用 $(16xy(1-x)(1-y))^a$。
- Burgers 严格使用 $0.01/\pi$ 黏性，并保留原始 gPINN 仓库的硬初边值输出变换、恒定 $10^{-3}$ Adam 学习率、20000 次 Adam 与后续 L-BFGS；只将网络统一为 6×128 float64。
- Allen–Cahn 严格使用 `1e-4`，同时包含初值、周期函数值和周期一阶空间导数损失；评估参考解由当前 PyTorch 项目的数值参考模块生成。
- Diffusion–reaction 使用 $D=1$ 和式 (22)的加和源项，与式 (23) 一致。
- Advection discontinuous 使用 $u_0(x-t)$ 作为精确参考。

## 歧义

论文正文、图注和附录未给出下列数值，且公开配套仓库无法访问。默认值均是可配置且保持制造解严格成立，不是隐式猜测：

| 参数 | 默认 | 依据 |
|---|---:|---|
| 1D Helmholtz $q(x)$ | 常数 1 | 可配置制造解默认 |
| 2D Helmholtz $(a_1,a_2,k)$ | $(1,4,1)$ | 标准多频制造解默认 |
| Advection–diffusion $\kappa$ | 0.01 | 小扩散、对流占优的明确默认 |
| Burgers 的 `Nbc/Nic` | 0/0 | 原仓库通过输出变换严格满足初值和边界，不采样软约束点 |
| Discontinuous advection 点数 | 25600/256/512 | 沿用论文时空算例数据规模 |

例如：

```bash
python -m gpinn_baselines.run --case advection_diffusion_1d --mode full --parameter diffusivity=0.005
```

## 参考解

```bash
python -m gpinn_baselines.references --all --root data/reference_data
python -m gpinn_baselines.validate --level references --reference-root data/reference_data
```

- Burgers：2049 个空间点、$\Delta t=2.5\times10^{-4}$ 的二阶 IMEX-BDF2 有限差分；与 1025 点、两倍时间步长比较。
- Allen–Cahn：1024 点周期 Fourier 伪谱 ETDRK4，$\Delta t=2.5\times10^{-4}$；与 512 点、两倍时间步长比较。
- L-shaped Poisson：401×401 精确 L 形掩码上的五点有限差分，CG 容差 `1e-11`；与 201×201 网格比较。
- Cavity：257×257 涡量–流函数法，半 Lagrangian 对流、隐式扩散和 DST Poisson 求解；以速度变化 `1e-8` 为稳态标准，与 129×129 网格比较。

正式参考 NPZ 不随本地开发结果预填。`run_server_pipeline.sh` 默认在服务器上使用 `--force` 重新生成四个数值参考解，并将该次服务器实际得到的网格加密差、迭代步数和收敛容差写入 `data/reference_data/<case>/reference.npz` 的 `metadata` 字段。

每个主参考文件的 `metadata` 字段保存方法、网格、时间步长、容差和网格加密差异。主参考网格高于 PINN 测试网格，比较前使用规则网格插值。

## 运行

单算例通用形式：

```bash
python -m gpinn_baselines.run --case allen_cahn_1d --mode full --seed 0
python -m gpinn_baselines.run --case allen_cahn_1d --mode test --seed 0
python -m gpinn_baselines.run --case allen_cahn_1d --mode full --seed 0 --resume
```

独立增强实验统一位于 `experiments/`：

```bash
python experiments/allen_cahn_1d_fixed/run.py --mode smoke --device cpu
python experiments/burgers_1d_fixed/run.py --mode smoke --device cpu
python experiments/helmholtz_1d_fixed.py --mode smoke --device cpu
bash experiments/poisson_1d_10seeds/run_10seeds.sh --gpus cpu --smoke-only
bash experiments/poisson_1d_gpinn_rar/run_smoke_then_full_10seeds.sh --gpus cpu --smoke-only
```

各实验的完整参数与十种子命令见对应目录中的 `README.md`。

13 个算例名称与命令：

```bash
python -m gpinn_baselines.run --case poisson_1d_boundary_layer --mode full --seed 0
python -m gpinn_baselines.run --case poisson_1d_steep --mode full --seed 0
python -m gpinn_baselines.run --case helmholtz_1d --mode full --seed 0
python -m gpinn_baselines.run --case poisson_2d_peak --mode full --seed 0
python -m gpinn_baselines.run --case poisson_2d_steep --mode full --seed 0
python -m gpinn_baselines.run --case helmholtz_2d --mode full --seed 0
python -m gpinn_baselines.run --case lid_driven_cavity --mode full --seed 0
python -m gpinn_baselines.run --case burgers_1d --mode full --seed 0
python -m gpinn_baselines.run --case allen_cahn_1d --mode full --seed 0
python -m gpinn_baselines.run --case diffusion_reaction_1d --mode full --seed 0
python -m gpinn_baselines.run --case advection_diffusion_1d --mode full --seed 0
python -m gpinn_baselines.run --case poisson_lshape --mode full --seed 0
python -m gpinn_baselines.run --case advection_discontinuous_1d --mode full --seed 0
```

手动顺序运行全部 13 个方程和 10 个种子：

```bash
python -m gpinn_baselines.run_all --mode full --seeds 0-9 --device auto --output-root results/gpinn_baselines_10seeds
```

推荐的服务器入口将 13×10=130 个任务分配到指定 GPU。全部 130 个 smoke 通过后才自动运行 130 个 full：

```bash
./run_server_pipeline.sh --gpus 0,1,2,3,4,5,6,7 --cpus 32 --seeds 0-9 --output-root results/gpinn_baselines_10seeds
./run_server_pipeline.sh --gpu cpu --cpus 32 --seeds 0-9 --output-root results/gpinn_baselines_10seeds
```

只复跑两个 1D Poisson 的独立入口位于 `experiments/poisson_1d_10seeds/`。它使用新的结果目录，不覆盖旧 Poisson 结果：

```bash
nohup ./experiments/poisson_1d_10seeds/run_10seeds.sh \
  --gpus 0,1,2,3,4,5,6,7 --cpus 32 --seeds 0-9 \
  > experiments/poisson_1d_10seeds/launcher.log 2>&1 &
```

`--seed N` 仍可用于单种子调试。正式统计应使用 `--seeds 0-9`。脚本会重用同一输出目录下已存在的 checkpoint，适合服务器中断后重启；为防止旧 seed 0 结果混入新协议，应使用新的 `results/gpinn_baselines_10seeds` 目录。`--smoke-only` 可只验收 smoke。

## 输出

```text
results/gpinn_baselines/
  smoke|full/
    <case>/seed_<seed>/
      config.json
      best.pt
      last.pt
      training_log.csv
      metrics.json
      prediction.npz
      reference.npz
      reference_master.npz   # 仅数值参考解算例
      loss_curve.png
      absolute_error.png
    results_summary.csv
    results_summary.md
    results_mean_std.csv    # 每项指标的数值 mean/std 和显示列
    results_mean_std.md
    results_mean_std.xlsx   # 单工作表、每个方程一行
```

`results_summary.*` 保存 130 行逐种子结果；`results_mean_std.*` 保存 13 行跨种子统计。XLSX 只有一个“十种子汇总”工作表，与原误差总表对应，并对 Cavity 的各速度指标分别统计。所有数据只来自实际 `metrics.json`，不包含论文数字或预填数字。Smoke 和 full 位于不同目录。

已有完整结果也可单独重新汇总：

```bash
python -m gpinn_baselines.summary --mode full --output-root results/gpinn_baselines_10seeds
python -m gpinn_baselines.aggregate --mode full --seeds 0-9 --output-root results/gpinn_baselines_10seeds
```

## 审计

```bash
python -m gpinn_baselines.validate --level static
python -m gpinn_baselines.validate --level references
python -m gpinn_baselines.validate --level results --mode smoke --seeds 0-9 --output-root results/gpinn_baselines_10seeds
python -m gpinn_baselines.validate --level results --mode full --seeds 0-9 --output-root results/gpinn_baselines_10seeds
```

审计包括：13 类导入；6×128 网络；解析解 PDE 残差；初/边/周期/速度条件；Allen–Cahn `1e-4`；参考解网格加密；指标从 NPZ 重算；结果 NaN/Inf 检查。

### GLIBCXX 错误

如果 Conda 环境报告系统 `libstdc++.so.6` 缺少 `GLIBCXX_3.4.29`，先在该环境安装 GCC 运行库：

```bash
conda install -y -c conda-forge "libstdcxx-ng>=12" "libgcc-ng>=12"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -c "import numpy, scipy, torch; print(numpy.__version__, scipy.__version__, torch.__version__)"
```

服务器脚本会自动将当前 `CONDA_PREFIX/lib` 放在动态库搜索路径前面，并在生成参考解或训练前检查 NumPy、SciPy 和 PyTorch 导入。

## 代码范围

Yu 等 2022 gPINN 论文仓库中的 DeepXDE/TensorFlow 脚本、逆问题脚本和旧数据已从当前工程移除。现有主线只保留本项目修改后的原生 PyTorch gPINN 实现；RAR 仅存在于名称明确的独立实验目录，不会混入 13 个纯 gPINN 基线。
