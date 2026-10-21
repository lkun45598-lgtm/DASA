# 1D Allen-Cahn gPINN + RAR

该目录实现一维 Allen-Cahn 方程的 gPINN（gradient-enhanced PINN）+ RAR（残差自适应加点）：

$$
u_t-10^{-4}u_{xx}+5u^3-5u=0,\qquad (x,t)\in[-1,1]\times[0,1],
$$

初值 $u(x,0)=x^2\cos(\pi x)$，周期边界 $u(-1,t)=u(1,t)$、$u_x(-1,t)=u_x(1,t)$（软损失，非硬变换）。

## 这个目录原来是什么

`run.py` 是独立可运行实现，做法参照本仓库 `experiments/burgers_1d_fixed/` 已经验证过的模式：保留 `gpinn_baselines` 中经过论文核对的纯 gPINN 损失结构，补回原论文的 RAR。依赖缺失外部框架、无法独立运行的参考文件已清理。

## 依据

- Yu, Lu, Meng, Karniadakis, *Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems*, CMAME 393 (2022) 114823，arXiv:2111.02801。
- 论文官方代码 `src/allen_cahn.py`（https://github.com/lu-group/gpinn ，DeepXDE/TensorFlow-1）。RAR 循环忠实复刻：每轮从 100,000 个随机候选点中取残差最大的 30 个加入训练集，如此循环。该文件的物理设定（$D=0.001$、硬输出变换 `t*(1-x^2)*y+x^2*cos(pi*x)`、空边界条件列表）**没有**照搬，原因见下。
- 本仓库 `gpinn_baselines/cases/time_dependent.py::AllenCahn1D`：已核对到论文式 (20) 的纯 gPINN 基线（$D=10^{-4}$，软初值 + 软周期值 + 软周期一阶导数损失，普通 MLP，无输出变换，对 $x,t$ 都做梯度增强）。本脚本只在其之上补回连续重采样和 RAR，其余损失结构原样保留。

## 与原始 `src/allen_cahn.py` 的差异（均为本仓库既有立场，非本次新决定）

1. **扩散系数用 $10^{-4}$，不是 $0.001$。** 论文式 (20a) 明确是 $10^{-4}$；原脚本及其配套的 `usol_D_0.001_k_5.mat` 用的是 $0.001$，这一点在 `IMPLEMENTATION_REPORT.md` 中已被指出并修正，本脚本延续该修正。
2. **不用硬输出变换。** 原脚本的 `t*(1-x^2)*y+x^2*cos(pi*x)` 会让 $u(-1,t)=u(1,t)=\cos(\pi)=-1$ 对任意 $t$ 恒成立（因为 $1-x^2$ 在两端恒为零），且完全没有约束 $u_x$——这不是周期边界条件真正的含义，只是一个凑巧简便的写法。本脚本保留 `AllenCahn1D` 已经在用的软周期值损失和软周期一阶导数损失，网络本身不做任何硬变换。
3. **训练点每步重新采样。** 原始 `gpinn_baselines` 用一次性采样的 25600 个固定点做子采样；本脚本每个 Adam 步都用新的 LHS 点，与 `burgers_1d_fixed` 修复 1500 个固定点被网络记住的方式一致。
4. **评估参考解独立生成。** 用周期 Fourier 伪谱 ETDRK4（$D=10^{-4}$，与 `gpinn_baselines/references.py::generate_allen_cahn` 相同格式）现算，不使用扩散系数不匹配的 `usol_D_0.001_k_5.mat`。
5. **RAR 轮数/步数改为固定、确定性的步数预算**，而不是原始的 `EarlyStopping(patience=2000)`（长度不确定，难以在多种子服务器流水线上复现/续跑）。默认 30 轮、每轮 2000 步、每轮加 30 个点（加点数量和候选池大小均取自原论文脚本），热身 20000 步，最后 1000 步 L-BFGS。

## 安装

使用仓库根目录的环境即可（`torch`、`numpy`、`scipy`、`matplotlib`）：

```bash
cd "/Users/0v0/Desktop/gpinn-main/gpinn汇总"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 单次运行

在项目根目录执行：

```bash
python experiments/allen_cahn_1d_fixed/run.py --device cuda
python experiments/allen_cahn_1d_fixed/run.py --device cpu
python experiments/allen_cahn_1d_fixed/run.py --device cuda --resume
python experiments/allen_cahn_1d_fixed/run.py --mode smoke --device cpu
```

结果保存在 `results/allen_cahn_1d_fixed/seed_<seed>/`，包括 `config.json`、`best.pt`/`last.pt`、`training_log.csv`、`rar_history.jsonl`、`rar_anchors.npz`、`metrics.json`、`prediction.npz`、`solution_and_error.png`、`training_curve.png`。

## 十次运行

```bash
python experiments/allen_cahn_1d_fixed/run_10seeds.py --gpus auto --seeds 0-9
bash experiments/allen_cahn_1d_fixed/start_10seeds_nohup.sh --gpus 0,1,2,3,4,5,6,7
```

检测到多张 CUDA GPU 时，每张卡同时跑一个种子；只有 CPU 时用 `--cpu-workers` 控制并行数。已完成的种子会自动跳过，存在检查点但未完成的会自动续跑（`--force` 强制重跑）。完成后在输出目录生成 `summary_10seeds.json/.csv/.md`（均值 ± 样本标准差，`ddof=1`）。

## 核验

`--mode smoke --device cpu` 用极小的网络（2×24）、极短的热身/RAR/候选池跑通整条流水线（含参考解生成、RAR 加点、L-BFGS、评估、画图），用于在提交完整运行前捕获形状、自动微分或数值错误，不代表收敛精度。
