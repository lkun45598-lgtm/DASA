# 1D Poisson gPINN / gPINN+RAR 十种子实验

本目录独立运行目标论文式 (10)-(11) 的两个 1D Poisson 算例：

- `boundary_layer`；
- `steep`。

网络固定为原生 PyTorch `float64`、6 个隐藏层、每层 128 个神经元、`tanh`、Xavier/Glorot uniform。Dirichlet 条件通过上游 Poisson 风格的输出变换严格满足。损失为

```text
MSE(r) + 0.01 MSE(r_x),    r = -u_xx - f.
```

不包含 DASA、SA-PINN、可训练点权重、Fourier 特征或损失归一化。

## 核对结论

不能把“原 Poisson 配置”和“原 RAR 配置”称作同一个上游配置：

1. 目标论文对式 (10)-(11) 规定 500 个 LHS 内点和 6×128 网络，但没有给出 Poisson gPINN+RAR 流程；
2. 官方 `lu-group/gpinn` 的 `src/poisson_1d.py` 求解另一个 `[0,π]` 平滑问题，使用 15 个均匀内点、Adam 20,000、恒定学习率 `1e-3`、gPINN 权重 `0.01`、Glorot uniform、无 L-BFGS、无 RAR；
3. 官方仓库只在 `burgers.py` 和 `allen_cahn.py` 中提供 gPINN+RAR。`burgers.py` 的流程是初始 Adam 20,000 → L-BFGS-B，然后 40 轮：100,000 随机候选点 → 按原始 `|r|` 选 Top-10 → Adam 最多 10,000（EarlyStopping）→ L-BFGS-B；
4. 上游依赖没有锁定 DeepXDE 版本。按 2022-02-28 的 DeepXDE 源码，L-BFGS 默认值为 maxiter=15,000、history=100、gtol=1e-8、ftol=0。PyTorch 版本使用对应的 `torch.optim.LBFGS` 参数。

核对的官方版本：

- 仓库：`https://github.com/lu-group/gpinn`；
- commit：`09dd5903c326389f041341ed8cfe4984b24278cd`；
- `src/poisson_1d.py` SHA-256：`23ea36217c22f2f52c277dffc3cfd3ec5fe6be76f19b732217a80b6647fb419a`；
- `src/burgers.py` SHA-256：`657d27db73a033e0ff5e3244ab53cee8ba93d6ecbd5c2cb4995600aba725768e`。

## 三种协议

程序禁止把不同来源的参数静默混合，必须通过 `--protocol` 选择：

### `burgers_rar_transfer`（默认）

这是用户要求的 gPINN with RAR：

- 方程、区间、解析边界、500 个 LHS 点来自目标论文；
- gPINN 权重、Adam 步数、学习率和初始化来自上游 Poisson；
- RAR 流程逐项移植自上游 Burgers；
- 只把网络改为 6×128，并改用原生 PyTorch `float64`。

该协议是“官方 Burgers-RAR 流程在目标 Poisson 上的显式移植”，不是上游已有的 Poisson-RAR 配置。完整流程非常重：每个种子初始一次 L-BFGS，随后最多 40 次 Adam+L-BFGS。

### `paper_gpinn`

- 目标论文的 500 个 LHS 点；
- Adam 20,000、学习率 `1e-3`、gPINN 权重 `0.01`；
- 无 L-BFGS、无 RAR。

这是适合论文公平对比的纯 gPINN 基线。

### `upstream_poisson`

- 上游 Poisson 的 15 个固定均匀内点；
- Adam 20,000、学习率 `1e-3`、gPINN 权重 `0.01`；
- 无 L-BFGS、无 RAR；
- 方程仍使用目标论文式 (10)-(11)，网络仍为 6×128。

它用于审计“除方程和网络外尽量贴近上游 Poisson”的效果，不满足目标论文的 500 点数据规模。

## 上传

先在服务器停止旧版任务：

```bash
pkill -TERM -f '/data2/user/cjm/poisson_1d_gpinn_rar'
pgrep -af 'poisson_1d_gpinn_rar' || true
```

再在本地 Mac 上传。该命令保留服务器旧结果：

```bash
rsync -avh --progress \
  --exclude 'results*/' \
  --exclude '__pycache__/' \
  experiments/poisson_1d_gpinn_rar/ \
  cjm@sus-server:~/experiments/poisson_1d_gpinn_rar/
```

## 服务器启动

默认运行 `burgers_rar_transfer`。启动脚本内部已有 `nohup`，命令前不要再次添加 `nohup`：

```bash
cd ~
conda activate dl

./experiments/poisson_1d_gpinn_rar/start_10seeds_nohup.sh \
  --protocol burgers_rar_transfer \
  --gpus 0,1,2,3,4,5,6,7 \
  --cpus 32 \
  --seeds 0-9
```

脚本先在服务器运行 smoke；20 个 smoke 全部通过后自动运行 full。重复执行同一命令会阻止重复流水线，已完成任务会跳过，中断任务会读取 checkpoint。

纯 gPINN 对照命令：

```bash
./experiments/poisson_1d_gpinn_rar/start_10seeds_nohup.sh \
  --protocol paper_gpinn \
  --gpus 0,1,2,3,4,5,6,7 \
  --cpus 32 \
  --seeds 0-9
```

不同协议自动写入不同目录，不会相互续训：

```text
results/poisson_1d_burgers_rar_transfer_10seeds/
results/poisson_1d_paper_gpinn_10seeds/
results/poisson_1d_upstream_poisson_10seeds/
```

## 进度和结果

```bash
tail -f ~/experiments/poisson_1d_gpinn_rar/results_burgers_rar_transfer_10seeds/pipeline.log
```

```bash
tail -f ~/experiments/poisson_1d_gpinn_rar/results_burgers_rar_transfer_10seeds/full/boundary_layer/seed_0/launcher.log
```

训练日志阶段依次为 `initial_adam`、`initial_lbfgs`，然后每轮 `rar_adam`、`rar_lbfgs`。全部完成后生成：

```text
results_burgers_rar_transfer_10seeds/full/results_summary.csv
results_burgers_rar_transfer_10seeds/full/results_mean_std.csv
results_burgers_rar_transfer_10seeds/full/results_mean_std.md
results_burgers_rar_transfer_10seeds/full/results_mean_std.xlsx
```

XLSX 只有一个总表，分别汇总两个方程 10 个随机种子的相对 L2 与最大逐点绝对误差 `mean ± sample std`。
