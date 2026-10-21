# 1D Poisson 十种子复跑

此目录只运行论文式 (10)–(11) 的两个 1D Poisson 算例：

- `poisson_1d_boundary_layer`
- `poisson_1d_steep`

## 协议

- 纯 gPINN：损失为 PDE 残差与残差对 $x$ 的导数增强项，不含 RAR、DASA、SA-PINN 或可训练点权重。
- 网络：6 个隐藏层，每层 128 个神经元，`tanh`，Xavier-normal，float64。
- 采样：500 个固定 Latin-hypercube 域内点，种子为 0–9。
- 优化：Adam 20,000 步，恒定学习率 $10^{-3}$，无 L-BFGS。
- gPINN 梯度权重：0.01。
- Dirichlet 条件：采用原始 gPINN Poisson 风格的输出变换严格满足，不再重复加入软边界损失。
- 测试：解析解上的 2,001 点致密网格。

这是修正后的独立协议。默认输出写入 `results/poisson_1d_pure_gpinn`，不会覆盖主线的 `results/gpinn_baselines_10seeds`。

## 启动

在仓库根目录执行：

```bash
conda activate dl
nohup ./experiments/poisson_1d_10seeds/run_10seeds.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --cpus 32 \
  --seeds 0-9 \
  > experiments/poisson_1d_10seeds/launcher.log 2>&1 &
```

脚本先运行两个算例的 20 个 smoke 任务；只有全部通过、指标能从预测文件重算且没有 NaN/Inf，才自动启动 20 个完整训练任务。中断后重复相同命令会从 checkpoint 继续。

查看进度：

```bash
tail -f results/poisson_1d_pure_gpinn/pipeline.log
```

检查是否完成：

```bash
test -f results/poisson_1d_pure_gpinn/full/results_mean_std.xlsx \
  && echo "全部完成" || echo "仍在运行或失败"
```

最终文件：

- `full/results_summary.csv`：20 行逐种子结果；
- `full/results_mean_std.csv`：两个算例的 mean、样本 std 和 mean ± std；
- `full/results_mean_std.xlsx`：只含一个工作表的两行总表；
- 每个 `seed_*` 目录：配置、checkpoint、日志、指标、预测、参考值和图。

## RAR 说明

原始 gPINN 仓库将 1D Poisson 列为普通 gPINN 示例；其 RAR 增强示例是 Burgers 和 Allen–Cahn。因此这里没有把 RAR 混入 Poisson 基线。若以后需要比较 RAR，它应作为单独命名的消融实验，不能与此纯 gPINN 结果混合统计。
