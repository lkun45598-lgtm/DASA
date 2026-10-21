# L-shaped Poisson：审稿意见 4–5 对比实验

本目录基于原始 `homogeneous Poisson(subnet).py` 重构为可命令行运行的实验。方程、L-shaped 区域、Modified MLP、DASA 子网络、采样规模、边界损失权重和 Adam 阶段均保持一致；新增了随机种子、统一采样流、参数统计、CUDA 同步计时、峰值显存记录、结构化结果和自动汇总。

## 0. 原始设置

- PDE：$-(u_{xx}+u_{yy})=1$，L-shaped domain 上齐次 Dirichlet 边界。
- Backbone：Modified MLP，6 个隐藏层，每层 128。
- DASA subnet：5 个隐藏层，每层 10，Sigmoid；输出层 bias 固定不训练。
- 内部候选点：10,000 个 LHS 点，删除右上象限后实际约 7,500 个。
- 边界点：8 条边，每条 128 个，共 1,024 个。
- Loss：$\mathcal L=\mathcal L_r+100\mathcal L_b$。
- DASA 点权重按论文 Eq. (8) 缩放为 $\hat\lambda_i=\alpha_iN/\sum_j\alpha_j$，因此批次均值为 1。ZIP 中保存的 notebook 在该表达式后额外写了 `+1`，会使均值固定为 2，与论文公式及其文字说明不一致；正式 rebuttal 实验遵循论文公式，原始归档文件保持不变。
- Adam：backbone LR $10^{-3}$，subnet LR $10^{-3}$。
- DASA 阶段：5,000 backbone warm-up；2,000 subnet pretraining；10,000 adversarial updates；10,000 unweighted fine-tuning。
- 原脚本 L-BFGS：1,000 次 outer call，每次 `max_iter=100`。

参数数目由代码自动统计。默认结构下：

- PINN/backbone：83,841 个 trainable parameters；
- DASA subnet：480 个 trainable parameters；
- DASA 总计：84,321；
- 参数增幅：0.573%。

## 1. 环境

```bash
conda create -n lshape-review python=3.10 -y
conda activate lshape-review
pip install -r requirements.txt
```

先做烟雾测试：

```bash
DEVICE=cpu bash run_smoke_test.sh
```

## 2. 第一组：计算开销对比

比较 `PINN` 与 `DASA`。两者使用完全相同的 backbone、同一随机种子下相同的 warm-up/main/fine-tuning 配点和相同的 backbone 更新次数。DASA 仅额外包含 subnet 预训练与联合更新。

正式建议先跑 Adam 部分：

```bash
DEVICE=cuda:0 SEEDS="0 1 2" PROFILE=adam bash run_cost_comparison.sh
```

输出：

```text
results_cost_adam/summaries/cost_summary.md
results_cost_adam/summaries/cost_summary.csv
results_cost_adam/summaries/all_runs.csv
```

如果论文中的最终结果确实使用了原脚本的 `1000 × max_iter=100` L-BFGS 设置，再运行：

```bash
DEVICE=cuda:0 SEEDS="0 1 2" PROFILE=full bash run_cost_comparison.sh
```

**不要未经核对直接启动 `PROFILE=full`。** 原脚本的 L-BFGS 外循环非常重，可能完全主导总训练时间。若论文实际只调用一次 L-BFGS，而不是 1,000 次，必须先修正配置和论文描述，不能拿错误脚本的运行时间回复审稿人。

该组自动输出：

- backbone、subnet 和总 trainable parameter 数；
- main/adversarial Adam 平均每步时间；
- Adam 总时间与完整训练总时间；
- peak GPU memory；
- relative $L_2$ error 和 maximum absolute error。

计时在阶段起止处执行 `torch.cuda.synchronize()`，不把 MAT 测试集评价、模型保存和绘图算入训练时间。

## 3. 第二组：DASA 超参数敏感性

采用 one-factor-at-a-time 设计。除被考察因素外，其余设置均固定为 baseline。所有变体在 adversarial stage 结束后立即评价，不运行无权重 fine-tuning 和 L-BFGS，避免后续优化掩盖 subnet 超参数本身的影响。

| 维度 | Variant | Subnet depth×width | Backbone LR | Subnet LR | Subnet:Backbone update ratio |
|---|---|---:|---:|---:|---:|
| Baseline | baseline | 5×10 | $10^{-3}$ | $10^{-3}$ | 1:1 |
| Architecture | small | 3×5 | $10^{-3}$ | $10^{-3}$ | 1:1 |
| Architecture | large | 7×20 | $10^{-3}$ | $10^{-3}$ | 1:1 |
| Learning rate | both low | 5×10 | $5\times10^{-4}$ | $5\times10^{-4}$ | 1:1 |
| Learning rate | subnet low | 5×10 | $10^{-3}$ | $5\times10^{-4}$ | 1:1 |
| Learning rate | subnet high | 5×10 | $10^{-3}$ | $2\times10^{-3}$ | 1:1 |
| Learning rate | both high | 5×10 | $2\times10^{-3}$ | $2\times10^{-3}$ | 1:1 |
| Update frequency | sparse | 5×10 | $10^{-3}$ | $10^{-3}$ | 1:5 |
| Update frequency | dense | 5×10 | $10^{-3}$ | $10^{-3}$ | 5:1 |

正式运行：

```bash
DEVICE=cuda:0 SEEDS="0 1 2" PROFILE=paper bash run_sensitivity.sh
```

快速筛查，不可直接用于论文：

```bash
DEVICE=cuda:0 SEEDS="0" PROFILE=quick bash run_sensitivity.sh
```

输出：

```text
results_sensitivity_paper/summaries/sensitivity_summary.md
results_sensitivity_paper/summaries/sensitivity_summary.csv
results_sensitivity_paper/summaries/all_runs.csv
```

### 3.1 多 GPU 并行运行全部正式实验

下面的入口会把 6 个成本实验和 27 个敏感性实验放入同一个动态队列，并在 tmux 后台运行。同一 seed 的 PINN/DASA 成本实验固定为同一队列单元，在同一张 GPU 上顺序执行。

```bash
SESSION=lshape_rebuttal_formal \
  bash run_all_parallel_tmux.sh results_rebuttal_formal 0 1 2 3 4 5 6 7
```

查看状态或进入 tmux：

```bash
bash run_all_parallel_tmux.sh --status results_rebuttal_formal
tmux attach -t lshape_rebuttal_formal
```

中断后继续运行，已经生成 `metrics.json` 的训练不会重复执行：

```bash
SESSION=lshape_rebuttal_formal \
  bash run_all_parallel_tmux.sh --resume results_rebuttal_formal 0 1 2 3 4 5 6 7
```

## 4. 公平性与复现细节

1. 同一 seed 下，PINN 与 DASA 在对应 backbone 阶段使用相同的确定性 LHS 配点；subnet pretraining 使用独立采样流，不会改变后续 backbone 的配点序列。
2. PINN 和 DASA 的 backbone 初始化在同一 seed 下完全一致。
3. 参数量统计只计 `requires_grad=True` 的参数，因此 subnet 被固定的 output bias 不计入 trainable parameters。
4. subnet-only 更新期间冻结 backbone 参数，去掉原 notebook 中计算后从未使用的 backbone parameter gradients；这不改变 subnet 的目标或梯度，只避免无效计算。
5. 每个正式表至少使用 3 个 seeds，并报告 mean ± standard deviation。

## 5. 结果解释底线

- 如果 DASA 的总时间明显更高，不要继续写 “on par”。改成 “introduces a moderate runtime overhead of XX% while reducing the error by YY%”。
- 只有在多数合理设置下误差接近 baseline，才能写 “not highly sensitive within a moderate range”。出现发散或 20% 以上退化时，应明确写出稳定区间与失败区间。
- 该成本实验只严格支持 **DASA 相对 classical PINN 的增量开销**。若正文仍声称与 SA-PINN、gPINN、hp-VPINN、Deep Ritz 等“所有 baselines”成本相当，就必须给这些方法也补相同硬件、相同计时口径下的结果，或者收缩原 claim。
