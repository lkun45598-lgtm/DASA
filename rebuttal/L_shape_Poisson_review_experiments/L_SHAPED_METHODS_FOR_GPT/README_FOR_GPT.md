# L-shaped Poisson 方法实现汇总

本目录用于把 L-shaped Poisson 各方法的实际代码交给 GPT，生成论文中的实验设置说明。原始代码没有覆盖；本目录只是按方法复制的代码包。请以代码和配置为准，不要把不同目录中的实现细节混写。

## 1. 统一问题定义

所有实现的目标问题是

$$
-\Delta u(x,y)=1, \qquad (x,y)\in\Omega,
$$

其中

$$
\Omega=[-1,1]^2\setminus([0,1]\times[0,1]).
$$

等价的多边形顶点（按边界顺序）为

```text
(-1,-1) -> (1,-1) -> (1,0) -> (0,0) -> (0,1) -> (-1,1).
```

边界条件为齐次 Dirichlet 条件：

$$
u(x,y)=0, \qquad (x,y)\in\partial\Omega.
$$

代码中的残差统一写成

$$
r(x,y)=-(u_{xx}+u_{yy})-1.
$$

右上角 $[0,1]\times[0,1]$ 被删除，内部点和评估点都不能落在该区域。

## 2. 目录和入口

```text
original_dasa/
  homogeneous Poisson(subnet).py       原始 PyTorch DASA notebook/script

refactored_dasa_pinn/
  src/lshape_experiment.py             可复现 PyTorch DASA/PINN runner
  configs/cost_dasa.json               DASA 成本实验配置
  configs/cost_pinn_same_modified_backbone.json
                                       成本公平对比配置：与 DASA 同 Modified MLP

sa_pinn_aligned/
  dasa_sa/problems/lshape.py           L-shaped PDE、采样、边界、参考解入口
  dasa_sa/networks.py                  TensorFlow Modified MLP
  dasa_sa/sa_weights.py                逐点可训练 SA mask 和 minimax 梯度
  dasa_sa/trainer.py                   SA-PINN Adam/L-BFGS 训练器
  configs/lshape_poisson.yaml          对齐配置

reference_data/
  Lpoission_more.mat                   原始 DASA 评估数据，键为 x、y、z

standard_pinn/modified_mlp/
  pinn_baselines/                       13 个标准 PINN 方程脚本
  pinn_baselines/common_modified_mlp_corrected.py
                                       与 DASA 对比使用的 gated Modified MLP
  reference_data/                       解析/数值参考解
  run_*.sh                              各方程启动脚本

hypinn/
  official_hpvpinn/                     官方 hp-VPINN 算例
  poisson_boundary_layer/               边界层 hp-VPINN 实现
  references/                           GradientPathologiesPINNs 参考代码
  root_scripts/                         统一运行/汇总脚本
```

重构代码中的 Python import 仍按仓库原始包路径书写；该目录的主要用途是查看和提供实现代码。若要直接运行，请从仓库根目录运行原始入口，并按各自 README 配置环境。

标准 PINN 的正式来源是服务器 `10.16.114.50` 上的
`/home/whn/hpy_pinns/260809_Chaoye_补实验_pinn.rar`。汇总包只保留其中的
`classical_pinn_baselines` 源码、启动脚本和必要参考数据，归档位置为
`standard_pinn/modified_mlp/`。13 个方程脚本均显式导入
`pinn_baselines/common_modified_mlp_corrected.py`；该文件实现 gated Modified MLP，
不是普通 fully-connected vanilla MLP。目录中的 `common.py` 是远程归档的旧实现，
仅用于历史口径审计，不是当前正式入口。

## 3. 原始 DASA 实现

代码：`original_dasa/homogeneous Poisson(subnet).py`。

### 网络

- Backbone 是 Modified MLP：6 个隐藏层，每层 128 个神经元，`tanh`，Xavier normal 初始化。
- 两个输入编码分支 `encoder_U` 和 `encoder_V` 各输出 128 维。
- 每个隐藏层先产生门控特征 $H$，再计算

  $$
  H\leftarrow(1-H)\odot U+H\odot V.
  $$

- DASA weighting subnet 是普通 DNN：5 个隐藏层，每层 10 个神经元，`Sigmoid`；输出层 bias（`fc5.bias`）被冻结。

### 采样

- 候选内部点：10,000 个二维 LHS 点，范围 $[-1,1]^2$。
- 删除右上象限后，保留约 7,500 个内部点；实际数量由 LHS 随机结果决定。
- 边界有 8 个线段，每段 128 点，共 1,024 个边界点：外边界六段加上内凹边界两段。
- 原脚本每 1,000 个 Adam step 重新采样一次。

### 损失和 DASA 四阶段

无权重残差损失为

$$
\mathcal L_r=\operatorname{mean}_i[r_i^2],
\qquad
\mathcal L_b=\operatorname{mean}_j[u(x_j,y_j)^2].
$$

总损失为

$$
\mathcal L=\mathcal L_r+100\mathcal L_b.
$$

使用 subnet 时，残差项变为

$$
\mathcal L_r^{\mathrm{weighted}}
 =\operatorname{mean}_i[(\hat\lambda_i r_i)^2].
$$

原始脚本的阶段是：

1. Stage 1：5,000 步，无权重训练 backbone。
2. Stage 2：2,000 步，仅更新 subnet，目标对 subnet 做梯度上升。
3. Stage 3：10,000 步，backbone 对损失下降、subnet 对损失上升；每 1,000 步更新 backbone 学习率。
4. Stage 4：10,000 步，关闭自适应权重，只训练 backbone。
5. L-BFGS：1,000 次外层调用，每次 `max_iter=100`，学习率 `0.8`，`tolerance_grad=1e-5`，`tolerance_change=1e-9`。

Adam 的 backbone 和 subnet 初始学习率均为 $10^{-3}$，Adam 的 $β=(0.9,0.999)$，Stage 3 使用指数衰减 `gamma=0.7`、每 1,000 步衰减一次。

### 原始归档中的权重归一化注意事项

原始 ZIP 文件中的 `subnet.minmaxscaler` 实际计算为

$$
\hat\lambda_i
 =\frac{\tilde\lambda_i}{\sum_j\tilde\lambda_j}N+1.
$$

因此其批次权重均值约为 2。当前正式重构的 DASA 代码遵循论文单位均值形式，去掉最后的 `+1`：

$$
\hat\lambda_i
 =\frac{\tilde\lambda_i}{\sum_j\tilde\lambda_j}N.
$$

不要在论文说明中把这两个版本写成同一个实现；原始文件保持原样，正式重构实验以 `cost_dasa.json` 为准。

## 4. 重构的 DASA/PINN 成本实验

代码：`refactored_dasa_pinn/src/lshape_experiment.py`。

该 runner 保留原问题、网络、采样、损失和训练阶段，同时增加确定性 seed、独立采样流、参数计数、CUDA 同步计时、显存统计和结构化 `metrics.json`。

### DASA 配置

文件：`refactored_dasa_pinn/configs/cost_dasa.json`。

- Backbone：Modified MLP，6x128，`tanh`，Xavier normal。
- Subnet：5x10，`Sigmoid`，输出 bias 冻结。
- 内部候选点 10,000，边界每段 128，边界权重 100。
- Adam 阶段 `[5000, 2000, 10000, 10000]`，总 Adam 预算 27,000 步；L-BFGS 配置为 1,000 个 outer steps、每个 outer step `max_iter=100`。
- 正式重构采用单位均值权重归一化，不带原始 ZIP 中的 `+1`。

### PINN 配置

文件 `cost_pinn_same_modified_backbone.json` 是公平成本对比应使用的配置：PINN 与 DASA 共享完全相同的 6x128 Modified MLP backbone、采样、边界权重和训练预算，PINN 仅去掉 weighting subnet 并使用固定残差权重 1。

## 5. SA-PINN 对齐实现

代码：`sa_pinn_aligned/`。

这不是 DASA 的 weighting subnet。它实现的是经典 SA-PINN 风格的 **每个残差点一个可训练 mask**：

$$
\mathcal L_r^{\mathrm{SA}}
 =\operatorname{mean}_i[(m_i r_i)^2].
$$

模型参数对总损失执行梯度下降，mask 参数对同一损失执行梯度上升。mask 在 `sa_weights.py` 中以独立随机 uniform 值初始化，代码没有 DASA subnet 的坐标到权重映射网络，也没有 DASA 的 subnet 预训练阶段。

### L-shaped SA 配置

文件：`sa_pinn_aligned/configs/lshape_poisson.yaml`。

- Modified MLP：6x128，`tanh`，Glorot normal，输入归一化到 $[-1,1]$。
- LHS 内部残差点：10,000 个；边界：8 段，每段 128 个，共 1,024 个。
- 残差权重 1，边界权重 100。
- 配置记录四段预算 `[5000, 2000, 10000, 10000]`，合计 27,000 个 Adam step，另有 1,000 个 L-BFGS step。
- 当前 `dasa_sa/trainer.py` 的 Adam 循环在每一步都执行 model/mask minimax 更新；`stages` 字段用于记录对齐预算，不能在说明中误称为已实现 DASA 四阶段。若需要严格四阶段 SA-PINN，必须另行实现阶段控制。

## 6. 标准 PINN（当前 DASA 对比入口）

论文表格和 DASA 的同 backbone 对比应使用
`standard_pinn/modified_mlp/`。其 13 个方程都使用：

- gated Modified MLP，6 个隐藏层、每层 128 个神经元；
- `tanh` 激活、Glorot/Xavier normal 初始化、`float64`；
- 代码实现：`standard_pinn/modified_mlp/pinn_baselines/common_modified_mlp_corrected.py`；
- 固定 PDE/边界/初值损失权重，不使用 DASA weighting subnet、SA mask 或对抗更新；
- 只输出 relative L2 error 和 maximum pointwise error。

其门控骨干按输入特征 $\phi$ 计算两个编码分支 $U,V$，先得到初始隐藏状态 $H_0$，
随后每一层产生门控变量 $Z$ 并更新

$$
H_{\ell+1}=(1-Z_{\ell})\odot U+Z_{\ell}\odot V,
\qquad
u_\theta= W_o H_L+b_o.
$$

这里的标准 PINN 只表示“固定损失权重的 PINN 优化目标”；网络骨干仍然是上述
Modified MLP。`refactored_dasa_pinn/configs/cost_pinn_same_modified_backbone.json`
是 L-shaped 成本实验中与 DASA 共享同一 backbone、采样和训练预算的 PyTorch PINN
配置，也应使用这一网络定义。

### 6.1 HP-VPINN

`hypinn/` 是 hp-VPINN/变分实现，使用局部单元、Gauss-Jacobi 求积和测试函数。
它与强形式 PINN、SA-PINN 和 DASA 的损失定义不同，应在论文中作为独立变分基线说明。

## 7. 数值参考解和评估

### 原始 DASA 参考数据

原始 PyTorch 脚本从 `reference_data/Lpoission_more.mat` 读取 `x`、`y`、`z`，其中 `z` 是评估网格上的参考解。原始评估只计算 relative L2 error。

### 当前统一数值参考

`sa_pinn_aligned/references/lshape_poisson.npz` 和同名 JSON manifest 是由 `reference_solvers.py` 生成的数值参考解：

- 五点有限差分离散；
- 稀疏直接求解器；
- 网格级别 101、201、401；
- 最终评估网格 401x401，移除右上象限；
- JSON manifest 保存方程参数、网格、收敛检查和 SHA-256。

SA-PINN、标准 PINN 和 DASA/HP-VPINN 的 L-shaped 结果若要严格横向比较，应统一评估坐标和同一份冻结参考值；重构 DASA/PINN cost runner 默认仍读取原始 MAT 文件，标准 PINN 默认使用 `standard_pinn/modified_mlp/reference_data/` 中的数值参考。

### 只报告两个指标

对形状相同的真实值和预测值：

$$
\mathrm{Relative\ L2}
 =\frac{\|y_{\mathrm{true}}-y_{\mathrm{pred}}\|_2}
 {\|y_{\mathrm{true}}\|_2},
$$

$$
\mathrm{Maximum\ pointwise\ error}
 =\max_i\left|y_{\mathrm{true},i}-y_{\mathrm{pred},i}\right|.
$$

代码实现位于 `sa_pinn_aligned/dasa_sa/metrics.py` 和
`standard_pinn/modified_mlp/pinn_baselines/common_modified_mlp_corrected.py`。
输出建议统一使用科学计数法，例如 `2.165968e-02`。

## 8. 给 GPT 的写作约束

请根据本目录代码生成一段专业的“L-shaped Poisson 实验设置”说明，至少包括：

1. PDE、区域和齐次 Dirichlet 边界；
2. 内部/边界采样数量及 LHS/边界分段方式；
3. 各方法 backbone 的层数、宽度、激活函数和初始化；
4. DASA 子网络、逐点 SA mask 与固定权重 PINN 的机制差异；
5. DASA 四阶段训练、SA-PINN 当前实际 minimax 训练循环、标准 PINN 的固定权重训练和 HP-VPINN 的变分训练；
6. 数值参考解来源及两个误差指标；
7. 明确 PINN-DASA 成本对比必须使用 `cost_pinn_same_modified_backbone.json`；
8. 不要声称 SA-PINN、PINN 或 DASA 在所有 PDE 上必然优于其他方法，也不要把运行时间或额外指标写成本文报告指标。
