# 论文实验设置记录

## 时间 PDE 可统一复用的基础设置

以下时间 PDE 当前确认为使用统一的训练点数量：1D Burgers、1D Allen-Cahn、1D Diffusion-Reaction、1D Advection-Diffusion 和 1D Advection（Discontinuous）。

```text
Nr  = 25600
Nbc = 256  # 每侧边界点
Nic = 512
```

当前以上时间 PDE 统一使用 modified MLP backbone：

```text
depth = 6
width = 128
```

其他当前可统一复用的设置为：

```text
activation     = tanh
initializer    = Xavier/Glorot normal
precision      = float64
initial_lr     = 0.001
sampling       = LHS
```

其中 `tanh` 是 modified MLP 隐藏层的激活函数。学习率衰减方式和总训练步数仍按各方程在论文中的专属训练调度设置，不能直接全部照搬。

设置来源说明：论文明确给出了 Diffusion-Reaction 的点数、LHS、modified MLP 和优化信息，并说明 Advection-Diffusion 的其他训练细节相同；论文全局训练说明给出了 `tanh`、Xavier-normal 和初始学习率 `0.001`。Burgers、Allen-Cahn 和 Discontinuous Advection 的统一点数与网络规模属于当前实验补充确认，不能表述为论文对应小节逐项明确给出。

其中 `Nr`、`Nbc`、`Nic` 分别表示残差点、每侧边界点和初始点数量。显式计算左右边界损失的方程，两侧共享同一组 LHS 时间坐标进行对称采样。Allen-Cahn 通过周期 Fourier 映射严格满足边界条件，因此其实际训练损失只有残差项和初始条件项，不单独计算边界损失。

## 1D Helmholtz

论文 Eq. 12 中的 1D Helmholtz 方程写为：

\[
-u_{xx}+q(x)u=f(x), \qquad x\in[0,1].
\]

现已确认：

\[
q(x)=-(20\pi)^2=-400\pi^2,\qquad f(x)=0.
\]

当前代码严格实现 PDF Eq. 12：

\[
-u_{xx}+q(x)u=f(x).
\]

将 \(u(x)=\sin(20\pi x)\) 和 \(q=-(20\pi)^2\) 代入后，得到 \(f(x)=0\)，公式与解析解自洽。

此外，齐次方程 \(-u_{xx}-(20\pi)^2u=0\) 配合 \(u(0)=u(1)=0\) 是非唯一的：零解和任意 \(A\sin(20\pi x)\) 都满足方程。此前 standard PINN 因此收敛到接近零解。

当前补充一个内部观测条件：

\[
u\left(\frac{1}{40}\right)=1.
\]

因为 \(\sin(20\pi/40)=1\)，该条件将振幅固定为 \(A=1\)，排除零解。代码将其作为独立 PointSetBC，损失权重为 100。需要明确：该内部观测条件是为解决非唯一性而补充的实验条件，PDF 论文没有打印这一条件，因此更新后的问题不再严格等同于论文只写出的 Eq. 12 和两端零边界。

## 2D Poisson

以下两项现已确认：

1. 打印的 peak 解析解公式确实漏写了 \(y^a\)，完整解析解为：

   \[
   u(x,y)=2^{4a}x^a(1-x)^a y^a(1-y)^a, \qquad a=10.
   \]

2. steep 算例的计算区域确认为 \([-1,1]^2\)，与 Figure 3b 一致。

## 2D Helmholtz

当前代码采用：

```text
a1 = 1
a2 = 4
k  = 1
```

实验参数现已确认：

即 \(a_1=1\)、\(a_2=4\)、\(k=1\)。当前 2D Helmholtz 脚本已经使用这组参数，无需修改方程参数。

## L-Shaped Poisson

论文 Section 3.3.1 和 Appendix D Eq. 59 明确给出：

- 方程：\(-\Delta u=1\)。
- 区域：\(\Omega=[-1,1]^2\setminus[0,1]^2\)。
- 边界条件：完整边界上 \(u=0\)。
- 训练点：`Nr=10000`，`Nbc=128×8=1024`，使用 LHS。
- 损失权重：`wr=1`，`wb=100`。
- DASA 阶段：`NS1=5000`、`NS2=2000`、`NS3=NS4=10000`，之后 L-BFGS 1000 步。

当前 classical PINN 采用 modified MLP `depth=6`、`width=128`，并将 `NS3+NS4` 映射为一个 20000 步的标准 Adam 阶段；DASA 专属的 `NS1/NS2` 不计入 baseline。论文没有单独写出 classical PINN 的迭代数，也没有在 L-shaped 小节再次列出 backbone 规模，因此这两项属于依据“classical PINN 与 DASA 使用相同 architecture 和 hyper-parameters”作出的 baseline 映射。

论文参考解使用 FEALPy 自适应 FEM。当前项目使用本地生成的高分辨率五点有限差分参考场，因此可以计算相对 L2 和最大逐点误差，但与 Table 1 的 `9.25e-2` 不保证完全同口径。

## Lid-Driven Cavity (Re=100)

项目中已找到 `Lid-driven Cavity/reference_u.csv` 和 `reference_v.csv`，并转换为 `reference_data/lid_driven_cavity_re100_reference.npz`。参考场为 Re=100 的 `100×100` 速度场，转置后与当前 `(x,y)` 网格一致。

当前脚本可以计算与速度模长相关的：

- `l2_relative_error`
- `max_pointwise_error`

同时保存 `u/v` 分量误差，以及原 `Lid-driven Cavity/NS.py` 使用的速度矩阵 2-范数指标 `paper_source_speed_matrix_2norm_relative_error`。

仍需确认：该 CSV 参考场是否就是论文 [19] 使用的原始有限差分场，以及 Table 1 的 L2 误差究竟采用速度模长的标准相对 L2 还是矩阵 2-范数。

## 1D Burgers

论文 Section 3.2.1 给出了 Eq. 19 的方程、时空区域、初始条件、边界条件和误差结果。当前统一设置已确认为 `Nr=25600`、每侧 `Nbc=256`、`Nic=512`，backbone 现已确认为 modified MLP `depth=6`、`width=128`，Adam 训练步数确认为 `60000`，但仍没有明确给出：

- 是否使用 Fourier feature。
- classical PINN 的损失权重和是否使用 L-BFGS。

当前代码使用已确认的 `Nr=25600`、每侧 `Nbc=256`、`Nic=512`，并使用 modified MLP `depth=6`、`width=128` 和 60000 次 Adam 训练；不使用论文未明示给出的 Fourier feature 和 L-BFGS。

当前数值参考场与 Eq. 19 的方程和初边值一致，但论文没有明确说明 Figure 5a 是否使用完全相同的参考数据。

待确认：论文 1D Burgers 实验除当前已确认设置之外的完整损失、Fourier/L-BFGS 设置和参考数据来源。

## 1D Diffusion-Reaction

论文 Section 3.2.3 明确给出了 Eq. 21--23、`Nr=25600`、`Nbc=256`、`Nic=512`、LHS、modified MLP `depth=6`、`width=128`、Fourier feature embedding、四阶段迭代数和 L-BFGS 1000 epochs。

1. 论文只说明使用 Fourier feature embedding，没有给出 Fourier 映射的具体阶数。当前代码采用

   \[
   [t,\cos(kx),\sin(kx)]_{k=1}^{8},
   \]

   即 8 阶周期映射，以覆盖解析解中的最高频率 `sin(8x)`。

2. 边界点数量现已确认：`Nbc=256` 表示左右边界各 256 点，共 512 个边界约束；左右边界共享同一组 LHS 时间坐标，进行对称采样。

待确认：论文实际使用的 Fourier 映射和阶数。

## 1D Advection-Diffusion

当前补充确认：Fourier 映射和阶数与 1D Diffusion-Reaction 一致，采用 `[t, cos(k*x), sin(k*x)]`，`k=1,...,8`。

作者已补充确认扩散系数为

\[
\kappa=0.01.
\]

当前代码据此实现论文 Eq. 24：\(u_t+u_x=0.01u_{xx}\)，并使用论文说明的 1000 项解析级数生成参考场。由于 `kappa=0.01` 时解析展开存在严重指数抵消，参考场使用 80 位高精度求和后再转换为 `float64`。

待确认：Table 1 相对 L2 误差使用的测试网格；当前采用 `256×101` 均匀时空网格。

## 1D Advection (Discontinuous)

论文 Section 3.3.2 明确给出了 Eq. 26--28、`velocity=1`、`kappa=0`、modified MLP `depth=6`、`width=128`、损失权重、四阶段迭代数和 L-BFGS 1000 epochs。训练点数量现已确认为 `Nr=25600`、`Nbc=256`、`Nic=512`，采样方法确认为 LHS，目前仍需确认：

- Table 1 相对 L2 误差所使用的测试网格。

当前代码使用已确认的训练点规模和固定 LHS 采样。测试场采用 `256 x 101` 的均匀时空网格。

待确认：论文该实验的误差测试网格。

## 1D Allen-Cahn

论文 Section 3.2.2 明确给出了 Eq. 20、`Nr=25600`、`Nic=512`、Fourier feature backbone、初始损失权重 100、Stage 3 的 150000 次迭代和学习率衰减。backbone 规模现已确认为 modified MLP `depth=6`、`width=128`，Adam 总训练步数现已确认为 `300000`，但仍有以下设置没有完整写出：

- DASA 作者实际使用的 Fourier feature embedding 阶数。
- Figure 5b 和 Table 1 使用的数值参考文件。

当前代码采用 modified MLP `depth=6`、`width=128` 和暂定的 8 阶周期 Fourier 映射 `[t, cos(k*pi*x), sin(k*pi*x)]`、`k=1,...,8`。classical PINN 的 Adam 训练预算为已确认的 300000，不使用论文未明确提到的 L-BFGS。

已确认项目内 DeepXDE 附带的 `Allen_Cahn.mat` 对应扩散系数 `0.001`，而论文 Eq. 20 使用 `0.0001`，因此该 MAT 文件不能用于当前论文方程的误差计算。当前改为使用 `generate_allen_cahn_reference.py` 为 `d=0.0001` 生成周期 Fourier 伪谱空间离散、SciPy DOP853 时间积分的数值参考场。

待确认：DASA 作者实际使用的 Fourier 阶数（引用 [21] 的 Allen-Cahn 设置为 `m=10`，当前项目暂定为 8），以及 Figure 5b 使用的原始参考场和测试网格。当前生成场可用于一致方程下的 L2 和最大逐点误差，但不保证与论文原始参考场完全同口径。
