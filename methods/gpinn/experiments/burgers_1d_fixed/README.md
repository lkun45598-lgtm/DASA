# 1D Burgers 修复版

该目录只运行一维黏性 Burgers 方程：

$$
u_t+u u_x-\frac{0.01}{\pi}u_{xx}=0,\qquad (x,t)\in[-1,1]\times[0,1].
$$

初值 $u(x,0)=-\sin(\pi x)$ 和边界 $u(-1,t)=u(1,t)=0$ 由网络输出变换严格满足，不作为软损失。

## 十次运行

默认使用 `nohup` 自动运行 seed 0–9；检测到多张 CUDA GPU 时，每张 GPU 同时运行一个种子，空闲后继续领取剩余种子：

```bash
bash experiments/burgers_1d_fixed/start_10seeds_nohup.sh
```

也可以显式指定 GPU：

```bash
bash experiments/burgers_1d_fixed/start_10seeds_nohup.sh --gpus 0,1,2,3,4,5,6,7
```

启动后总日志保存在 `results/burgers_1d_fixed_10seeds/nohup_launcher.log`，PID 保存在同一目录。已完成的种子会自动跳过，存在检查点但未完成的种子会自动续跑。十次实验完成后生成 `summary_10seeds.json`、`summary_10seeds.csv` 和 `summary_10seeds.md`，其中包含均值和样本标准差。

## 单次运行

在项目根目录执行：

```bash
python experiments/burgers_1d_fixed/run.py --device cuda
```

没有 CUDA 时可以使用 CPU，但完整训练会比较慢：

```bash
python experiments/burgers_1d_fixed/run.py --device cpu
```

单次任务中断后从最近检查点继续：

```bash
python experiments/burgers_1d_fixed/run.py --device cuda --resume
```

先做快速完整性检查：

```bash
python experiments/burgers_1d_fixed/run.py --mode smoke --device cpu
```

结果默认保存在 `results/burgers_1d_fixed/seed_0/`，包括指标、预测数据、RAR 加点、检查点、训练曲线和解的误差图。参考解位于 `data/burgers_1d_fixed_reference_data/`。

完整模式默认仍使用与原项目一致的 6×128、float64 网络，共执行 20000 步预训练和 10 轮、每轮 3000 步的 RAR 训练。

## 修复点

- Adam 阶段持续重采样内部点，避免大网络记住固定的 1500 个配点。
- 恢复 Burgers 官方实现中的残差自适应加点（RAR），重点解析陡峭梯度区。
- 根据方程、初值和边界的对称性，将解严格参数化为关于 $x$ 的奇函数，固定 $x=0$ 的薄激波位置，消除不同种子下的激波漂移。
- 使用独立、固定的留出配点监控连续域残差，并按留出残差保存最佳模型。
- L-BFGS 使用更密的固定点集和全部 RAR 锚点；结束后仍以留出残差选择模型，防止再次过拟合。
- 使用独立的高分辨率 IMEX-BDF2 数值解评估，不把参考解用于训练。

## 核验

修复版已通过 5000 步 Adam、3 轮 RAR 和 200 步 L-BFGS 的 CPU 中等规模核验。3×32 网络在 6579 个测试点上得到：相对 L2 误差 $2.6074\times10^{-2}$，最大绝对误差 $1.6277\times10^{-1}$。该核验规模明显小于完整默认配置。

默认高分辨率参考解与官方 `Burgers.npz` 在共同网格上的相对 L2 差为 $2.9080\times10^{-4}$，最大绝对差为 $3.1014\times10^{-3}$。
