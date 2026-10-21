# 1D Poisson boundary-layer hp-VPINNs

本目录只包含该方程的 hp-VPINNs 实现，核心代码为
`poisson_boundary_hpvpinn.py`。

方程定义在 `x in [-1,1]`：

```text
-u_xx = f,
u(-1) = u_exact(-1),  u(1) = u_exact(1)
```

精确解为：

```text
u_exact(x) = 0.1 sin(5 pi x) + exp((0.01-(x+1))/0.01)
```

右端项 `f` 由该精确解代入方程得到。误差在 2001 个等距点上计算：

```python
relative_l2 = np.linalg.norm(y_true-y_pred) / np.linalg.norm(y_true)
max_pointwise = np.max(np.abs(y_true-y_pred))
```

hp-VPINNs 将区间划分为三个单元：

```text
[-1,-0.98], [-0.98,-0.9], [-0.9,1]
```

测试函数为 Legendre bubble 函数
`v_n=P_(n+1)-P_(n-1)`，积分使用 Gauss-Lobatto-Legendre 求积。
损失由各单元的加权方程残差和两端 Dirichlet 边界误差组成。

神经网络固定为 float64 的 `6×128 modified MLP`。训练分为四个连续阶段：

| 阶段 | p | q | 迭代数 | 学习率 |
|---|---:|---:|---:|---:|
| 1 | 60 | 80 | 40001 | 1e-3，gamma=0.9/2000步 |
| 2 | 60 | 80 | 20001 | 1e-5 |
| 3 | 100 | 140 | 30001 | 1e-5 |
| 4 | 180 | 200 | 30001 | 1e-5 |

后一阶段读取前一阶段的网络参数，最终结果取第四阶段。

