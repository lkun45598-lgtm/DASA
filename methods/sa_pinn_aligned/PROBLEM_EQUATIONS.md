# 高误差案例的控制方程

本文档记录 seed 0 全量训练中误差相对较高的四个案例。这里的“高误差”只表示本次
SA-PINN 结果需要进一步检查，并不等价于控制方程本身存在错误。以下公式、系数和条件
均以实际训练代码及其解析后的配置为准。

## 1. Advection-diffusion

### 定义域与未知量

未知量为标量场 $u(t,x)$，定义域为

$$
(t,x) \in [0,1] \times [-1,1].
$$

### 控制方程

$$
\frac{\partial u}{\partial t}
+c\frac{\partial u}{\partial x}
-\kappa\frac{\partial^2u}{\partial x^2}=0.
$$

其中

$$
c=1, \qquad \kappa=0.01.
$$

代入系数后，本次训练使用的方程为

$$
\frac{\partial u}{\partial t}
+\frac{\partial u}{\partial x}
-0.01\frac{\partial^2u}{\partial x^2}=0.
$$

### 初始条件与边界条件

$$
u(0,x)=-\sin(\pi x), \qquad x\in[-1,1],
$$

$$
u(t,-1)=u(t,1)=0, \qquad t\in[0,1].
$$

评估标签由 2049 个空间网格点上的四阶有限差分离散和稀疏矩阵指数生成。配置中的
$\kappa=0.01$ 标记为推定值，而方程、采样和网络设置标记为来自论文。

### Seed 0 指标

| Relative L2 Error | Maximum Pointwise Error |
|---:|---:|
| 5.078314e-01 | 2.334402e+00 |

## 2. Discontinuous advection

### 定义域与未知量

未知量为标量场 $u(t,x)$，定义域为

$$
(t,x) \in [0,1] \times [-1.5,1.5].
$$

### 控制方程

$$
\frac{\partial u}{\partial t}
+c\frac{\partial u}{\partial x}
-\kappa\frac{\partial^2u}{\partial x^2}=0.
$$

其中

$$
c=1, \qquad \kappa=0.
$$

因此本次训练实际求解的是纯对流方程

$$
\frac{\partial u}{\partial t}
+\frac{\partial u}{\partial x}=0.
$$

### 初始条件与边界条件

初始条件是半宽为 $0.2$ 的矩形脉冲：

$$
u(0,x)=
\begin{cases}
1, & |x|\leq 0.2,\\
0, & |x|>0.2.
\end{cases}
$$

边界条件为

$$
u(t,-1.5)=u(t,1.5)=0, \qquad t\in[0,1].
$$

对应的平移解为

$$
u(t,x)=u(0,x-t).
$$

### Seed 0 指标

| Relative L2 Error | Maximum Pointwise Error |
|---:|---:|
| 9.632342e-02 | 9.822161e-01 |

## 3. Lid-driven cavity

### 定义域与未知量

定义域为单位正方形

$$
(x,y)\in[0,1]^2.
$$

未知量为水平速度 $u(x,y)$、竖直速度 $v(x,y)$ 和压力 $p(x,y)$。

### 控制方程

本次训练求解稳态二维不可压缩 Navier-Stokes 方程：

$$
\begin{aligned}
u\frac{\partial u}{\partial x}
+v\frac{\partial u}{\partial y}
+\frac{\partial p}{\partial x}
-\frac{1}{\mathrm{Re}}
\left(
\frac{\partial^2u}{\partial x^2}
+\frac{\partial^2u}{\partial y^2}
\right)&=0,\\[4pt]
u\frac{\partial v}{\partial x}
+v\frac{\partial v}{\partial y}
+\frac{\partial p}{\partial y}
-\frac{1}{\mathrm{Re}}
\left(
\frac{\partial^2v}{\partial x^2}
+\frac{\partial^2v}{\partial y^2}
\right)&=0,\\[4pt]
\frac{\partial u}{\partial x}
+\frac{\partial v}{\partial y}&=0.
\end{aligned}
$$

其中 Reynolds 数为

$$
\mathrm{Re}=100.
$$

### 边界条件与压力规范

上壁以单位速度向右运动：

$$
u(x,1)=1, \qquad v(x,1)=0.
$$

其余三面为无滑移静止壁面：

$$
u(x,0)=v(x,0)=0,
$$

$$
u(0,y)=v(0,y)=0,
$$

$$
u(1,y)=v(1,y)=0.
$$

压力通过中心点规范条件消除常数不唯一性：

$$
p(0.5,0.5)=0.
$$

训练时边界采样不包含四个角点，因此上壁速度与侧壁速度在角点处的冲突没有直接进入
边界损失。压力规范条件和角点处理策略在配置中标记为推定设置。

本案例的网络输出为 $(u,v,p)$，但 seed 0 的两个误差指标只使用水平速度 $u$，并与
`data/cavity_reference_u.csv` 中的参考数据比较。

### Seed 0 指标（仅水平速度 $u$）

| Relative L2 Error | Maximum Pointwise Error |
|---:|---:|
| 8.733442e-02 | 2.211738e-01 |

## 4. Poisson 1D boundary layer

### 定义域与未知量

未知量为标量函数 $u(x)$，定义域为

$$
x\in[-1,1].
$$

### 控制方程

方程写作

$$
-\frac{\mathrm{d}^2u}{\mathrm{d}x^2}=f(x).
$$

其中

$$
f(x)=0.1\omega^2\sin(\omega x)
-\frac{1}{\varepsilon^2}
\exp\left(\frac{\varepsilon-(x+1)}{\varepsilon}\right),
$$

且

$$
\omega=5\pi, \qquad \varepsilon=0.01.
$$

因此展开系数后的方程为

$$
-\frac{\mathrm{d}^2u}{\mathrm{d}x^2}
=2.5\pi^2\sin(5\pi x)
-10^4\exp\left(\frac{0.01-(x+1)}{0.01}\right).
$$

### 边界条件与解析解

该问题使用的解析解为

$$
u(x)=0.1\sin(5\pi x)
+\exp\left(\frac{0.01-(x+1)}{0.01}\right).
$$

边界值由解析解直接给出：

$$
u(-1)=e, \qquad u(1)=e^{-199}.
$$

指数项在 $x=-1$ 附近形成尺度为 $\varepsilon=0.01$ 的尖锐边界层。

### Seed 0 指标

| Relative L2 Error | Maximum Pointwise Error |
|---:|---:|
| 5.119606e-02 | 2.877297e-02 |

## 实现来源

- 时间方程及初边值条件：`dasa_sa/problems/time_1d.py`
- Cavity 方程及边界条件：`dasa_sa/problems/cavity.py`
- 一维 Poisson 方程：`dasa_sa/problems/static_1d.py`
- 实验系数：`configs/cases/*.yaml`
- Seed 0 指标：`runs/seed0/*/metrics.json`
