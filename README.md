# DASA-PINNs 实验代码

按 **method → 方程** 组织:每个 `methods/<method>/` 对应一种方法,其下按方程分目录或配置。
`experiments/` 存放跨方法的对比实验(rebuttal)及其结果。

```text
methods/
├── sa_pinn/            原版 Self-Adaptive PINN(TensorFlow)
│   ├── allen_cahn/ burgers/ helmholtz/   GPU 兼容修改版
│   ├── original/       McClenny & Braga-Neto 原始代码(未改动)
│   └── tests/
├── sa_pinn_aligned/    SA-PINN 对齐 DASA 论文 13 个正向算例(PyTorch,多种子)
├── dasa/               DASA-PINNs(权重预测子网络)
│   └── l_shape_poisson/{original,refactored}
├── plain_pinn/         经典 PINN 基线
│   ├── poisson_1d/  allen_cahn_1d/
├── deepxde_pinn/       DeepXDE 版 PINN 基线(13 算例)
├── gpinn/              gPINN / RAR 基线
│   ├── gpinn_baselines/(13 算例统一实现)  experiments/(Allen–Cahn、Burgers、Helmholtz、Poisson)
├── hp_vpinn/
│   └── poisson_1d_boundary_layer/
experiments/
├── rebuttal_l_shape_poisson/   L 形 Poisson 审稿实验(含各方法代码快照与结果)
└── rebuttal_wave_equation/     波动方程审稿实验
docs/superpowers/               设计文档与实施计划
```

方程覆盖见 [methods/README.md](methods/README.md)。

## 注意

- 各 method 内部是自包含的:从仓库根目录运行,例如
  `python -m methods.sa_pinn_aligned.scripts.run_case ...`、`python -m methods.deepxde_pinn.scripts.run_case ...`。
- `sa_pinn_aligned` 与 `deepxde_pinn` 读取 `methods/sa_pinn/burgers/burgers_shock.mat`
  和 `methods/sa_pinn/allen_cahn/AC.mat`。
- `experiments/` 下的代码快照是当时实验的原样记录,内部相对路径保持旧布局,不随 `methods/` 调整。
- 论文 PDF、压缩包、截图与模型权重(`.pt/.pth`)不入库,见 `.gitignore`。
- 环境:`environment.yml` / `environment-lock.yml`。

原 SA-PINN 说明见 [methods/sa_pinn/README.md](methods/sa_pinn/README.md)。
