# Classical PINN A100 Rerun Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every PINN baseline gated Modified MLP with the original 6x128 `tanh` feed-forward PINN network, preserve old 4090 outputs, and rerun the L-shaped cost comparison and all 13 benchmark cases on A100 GPUs.

**Architecture:** Both runners receive an explicit classical fully connected network made of six hidden affine layers with 128 units, `tanh` activations, Xavier-normal weights, and trainable biases. The general DeepXDE suite uses normalized raw coordinates and no Fourier embedding; DASA retains its Modified MLP and paper-consistent unit-mean weighting. New A100 runs use separate output roots.

**Tech Stack:** Python 3.10, PyTorch 2.6, DeepXDE, pytest, Bash, tmux, NVIDIA A100-SXM4-80GB.

---

### Task 1: Replace the DeepXDE Modified MLP

**Files:**
- Modify: `benchmarks/deepxde_pinn/tests/test_network.py`
- Modify: `benchmarks/deepxde_pinn/network.py`
- Modify: `benchmarks/deepxde_pinn/trainer.py`
- Modify: `benchmarks/deepxde_pinn/README.md`

- [ ] **Step 1: Write a failing classical-network test**

Change the first network test to import `ClassicalMLP` and assert six hidden affine layers, width 128, standard output shapes, deterministic initialization, and absence of `u_encoder`, `v_encoder`, and `mixing_layers`. The expected parameter count is 82,945 for a 1D input and scalar output.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
DDE_BACKEND=pytorch /home/lz/miniconda3/envs/pytorch/bin/python -m pytest -q benchmarks/deepxde_pinn/tests/test_network.py
```

Expected: failure because `ClassicalMLP` is not defined.

- [ ] **Step 3: Implement the standard feed-forward network**

In `network.py`, replace the gated implementation with `ClassicalMLP`. Normalize each raw coordinate to `[-1, 1]`, apply six `Linear+tanh` hidden blocks, then one linear output layer. Initialize every weight with `torch.nn.init.xavier_normal_` and every bias with zeros. Preserve DeepXDE input/output transform hooks. Do not accept or apply Fourier mappings.

- [ ] **Step 4: Use and record the effective baseline network**

In `trainer.py`, instantiate `ClassicalMLP` without `config.network.fourier`. Add this explicit resolved metadata beside the shared case configuration:

```python
"baseline_network": {
    "architecture": "classical_fnn",
    "hidden_layers": 6,
    "width": 128,
    "activation": "tanh",
    "initializer": "glorot_normal",
    "input_mapping": "normalized_raw_coordinates",
}
```

This prevents shared DASA configuration fields from being mistaken for the effective PINN implementation.

- [ ] **Step 5: Update the README and run the focused suite**

State that the baseline follows Raissi et al. with a standard 6x128 `tanh` FNN and no gated or Fourier architecture. Run:

```bash
DDE_BACKEND=pytorch /home/lz/miniconda3/envs/pytorch/bin/python -m pytest -q benchmarks/deepxde_pinn/tests
```

Expected: all tests pass.

### Task 2: Add the classical network to the L-shaped cost runner

**Files:**
- Modify: `rebuttal/L_shape_Poisson_review_experiments/tests/test_weight_subnet.py`
- Modify: `rebuttal/L_shape_Poisson_review_experiments/src/lshape_experiment.py`
- Modify: `rebuttal/L_shape_Poisson_review_experiments/configs/cost/cost_pinn.json`
- Modify: `rebuttal/L_shape_Poisson_review_experiments/configs/cost/cost_dasa.json`
- Modify: `rebuttal/L_shape_Poisson_review_experiments/README.md`

- [ ] **Step 1: Write a failing architecture-selection test**

Add a test that constructs a 2D-input, scalar-output `VanillaMLP` with depth 6 and width 128, asserts six hidden affine layers, output shape `(4, 1)`, parameter count 83,073, and absence of gated encoder attributes.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
/home/lz/miniconda3/envs/pytorch/bin/python -m pytest -q rebuttal/L_shape_Poisson_review_experiments/tests/test_weight_subnet.py
```

Expected: failure because `VanillaMLP` is not defined.

- [ ] **Step 3: Implement explicit backbone selection**

Add `VanillaMLP` with six standard `Linear+tanh` hidden layers and Xavier-normal initialization. Add a small `build_backbone(backbone_cfg)` factory accepting `vanilla_mlp` and `modified_mlp`. Set `cost_pinn.json` to `vanilla_mlp` and `cost_dasa.json` to `modified_mlp`.

- [ ] **Step 4: Run local tests and smoke runs**

Run the complete L-shaped tests, followed by two-step CPU smoke runs for both cost configurations. Expected: tests pass, both runs create finite `metrics.json`, PINN reports 83,073 trainable parameters, and DASA reports 84,321 trainable parameters.

### Task 3: Deploy and launch the A100 experiments

**Files:**
- Preserve: `benchmarks/deepxde_pinn/runs/seeds_0_9`
- Create remotely: `benchmarks/deepxde_pinn/runs/classical_fnn_seeds_0_9_a100`
- Create remotely: `rebuttal/L_shape_Poisson_review_experiments/results_cost_classical_pinn_vs_dasa_full`

- [ ] **Step 1: Confirm the old 4090 process is stopped**

Check `tmux list-sessions` and the process table. Do not delete or resume the stale `seeds_0_9` queue. Expected: no active DeepXDE PINN process.

- [ ] **Step 2: Synchronize only required source and data to A100**

Copy the updated `benchmarks/deepxde_pinn`, its shared `benchmarks/dasa_aligned_sa` configuration/reference dependencies, and the updated L-shaped runner to `/home/lz/SA-PINNs` on `10.16.46.215`. Exclude local run directories and caches.

- [ ] **Step 3: Run one A100 smoke case**

Run Burgers seed 0 with two Adam steps, no L-BFGS, and reduced sampling. Verify `config_resolved.json` says `classical_fnn`, contains no gated/Fourier effective network, and metrics are finite.

- [ ] **Step 4: Prepare the L-shaped method-level result root**

Copy the three corrected DASA result directories into `results_cost_classical_pinn_vs_dasa_full/cost_dasa`. Start classical PINN seeds 0, 1, and 2 on A100 GPUs 0, 1, and 2 with full Adam and L-BFGS under tmux session `lshape_classical_pinn_full`.

- [ ] **Step 5: Start the 13-case matrix on the other five GPUs**

Launch seeds 0–9 for all 13 cases with run root `classical_fnn_seeds_0_9_a100` on GPUs 3–7 under tmux session `deepxde_pinn_classical_a100`. This creates 130 tasks without reusing any old Modified-MLP result.

- [ ] **Step 6: Reassign GPUs after L-shaped completion**

After all three L-shaped PINN metrics exist, add DeepXDE queue workers on GPUs 0–2. Verify eight workers are active or the queue is already exhausted.

### Task 4: Aggregate and update Comment 4

**Files:**
- Modify: `rebuttal/L_shape_Poisson_review_experiments/REBUTTAL_DRAFT.md`
- Create locally: `rebuttal/L_shape_Poisson_review_experiments/results_cost_classical_pinn_vs_dasa_full/`

- [ ] **Step 1: Aggregate the completed L-shaped runs**

Require six `metrics.json` files, run `aggregate_results.py`, and verify both error metrics and all timing fields are finite.

- [ ] **Step 2: Synchronize raw results and summaries locally**

Copy all six metrics, logs, checkpoints, and generated summaries to the local result directory without deleting the earlier same-backbone result root.

- [ ] **Step 3: Update the rebuttal table with correct attribution**

Use the classical PINN comparison as the primary table. State that the method-level runtime difference includes both DASA's Modified MLP and subnet. Retain the earlier same-backbone result as a controlled supplemental measurement of subnet overhead.

- [ ] **Step 4: Final verification**

Run both local test suites, recompute the table from raw metrics, verify the A100 model string, and check that the draft values match the CSV exactly. Do not claim completion of the 130-run matrix until all 130 metrics exist; report its tmux session and current status while it remains active.

## Repository Note

The project is an untracked directory inside the dirty parent Git repository `/data1/user/lz`. No commits will be created because doing so would mix this work with unrelated repository state. Verification will use scoped file inspection and test output instead.
