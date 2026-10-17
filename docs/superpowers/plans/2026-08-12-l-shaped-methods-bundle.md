# L-shaped Methods Bundle Implementation Plan

> **For agentic workers:** This plan was executed inline in the current workspace; source files were copied into an isolated bundle without modifying the original implementations.

**Goal:** Create a self-contained code bundle and precise experiment-setting guide for the L-shaped Poisson implementations used by PINN, SA-PINN, and DASA.

**Architecture:** Preserve each method in its own subdirectory, copy source/configuration/entry-point files, and use one README to document equations, domain, sampling, architectures, optimization, metrics, references, and fairness caveats.

**Tech Stack:** PyTorch, TensorFlow, DeepXDE, YAML/JSON, Markdown.

---

### Task 1: Inspect and map source files

- [x] Identify the original DASA source and all current method modules/configs.
- [x] Verify each path and implementation setting against source text.

### Task 2: Build the code bundle

- [x] Create `rebuttal/L_shape_Poisson_review_experiments/L_SHAPED_METHODS_FOR_GPT/` with method-specific subdirectories.
- [x] Copy source files, configs, runners, and reference-generation code without copying result artifacts or caches.
- [x] Add `cost_pinn_same_modified_backbone.json` as a separate same-backbone PINN configuration.

### Task 3: Write the GPT-facing experiment guide

- [x] Create `README_FOR_GPT.md` with exact PDE, geometry, BC, sampling, architectures, loss functions, schedules, metric equations, and entry points.
- [x] State implementation differences and comparison requirements explicitly.

### Task 4: Verify the bundle

- [x] Check copied paths, JSON/YAML parsing, and README references.
- [x] Confirm source code and configs match documented values.
- [x] Validate the distributable ZIP archive.
