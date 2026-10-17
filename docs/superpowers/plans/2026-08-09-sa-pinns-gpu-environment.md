# SA-PINNs GPU Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create and verify an isolated Python 3.10/TensorFlow 2.15.1 environment that executes SA-PINNs-style second derivatives on an RTX 4090 GPU.

**Architecture:** A pinned `environment.yml` defines Conda and pip dependencies, while a project-local `.conda-env` contains the resolved installation. Verification covers imports, bundled MATLAB data, a real GPU matrix operation, and nested automatic differentiation before an exported lock file records the resolved environment.

**Tech Stack:** Conda 26, Python 3.10, TensorFlow 2.15.1 with pip-managed CUDA 12 libraries, NumPy 1.26.4, SciPy 1.11.4, Matplotlib 3.7.5, pyDOE 0.3.8

---

### Task 1: Add the reproducible environment declaration

**Files:**
- Create: `environment.yml`
- Create: `.gitignore`

- [ ] **Step 1: Add the pinned environment manifest**

Create `environment.yml` with:

```yaml
name: sa-pinns-gpu
channels:
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
      - tensorflow[and-cuda]==2.15.1
      - numpy==1.26.4
      - scipy==1.11.4
      - matplotlib==3.7.5
      - pyDOE==0.3.8
```

- [ ] **Step 2: Exclude local and generated artifacts**

Create `.gitignore` with:

```gitignore
.conda-env/
__pycache__/
*.py[cod]
.DS_Store
.matplotlib/
```

- [ ] **Step 3: Check the manifest content**

Run:

```bash
conda env create --dry-run --prefix /data1/user/lz/SA-PINNs/.conda-env --file environment.yml
```

Expected: Conda resolves Python 3.10 and pip without a YAML or solver error.

The current project is untracked inside a parent repository whose `.git` directory is read-only, so a Git commit is unavailable. Preserve the files in the workspace and report this limitation.

### Task 2: Create the isolated environment

**Files:**
- Create: `.conda-env/` through Conda

- [ ] **Step 1: Create the Conda prefix**

Run:

```bash
conda env create --prefix /data1/user/lz/SA-PINNs/.conda-env --file environment.yml
```

Expected: exit code 0 and a Python executable at `.conda-env/bin/python`.

- [ ] **Step 2: Confirm the selected interpreter and package versions**

Run:

```bash
conda run --prefix /data1/user/lz/SA-PINNs/.conda-env python -c "import sys,tensorflow as tf,numpy,scipy,matplotlib,pyDOE; print(sys.version); print('tensorflow',tf.__version__); print('numpy',numpy.__version__); print('scipy',scipy.__version__); print('matplotlib',matplotlib.__version__); print('pyDOE import OK')"
```

Expected: Python 3.10, TensorFlow 2.15.1, NumPy 1.26.4, SciPy 1.11.4, Matplotlib 3.7.5, and a successful pyDOE import.

### Task 3: Verify data and GPU computation

**Files:**
- Read: `Allen-Cahn/AC.mat`
- Read: `Burgers/burgers_shock.mat`

- [ ] **Step 1: Validate the bundled MATLAB data**

Run:

```bash
conda run --prefix /data1/user/lz/SA-PINNs/.conda-env python -c "import scipy.io; ac=scipy.io.loadmat('Allen-Cahn/AC.mat'); bg=scipy.io.loadmat('Burgers/burgers_shock.mat'); assert ac['x'].shape==(1,512); assert ac['tt'].shape==(1,201); assert ac['uu'].shape==(512,201); assert bg['x'].shape==(256,1); assert bg['t'].shape==(100,1); assert bg['usol'].shape==(256,100); print('MAT data OK')"
```

Expected: `MAT data OK`.

- [ ] **Step 2: Verify TensorFlow discovers the selected RTX 4090**

Run:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --prefix /data1/user/lz/SA-PINNs/.conda-env python -c "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); print(g); assert g, 'TensorFlow found no GPU'"
```

Expected: one visible physical GPU and exit code 0.

- [ ] **Step 3: Execute a real matrix multiplication on the GPU**

Run:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --prefix /data1/user/lz/SA-PINNs/.conda-env python -c "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); assert g; tf.config.experimental.set_memory_growth(g[0],True); a=tf.ones((2048,2048)); c=tf.matmul(a,a); assert c.device.endswith('GPU:0'); print(c.device,float(c[0,0]))"
```

Expected: a device ending in `GPU:0` and the value `2048.0`.

- [ ] **Step 4: Verify the second derivative used by PINN residuals**

Run:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --prefix /data1/user/lz/SA-PINNs/.conda-env python -c "import tensorflow as tf; x=tf.Variable([[0.5]],dtype=tf.float32); outer=tf.GradientTape(); inner=tf.GradientTape(); outer.__enter__(); outer.watch(x); inner.__enter__(); inner.watch(x); y=x**3; inner.__exit__(None,None,None); dy=inner.gradient(y,x); outer.__exit__(None,None,None); d2y=outer.gradient(dy,x); tf.debugging.assert_near(d2y,tf.constant([[3.0]])); assert d2y.device.endswith('GPU:0'); print(d2y.device,d2y.numpy())"
```

Expected: a GPU device and `[[3.]]`.

### Task 4: Record and recheck the resolved environment

**Files:**
- Create: `environment-lock.yml`

- [ ] **Step 1: Export exact resolved dependencies**

Run:

```bash
conda env export --prefix /data1/user/lz/SA-PINNs/.conda-env --no-builds > environment-lock.yml
sed -i '/^prefix:/d' environment-lock.yml
```

Expected: `environment-lock.yml` includes Python, TensorFlow, CUDA pip wheels, and scientific dependencies without a machine-specific prefix.

- [ ] **Step 2: Run the final combined verification**

Run:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --prefix /data1/user/lz/SA-PINNs/.conda-env python -c "import tensorflow as tf,numpy,scipy,matplotlib,pyDOE; g=tf.config.list_physical_devices('GPU'); assert g; tf.config.experimental.set_memory_growth(g[0],True); x=tf.Variable([[0.5]],dtype=tf.float32); outer=tf.GradientTape(); inner=tf.GradientTape(); outer.__enter__(); outer.watch(x); inner.__enter__(); inner.watch(x); y=x**3; inner.__exit__(None,None,None); dy=inner.gradient(y,x); outer.__exit__(None,None,None); d2y=outer.gradient(dy,x); tf.debugging.assert_near(d2y,[[3.0]]); print('ENVIRONMENT_OK',tf.__version__,g,d2y.numpy())"
```

Expected: `ENVIRONMENT_OK 2.15.1`, one GPU, and `[[3.]]`.

- [ ] **Step 3: Document activation and execution commands in the handoff**

Report these commands:

```bash
conda activate /data1/user/lz/SA-PINNs/.conda-env
CUDA_VISIBLE_DEVICES=1 python Allen-Cahn/AC.py
```

Also state that source compatibility fixes remain a separate task and the unmodified training script is not yet claimed to complete.
