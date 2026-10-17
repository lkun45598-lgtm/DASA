# SA-PINNs Minimal Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the canonical SA-PINNs helpers run correctly under Python 3.10, TensorFlow 2.15.1, and Matplotlib 3.7.5 while preserving all numerical defaults.

**Architecture:** Keep the three research examples independent and apply the same focused compatibility fix to each duplicated helper. Add one standard-library regression suite that loads helpers by file path, then verify the unchanged Burgers entry point end to end on GPU 1.

**Tech Stack:** Python 3.10, TensorFlow 2.15.1, Matplotlib 3.7.5, `unittest`, NVIDIA RTX 4090

---

### Task 1: Add regression tests that expose the current failures

**Files:**
- Create: `tests/test_legacy_helpers.py`
- Test: `tests/test_legacy_helpers.py`

- [x] **Step 1: Create the helper regression test module**

Create `tests/test_legacy_helpers.py` with:

```python
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock


os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
CASES = ("Allen-Cahn", "Burgers", "Helmholtz")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlottingCompatibilityTests(unittest.TestCase):
    def test_plotting_helpers_import_without_latex(self):
        import matplotlib.pyplot as plt

        for case in CASES:
            with self.subTest(case=case):
                with mock.patch("shutil.which", return_value=None):
                    module = load_module(ROOT / case / "plotting.py", f"plotting_{case}")
                self.assertFalse(module.mpl.rcParams["text.usetex"])
                fig, _ = module.newfig(0.5)
                plt.close(fig)


class LbfgsCompatibilityTests(unittest.TestCase):
    def test_stops_when_loss_change_is_below_tolerance(self):
        for case in CASES:
            with self.subTest(case=case):
                module = load_module(ROOT / case / "eager_lbfgs.py", f"lbfgs_tol_{case}")
                evaluated = []

                def numerically_unchanged_loss(x):
                    x = tf.convert_to_tensor(x)
                    evaluated.append(x.numpy().copy())
                    with tf.GradientTape() as tape:
                        tape.watch(x)
                        value = tf.constant(1e20, dtype=x.dtype) + tf.reduce_sum(x)
                    return value, tape.gradient(value, x)

                _, _, function_evaluations = module.lbfgs(
                    numerically_unchanged_loss,
                    tf.constant([0.0]),
                    module.Struct(),
                    maxIter=5,
                    do_verbose=False,
                )

                self.assertEqual(function_evaluations, 2)
                self.assertEqual(len(evaluated), 2)

    def test_final_parameter_is_evaluated(self):
        for case in CASES:
            with self.subTest(case=case):
                module = load_module(ROOT / case / "eager_lbfgs.py", f"lbfgs_{case}")
                evaluated = []

                def quadratic(x):
                    x = tf.convert_to_tensor(x)
                    evaluated.append(x.numpy().copy())
                    with tf.GradientTape() as tape:
                        tape.watch(x)
                        value = tf.reduce_sum(tf.square(x - 3.0))
                    return value, tape.gradient(value, x)

                result, history, _ = module.lbfgs(
                    quadratic,
                    tf.constant([0.0]),
                    module.Struct(),
                    maxIter=2,
                    learningRate=0.1,
                    do_verbose=False,
                )

                np.testing.assert_allclose(evaluated[-1], result.numpy())
                self.assertLess(float(history[-1]), float(history[0]))


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run the tests and verify the red state**

Run:

```bash
CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=3 MPLBACKEND=Agg MPLCONFIGDIR=/data1/user/lz/SA-PINNs/.matplotlib conda run --prefix /data1/user/lz/SA-PINNs/.conda-env python -m unittest -v tests/test_legacy_helpers.py
```

Expected: plotting subtests fail at `pgf.preamble`; L-BFGS runs past the unchanged-loss convergence point and fails the function-evaluation assertions; and L-BFGS fails the final-parameter assertion. The plotting test explicitly mocks LaTeX discovery so the fallback is covered even on machines that have a `latex` executable.

The current project is not tracked by a writable Git repository, so commits are unavailable. The untouched `SA-PINNs-master/` helper copies are the rollback source for every modified canonical helper.

### Task 2: Make plotting helpers compatible with modern Matplotlib

**Files:**
- Modify: `Allen-Cahn/plotting.py:10-41`
- Modify: `Burgers/plotting.py:10-41`
- Modify: `Helmholtz/plotting.py:10-41`
- Test: `tests/test_legacy_helpers.py`

- [x] **Step 1: Add executable detection and modern rcParams values to all three helpers**

In each file, add this import:

```python
import shutil
```

Replace the corresponding rcParams entries with:

```python
    "text.usetex": shutil.which("latex") is not None,
    "pgf.preamble": "\n".join([
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
    ]),
```

Leave all sizes, fonts, `newfig`, and `savefig` behavior unchanged.

- [x] **Step 2: Run only the plotting regression**

Run:

```bash
TF_CPP_MIN_LOG_LEVEL=3 MPLBACKEND=Agg MPLCONFIGDIR=/data1/user/lz/SA-PINNs/.matplotlib conda run --prefix /data1/user/lz/SA-PINNs/.conda-env python -m unittest -v tests.test_legacy_helpers.PlottingCompatibilityTests
```

Expected: one test passes with three successful subtests.

### Task 3: Correct the custom L-BFGS final update and convergence check

**Files:**
- Modify: `Allen-Cahn/eager_lbfgs.py:137-179`
- Modify: `Burgers/eager_lbfgs.py:137-179`
- Modify: `Helmholtz/eager_lbfgs.py:137-179`
- Test: `tests/test_legacy_helpers.py`

- [x] **Step 1: Evaluate every accepted parameter update**

In all three files, replace:

```python
    if nIter != maxIter:
    # re-evaluate function only if not in last iteration
    # the reason we do this: in a stochastic setting,
    # no use to re-evaluate that function here
      f, g = opfunc(x)
```

with:

```python
    # The objective closure also applies x to the model, including on the final step.
    f, g = opfunc(x)
```

- [x] **Step 2: Compare the actual loss difference**

In all three files, replace:

```python
    if tf.abs(f,f_old) < tolX:
```

with:

```python
    if tf.abs(f - f_old) < tolX:
```

- [x] **Step 3: Run only the L-BFGS regression**

Run:

```bash
CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=3 conda run --prefix /data1/user/lz/SA-PINNs/.conda-env python -m unittest -v tests.test_legacy_helpers.LbfgsCompatibilityTests
```

Expected: two tests pass with all three copies stopping on unchanged loss, evaluating the returned final parameter, and reducing the quadratic objective.

### Task 4: Remove the Helmholtz boundary argument's global dependency

**Files:**
- Modify: `Helmholtz/helmholtz.py:67-88`

- [x] **Step 1: Correct the argument name**

Replace:

```python
         x_lftb, ylftb,
```

with:

```python
         x_lftb, y_lftb,
```

The function body and all callers already use `y_lftb`, so no interface changes are required elsewhere.

- [x] **Step 2: Compile every canonical Python file**

Run:

```bash
conda run --prefix /data1/user/lz/SA-PINNs/.conda-env python -m compileall -q Allen-Cahn Burgers Helmholtz tests versiontest.py
```

Expected: exit code 0.

### Task 5: Run the complete helper regression suite on GPU 1

**Files:**
- Test: `tests/test_legacy_helpers.py`

- [x] **Step 1: Run all helper regressions**

Run:

```bash
CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=3 MPLBACKEND=Agg MPLCONFIGDIR=/data1/user/lz/SA-PINNs/.matplotlib conda run --prefix /data1/user/lz/SA-PINNs/.conda-env python -m unittest -v tests/test_legacy_helpers.py
```

Expected: three tests pass, covering three plotting copies and three L-BFGS copies.

- [x] **Step 2: Verify TensorFlow still executes on GPU 1**

Run:

```bash
CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=3 conda run --prefix /data1/user/lz/SA-PINNs/.conda-env python -c "import tensorflow as tf; g=tf.config.list_physical_devices('GPU'); assert len(g)==1; x=tf.ones((256,256)); y=tf.matmul(x,x); assert 'GPU:0' in y.device; print('GPU_TEST_OK',y.device,float(y[0,0]))"
```

Expected: `GPU_TEST_OK`, a `GPU:0` device, and `256.0`.

### Task 6: Execute the unchanged Burgers workflow end to end

**Files:**
- Read: `Burgers/burgers.py`
- Read: `Burgers/burgers_shock.mat`

- [x] **Step 1: Run Burgers from its expected working directory**

Run from `/data1/user/lz/SA-PINNs/Burgers`:

```bash
CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=3 MPLBACKEND=Agg MPLCONFIGDIR=/data1/user/lz/SA-PINNs/.matplotlib conda run --prefix /data1/user/lz/SA-PINNs/.conda-env python burgers.py
```

Expected: Adam and L-BFGS complete, the script prints a finite `Error u`, all plots render through the noninteractive backend, and the process exits 0.

- [x] **Step 2: Verify the archival snapshot was not modified**

Run:

```bash
find SA-PINNs-master -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum
```

Expected: `02540900d8d869c696c8b258ca2f8e24f59ecd3e67942ceda76289b33540914f  -`. A content hash is used because the parent repository reports the whole project as untracked and therefore cannot show individual archival changes.

### Plan Review

- **Goal clarity:** Success is objectively defined by helper regressions, compilation, GPU execution, and an end-to-end Burgers exit code.
- **Edit boundary:** Only three canonical plotting helpers, three canonical L-BFGS helpers, `Helmholtz/helmholtz.py`, and one new test file change. Main numerical scripts and `SA-PINNs-master/` remain unchanged.
- **Dependency impact:** Plotting helpers are imported by their local examples; L-BFGS helpers are imported by the local training scripts. Function names and public call signatures remain compatible.
- **Rollback:** Each modified helper can be restored from its exact archival copy under `SA-PINNs-master/`; the new test file can be removed independently. The archival tree's pre-change SHA-256 manifest hash is recorded in Task 6.
- **Verification:** Red/green unit tests, compileall, explicit GPU operation, and Burgers end-to-end execution cover the change.
- **Risks:** The custom fixed-step L-BFGS may expose another pre-existing numerical issue during Burgers training. Any new failure must be diagnosed before expanding this scope.

Plan review passed; implementation may begin after the execution approach is selected.
