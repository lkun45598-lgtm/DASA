# SA-PINNs GPU Environment Design

## Goal

Create a reproducible, isolated environment that runs the existing SA-PINNs TensorFlow
implementation on the host's NVIDIA RTX 4090 GPUs. Preserve the published algorithm while
using a TensorFlow/CUDA stack that supports Ada GPUs.

## Decisions

- Use Python 3.10 and TensorFlow 2.15.1 instead of the README's Python 3.7 and TensorFlow
  2.3. TensorFlow 2.3's CUDA 10.1/cuDNN 7 stack does not officially support RTX 4090 GPUs.
- Install TensorFlow's pip-managed CUDA 12 dependencies so the environment does not depend
  on a matching host CUDA toolkit. The host NVIDIA 550.54.14 driver remains the only system
  dependency.
- Create an isolated Conda prefix under the project so the environment can be managed without
  changing existing Conda environments.
- Pin NumPy, SciPy, Matplotlib, and pyDOE to versions compatible with Python 3.10 and
  TensorFlow 2.15. Do not install standalone Keras 2.2.4; the scripts use `tensorflow.keras`.
- Keep environment setup separate from later source compatibility fixes. A successful
  environment means TensorFlow can use an RTX 4090 and perform the derivatives required by
  the PDE residuals, not that every unmodified training script already completes.

## Artifacts

- `environment.yml`: human-readable, reproducible Conda and pip dependency declaration.
- `.gitignore`: excludes the local environment and generated training/plot artifacts.
- `.conda-env/`: local environment prefix, excluded from version control.

## Installation Flow

1. Create `.conda-env` with Python 3.10 and pip.
2. Install the pinned scientific Python packages and TensorFlow 2.15.1 with CUDA dependencies.
3. Run all verification on an idle GPU, initially GPU 1.
4. Record exact resolved package versions after installation.

## Verification

The environment is accepted only when all of the following pass:

1. Python reports version 3.10.
2. TensorFlow imports and reports version 2.15.1.
3. TensorFlow lists at least one physical GPU.
4. A matrix multiplication executes on the selected GPU.
5. Nested automatic differentiation computes a finite second derivative, matching the
   operation required by the Allen-Cahn, Burgers, and Helmholtz residuals.
6. NumPy, SciPy, Matplotlib, and pyDOE import successfully.
7. Both bundled MATLAB data files load with their expected keys and shapes.

## Failure Handling

- If pip cannot resolve the CUDA extras, install TensorFlow 2.15.1 and its NVIDIA CUDA 12
  wheels using explicit pinned packages.
- If TensorFlow cannot load CUDA libraries, inspect the resolved shared libraries and adjust
  only environment activation paths; do not alter host drivers.
- If RTX 4090 execution fails despite GPU discovery, capture the exact kernel/runtime error
  before considering a newer TensorFlow release.
- Do not claim the environment is GPU-ready based only on `nvidia-smi` or GPU discovery;
  an actual TensorFlow GPU operation and second-derivative test are required.

## Out of Scope

- Fixing the custom L-BFGS implementation.
- Refactoring the training scripts or adding command-line configuration.
- Reproducing full paper metrics.
- Changing NVIDIA drivers or system-wide CUDA installations.
