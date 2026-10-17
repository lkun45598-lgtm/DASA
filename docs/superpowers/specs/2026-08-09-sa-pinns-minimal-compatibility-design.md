# SA-PINNs Minimal Compatibility Design

## Goal

Make the canonical Allen-Cahn, Burgers, and Helmholtz examples compatible with the
project's Python 3.10/TensorFlow 2.15.1 GPU environment without refactoring the research
scripts or changing their numerical defaults.

## Scope

The canonical top-level example directories are the maintained implementation. The exact
duplicate under `SA-PINNs-master/` remains an archival snapshot and will not be modified.

The change covers:

- Modern Matplotlib configuration in the three `plotting.py` copies.
- Correct convergence checking and final parameter application in the three
  `eager_lbfgs.py` copies.
- The misspelled left-boundary argument in `Helmholtz/helmholtz.py`.
- Standard-library regression tests for plotting imports and all three L-BFGS copies.

The change does not cover:

- Packaging the examples as a reusable library.
- Adding command-line options or `__main__` guards.
- Changing PDE definitions, network widths, sample counts, learning rates, or iteration
  counts.
- Repairing or implementing the unfinished NTK analysis in `helmholtz-NTK.py`.
- Removing duplicated source directories.

## Plotting Compatibility

Each plotting helper will keep the existing publication-oriented style. The PGF preamble
will use the string format required by modern Matplotlib. `text.usetex` will be enabled only
when a `latex` executable is available; otherwise labels will use Matplotlib's built-in math
text so importing and plotting do not fail solely because TeX is absent.

## L-BFGS Correctness

The custom L-BFGS implementation will compare the actual loss change using
`tf.abs(f - f_old)`. It will evaluate the objective after every parameter update, including
the final iteration, so the model owned by the objective closure receives the returned final
weights. The optimization method, correction history, tolerances, and fixed learning-rate
behavior remain unchanged.

## Helmholtz Boundary Argument

The ordinary Helmholtz loss function will consistently name and use `y_lftb`. This removes
its accidental dependency on a global variable while preserving the call signature used by
the existing training code.

## Tests

A standard-library `unittest` module will load each helper directly from its file path.

1. Plotting import tests will verify all three helpers import under the configured modern
   Matplotlib environment without requiring LaTeX.
2. L-BFGS tests will optimize a one-dimensional quadratic for two iterations and verify that
   the objective is evaluated at the returned final parameter. This fails on the original
   implementation because of the malformed `tf.abs` call and skipped final evaluation.
3. Static compilation will cover all canonical Python files.
4. A GPU 1 test run will execute the unit tests in the configured environment.
5. The existing Burgers script will run with its current 100 Adam and 100 L-BFGS iterations
   under a noninteractive Matplotlib backend as an end-to-end smoke test.

## Failure Handling

- Test failures are diagnosed before further edits.
- The Burgers smoke test runs from its own directory so existing relative data imports remain
  unchanged.
- If full Burgers plotting requires unavailable external TeX behavior, the plotting helper's
  fallback is treated as part of this compatibility change; system TeX packages are not
  installed.
- Allen-Cahn and Helmholtz full 10,000/10,000 training runs are outside this change because
  they are long numerical experiments rather than compatibility tests.

## Success Criteria

- All new unit tests pass in `.conda-env`.
- All canonical Python files compile.
- The tests execute with TensorFlow seeing GPU 1.
- The Burgers process completes training, prediction, and plotting with exit code 0.
- No numerical defaults or archived files change.
