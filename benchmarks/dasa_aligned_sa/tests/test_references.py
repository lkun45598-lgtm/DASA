import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmarks.dasa_aligned_sa.dasa_sa.reference_solvers import (
    solve_lshape_reference,
)
from benchmarks.dasa_aligned_sa.dasa_sa.references import (
    compare_refinements,
    load_reference,
    write_reference,
)
from benchmarks.dasa_aligned_sa.scripts.generate_references import REFERENCE_SPECS


class ReferenceArtifactTests(unittest.TestCase):
    def test_only_lshape_requires_a_generated_reference(self):
        self.assertEqual(set(REFERENCE_SPECS), {"lshape_poisson"})
        self.assertTrue(all(len(spec["grid_levels"]) == 3 for spec in REFERENCE_SPECS.values()))

    def test_refinement_comparison_uses_only_the_approved_formulas(self):
        fine = np.array([[3.0], [4.0]])
        coarse = np.array([[0.0], [4.0]])
        self.assertEqual(
            compare_refinements(coarse, fine),
            {"l2_relative_error": 0.6, "maximum_pointwise_error": 3.0},
        )

    def test_round_trip_manifest_contains_required_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator = root / "generator.py"
            generator.write_text("# deterministic generator\n", encoding="utf-8")
            points = np.array([[0.0], [1.0]])
            values = np.array([[1.0], [2.0]])
            write_reference(
                "example",
                points,
                values,
                root=root,
                generator_path=generator,
                equation_parameters={"coefficient": 1.0},
                solver={"method": "test"},
                grid_levels=[8, 16, 32],
                tolerances={"l2_relative_error": 1e-5, "maximum_pointwise_error": 1e-4},
                convergence={"l2_relative_error": 1e-6, "maximum_pointwise_error": 1e-5},
            )
            loaded_points, loaded_values, manifest = load_reference("example", root=root)
            np.testing.assert_array_equal(loaded_points, points)
            np.testing.assert_array_equal(loaded_values, values)
            self.assertEqual(
                {
                    "case_id",
                    "equation_parameters",
                    "solver",
                    "grid_levels",
                    "tolerances",
                    "convergence",
                    "generated_at",
                    "generator_sha256",
                    "reference_sha256",
                },
                set(manifest),
            )
            json.loads((root / "example.json").read_text(encoding="utf-8"))

    def test_rejects_a_tampered_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator = root / "generator.py"
            generator.write_text("# generator\n", encoding="utf-8")
            write_reference(
                "example",
                np.array([[0.0]]),
                np.array([[1.0]]),
                root=root,
                generator_path=generator,
                equation_parameters={},
                solver={"method": "test"},
                grid_levels=[2, 4, 8],
                tolerances={"l2_relative_error": 1.0, "maximum_pointwise_error": 1.0},
                convergence={"l2_relative_error": 0.0, "maximum_pointwise_error": 0.0},
            )
            with (root / "example.npz").open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_reference("example", root=root)


class ReferenceSolverTests(unittest.TestCase):
    def test_lshape_solver_has_zero_boundary_and_positive_interior(self):
        points = np.array(
            [[-1.0, -0.5], [0.0, 0.5], [0.5, 0.0], [-0.5, -0.5], [-0.5, 0.5]]
        )
        values = solve_lshape_reference(33, points)
        np.testing.assert_allclose(values[:3], 0.0, atol=1e-12)
        self.assertTrue(np.all(values[3:] > 0.0))

if __name__ == "__main__":
    unittest.main()
