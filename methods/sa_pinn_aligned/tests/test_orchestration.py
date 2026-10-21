import csv
import json
from pathlib import Path
import tempfile
import unittest

from methods.sa_pinn_aligned.scripts.aggregate_results import aggregate_results
from methods.sa_pinn_aligned.scripts.monitor_runs import collect_status


ROOT = Path(__file__).resolve().parents[1]


class OrchestrationTests(unittest.TestCase):
    def test_smoke_script_runs_all_cases_with_tiny_overrides(self):
        script = (ROOT / "scripts" / "run_smoke_matrix.sh").read_text(encoding="utf-8")
        smoke_lines = [
            line.strip() for line in script.splitlines() if line.startswith("smoke_case ")
        ]
        self.assertEqual(len(smoke_lines), 13)
        self.assertIn("--adam-steps 2", script)
        self.assertIn("--lbfgs-steps 1", script)
        self.assertIn("wait", script)
        self.assertIn("methods/sa_pinn/burgers/burgers_shock.mat", script)
        self.assertIn("methods/sa_pinn/allen_cahn/AC.mat", script)
        self.assertIn("data/cavity_reference_u.csv", script)
        self.assertIn("for reference in lshape_poisson", script)

    def test_full_launch_has_exact_concurrent_gpu_mapping(self):
        script = (ROOT / "scripts" / "launch_full_tmux.sh").read_text(encoding="utf-8")
        expected = {
            "allen_cahn": "0",
            "diffusion_reaction": "1",
            "advection_diffusion": "2",
            "cavity": "3",
            "poisson_1d_boundary": "3",
            "lshape_poisson": "4",
            "poisson_1d_steep": "4",
            "helmholtz_2d": "5",
            "helmholtz_1d": "5",
            "burgers": "6",
            "poisson_2d_peak": "6",
            "discontinuous_advection": "7",
            "poisson_2d_steep": "7",
        }
        launch_lines = [
            line.strip() for line in script.splitlines() if line.startswith("launch_case ")
        ]
        self.assertEqual(len(launch_lines), 13)
        parsed = {line.split()[1]: line.split()[2] for line in launch_lines}
        self.assertEqual(parsed, expected)
        self.assertIn('SESSION="sa_dasa_full_seed0"', script)
        self.assertIn("TF_FORCE_GPU_ALLOW_GROWTH=true", script)
        self.assertIn("monitor_runs", script)
        self.assertNotIn("--adam-steps", script)
        self.assertNotIn("--lbfgs-steps", script)
        self.assertNotIn("\nwait\n", script)

    def test_aggregator_emits_exact_metric_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "helmholtz_1d"
            run.mkdir()
            (run / "metrics.json").write_text(
                json.dumps(
                    {
                        "case": "helmholtz_1d",
                        "seed": 0,
                        "l2_relative_error": 0.1,
                        "maximum_pointwise_error": 0.2,
                    }
                ),
                encoding="utf-8",
            )
            csv_path = root / "results.csv"
            markdown_path = root / "results.md"
            rows, failures = aggregate_results(root, csv_path, markdown_path)
            self.assertEqual(failures, [])
            self.assertEqual(len(rows), 1)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                self.assertEqual(
                    next(reader),
                    [
                        "case",
                        "seed",
                        "l2_relative_error",
                        "maximum_pointwise_error",
                    ],
                )

    def test_monitor_keeps_failures_outside_metric_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failed = root / "burgers"
            failed.mkdir()
            (failed / "failure.json").write_text('{"error":"failed"}', encoding="utf-8")
            statuses = collect_status(root, ["burgers", "allen_cahn"])
            self.assertEqual(statuses["burgers"]["status"], "failed")
            self.assertEqual(statuses["allen_cahn"]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
