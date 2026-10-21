import unittest

from methods.sa_pinn_aligned.scripts.run_case import build_parser


class RunCaseCLITests(unittest.TestCase):
    def test_requires_case_seed_and_output(self):
        arguments = build_parser().parse_args(
            ["--case", "helmholtz_1d", "--seed", "4", "--output", "/tmp/run"]
        )
        self.assertEqual(arguments.case, "helmholtz_1d")
        self.assertEqual(arguments.seed, 4)
        self.assertIsNone(arguments.adam_steps)
        self.assertIsNone(arguments.lbfgs_steps)

    def test_accepts_explicit_smoke_budget_overrides(self):
        arguments = build_parser().parse_args(
            [
                "--case",
                "helmholtz_1d",
                "--seed",
                "0",
                "--output",
                "/tmp/run",
                "--adam-steps",
                "2",
                "--lbfgs-steps",
                "1",
            ]
        )
        self.assertEqual((arguments.adam_steps, arguments.lbfgs_steps), (2, 1))


if __name__ == "__main__":
    unittest.main()
