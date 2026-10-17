import math
import unittest

from benchmarks.dasa_aligned_sa.dasa_sa.config import (
    ALLOWED_PROVENANCE,
    list_case_ids,
    load_case_config,
)


EXPECTED_CASES = {
    "poisson_1d_boundary",
    "poisson_1d_steep",
    "helmholtz_1d",
    "poisson_2d_peak",
    "poisson_2d_steep",
    "helmholtz_2d",
    "cavity",
    "burgers",
    "allen_cahn",
    "diffusion_reaction",
    "advection_diffusion",
    "lshape_poisson",
    "discontinuous_advection",
}

NETWORK_OVERRIDES = {
    "diffusion_reaction": (4, 64),
    "discontinuous_advection": (4, 64),
}


class ConfigurationTests(unittest.TestCase):
    def test_loads_exactly_the_approved_cases(self):
        self.assertEqual(set(list_case_ids()), EXPECTED_CASES)

    def test_all_cases_use_the_approved_network_and_budget(self):
        for case_id in EXPECTED_CASES:
            with self.subTest(case_id=case_id):
                config = load_case_config(case_id)
                self.assertEqual(config.case_id, case_id)
                expected_depth, expected_width = NETWORK_OVERRIDES.get(
                    case_id, (6, 128)
                )
                self.assertEqual(config.network.hidden_layers, expected_depth)
                self.assertEqual(config.network.width, expected_width)
                self.assertEqual(config.network.activation, "tanh")
                self.assertEqual(config.network.initializer, "glorot_normal")
                self.assertEqual(
                    config.optimizer.adam_steps,
                    sum(config.optimizer.stages),
                )
                self.assertGreater(config.sampling.residual, 0)
                self.assertTrue(
                    set(config.provenance.values()).issubset(ALLOWED_PROVENANCE)
                )

    def test_resolves_the_approved_coefficients(self):
        advection = load_case_config("advection_diffusion")
        self.assertEqual(advection.parameters["velocity"], 1.0)
        self.assertEqual(advection.parameters["kappa"], 0.01)
        self.assertEqual(advection.provenance["parameters.kappa"], "inferred")

        helmholtz_1d = load_case_config("helmholtz_1d")
        self.assertEqual(helmholtz_1d.parameters["q"], (20.0 * math.pi) ** 2)
        self.assertEqual(helmholtz_1d.provenance["parameters.q"], "user")

        helmholtz_2d = load_case_config("helmholtz_2d")
        self.assertEqual(
            {key: helmholtz_2d.parameters[key] for key in ("a1", "a2", "k")},
            {"a1": 1.0, "a2": 4.0, "k": 1.0},
        )
        poisson_2d_steep = load_case_config("poisson_2d_steep")
        self.assertEqual(poisson_2d_steep.domain, ((-1.0, 1.0), (-1.0, 1.0)))

        boundary = load_case_config("poisson_1d_boundary")
        self.assertFalse(hasattr(boundary.optimizer, "mask_learning_rate"))
        self.assertEqual(boundary.optimizer.adam_beta_1, 0.99)
        self.assertEqual(helmholtz_1d.optimizer.adam_beta_1, 0.9)

    def test_rejects_unknown_case(self):
        with self.assertRaisesRegex(KeyError, "unknown case"):
            load_case_config("not_a_case")


if __name__ == "__main__":
    unittest.main()
