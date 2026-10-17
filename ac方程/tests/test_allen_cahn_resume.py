import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from plain_pinn_baselines.allen_cahn_1d import (
    build_problem,
    checkpoint_signature,
    latest_checkpoint,
    train_with_checkpoints,
)
from plain_pinn_baselines.common import TrainConfig, dde, set_seed


class AllenCahnResumeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dde.config.set_default_float("float64")

    def test_resumed_adam_matches_uninterrupted_training(self):
        seed = 9917
        args = SimpleNamespace(seed=seed, num_domain=16, num_initial=8)
        full_config = TrainConfig(
            adam_iterations=4,
            decay_steps=2,
            display_every=100,
        )
        partial_config = TrainConfig(
            adam_iterations=2,
            decay_steps=2,
            display_every=100,
        )

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            set_seed(seed)
            uninterrupted_model = build_problem(args.num_domain, args.num_initial)
            signature = checkpoint_signature(args, full_config)
            train_with_checkpoints(
                uninterrupted_model,
                full_config,
                root / "uninterrupted",
                checkpoint_every=1,
                resume_mode="never",
                signature=signature,
            )
            uninterrupted_state = {
                name: value.detach().clone()
                for name, value in uninterrupted_model.net.state_dict().items()
            }

            set_seed(seed)
            partial_model = build_problem(args.num_domain, args.num_initial)
            train_with_checkpoints(
                partial_model,
                partial_config,
                root / "resumed",
                checkpoint_every=1,
                resume_mode="never",
                signature=checkpoint_signature(args, partial_config),
            )

            set_seed(seed)
            resumed_model = build_problem(args.num_domain, args.num_initial)
            _, _, resume_metadata = train_with_checkpoints(
                resumed_model,
                full_config,
                root / "resumed",
                checkpoint_every=1,
                resume_mode="auto",
                signature=signature,
            )

            final_step, _ = latest_checkpoint(root / "resumed")
            self.assertEqual(resume_metadata["resumed_step"], 2)
            self.assertEqual(final_step, 4)
            for name, value in resumed_model.net.state_dict().items():
                self.assertTrue(torch.equal(value, uninterrupted_state[name]), name)


if __name__ == "__main__":
    unittest.main()
