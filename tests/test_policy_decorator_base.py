"""DP_SAC_BASE_TEST=1 enables the real Task1 checkpoint test."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("DP_SAC_BASE_TEST") == "1", "Requires Task1 checkpoint and observations")
class PolicyDecoratorBaseTests(unittest.TestCase):
    def test_real_dp_features_gpu_update_and_checkpoint(self):
        import torch
        from algos.dp_sac.base import FrozenDP
        from algos.dp_sac.policy_decorator import PolicyDecorator
        from algos.dp_sac.sac import compose_action
        from runners.train_dp_sac import DEFAULT_BASE
        torch.set_num_threads(2)
        base = FrozenDP(ROOT / DEFAULT_BASE, "cuda", 20261101)
        source = ROOT / "outputs/runs/task1/dp_sac/integration_001/episode_001"
        cfg = json.loads((ROOT / "configs/experiments/task1_policy_decorator.json").read_text())
        cfg.update(batch_size=4, replay_capacity=8, learning_starts=4,
                   progressive_exploration_steps=4)
        agent = PolicyDecorator(base.feature_dim, cfg, "cuda", 20261101)
        before = {key: p.detach().clone() for key, p in base.model.named_parameters()}
        for i in range(4):
            with np.load(source / f"obs_{i:03d}.npz", allow_pickle=False) as f:
                features, action_base = base.predict({k: f[k].copy() for k in f.files})
            self.assertEqual(features.shape, (841,))
            residual, info = agent.behavior_action(features, True)
            final = compose_action(action_base, residual, cfg["residual_scale"])
            self.assertLessEqual(float(np.max(np.abs(final - action_base))), .050001)
            # Synthetic terminal reward tests the update path, not task performance.
            agent.replay.append(features, residual, float(i == 3), features, 0.)
        agent.update()
        for key, p in base.model.named_parameters():
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)
            self.assertTrue(torch.equal(p, before[key]))
        with tempfile.TemporaryDirectory() as directory:
            cp = Path(directory) / "checkpoint.pt"
            torch.save(agent.state_dict(), cp)
            expected = agent.update()
            restored = PolicyDecorator(base.feature_dim, cfg, "cuda", 7)
            restored.load_state_dict(torch.load(cp, map_location="cpu", weights_only=False))
            self.assertEqual(expected, restored.update())
        self.assertEqual(restored.behavior_steps, 4)
        self.assertEqual(agent.actor.net[0].in_features, 417)


if __name__ == "__main__":
    unittest.main()
