"""Run with DP_SAC_BASE_TEST=1 on the Task1 server."""
import json
import os
from pathlib import Path
import unittest

import numpy as np
from runners.train_dp_sac import DEFAULT_BASE

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("DP_SAC_BASE_TEST") == "1", "Requires trained DP and recorded Task1 images")
class FrozenDPIntegrationTests(unittest.TestCase):
    def test_original_dp_equivalence_and_frozen_weights(self):
        import torch
        from algos.dp_sac.base import FrozenDP
        from algos.dp_sac.sac import SAC, compose_action
        torch.set_num_threads(2)
        seed = 20261101
        base = FrozenDP(ROOT / DEFAULT_BASE, "cuda", seed)
        source = ROOT / "outputs/runs/task1/diffusion_policy/full_flow_bc_v001/eval/episode_01/obs_000.npz"
        with np.load(source, allow_pickle=False) as f:
            obs = {k: f[k].copy() for k in f.files}
        batch = base.batch(obs)
        self.assertEqual(tuple(batch["state"].shape), (1, 2, 33))
        base.reset()
        torch.manual_seed(seed)
        with torch.no_grad():
            reference = base.model.predict_action(batch)["action"][0, 0].clamp(-1, 1).cpu().numpy()
        features, action = base.predict(obs)
        self.assertEqual(features.shape, (841,))
        np.testing.assert_allclose(action, reference, atol=1e-6, rtol=1e-5)
        np.testing.assert_array_equal(compose_action(action, np.zeros(7), [.05]*7), action)
        cfg = json.loads((ROOT / "configs/experiments/task1_dp_sac.json").read_text())
        cfg.update(batch_size=4, replay_capacity=8)
        agent = SAC(base.feature_dim, cfg, "cuda", seed)
        before = {k: p.detach().clone() for k, p in base.model.named_parameters()}
        for _ in range(4):
            agent.replay.append(features, np.zeros(7), .05, features, 1.)
        agent.update()
        for k, p in base.model.named_parameters():
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)
            self.assertTrue(torch.equal(p, before[k]))
        np.testing.assert_array_equal(agent.replay.arrays["observations"][0], features)


if __name__ == "__main__":
    unittest.main()
