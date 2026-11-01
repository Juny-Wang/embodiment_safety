import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from algos.dp_sac.sac import Actor, Replay, SAC, compose_action, transition_mask
from runners.train_dp_sac import validate_config

ROOT = Path(__file__).resolve().parents[1]


def config():
    c = json.loads((ROOT / "configs/experiments/task1_dp_sac.json").read_text())
    c.update(device="cpu", hidden_dim=32, batch_size=8, replay_capacity=32, learning_starts=8)
    return c


@unittest.skipIf(torch is None, "Requires the Diffusion Policy PyTorch environment")
class ResidualSACTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_zero_residual_and_action_bounds(self):
        base = np.array([.99, -.99, .2, -.4, 0, .8, -.5], np.float32)
        np.testing.assert_array_equal(compose_action(base, np.zeros(7), [.05]*7), base)
        corrected = compose_action(base, [1, -1, 1, -1, 1, 1, -1], [.05]*7)
        self.assertLessEqual(float(np.max(np.abs(corrected))), 1)
        self.assertLessEqual(float(np.max(np.abs(corrected-base))), .050001)
        actor = Actor(9, 32)
        action, log_prob = actor.sample(torch.randn(3, 9), deterministic=True)
        self.assertTrue(torch.equal(action, torch.zeros_like(action)))
        self.assertTrue(torch.isfinite(log_prob).all())

    def test_real_failure_and_time_limit_masks(self):
        label = dict(valid_for_training=True, terminated=True, truncated=False, mask=0.)
        self.assertEqual(transition_mask(label), 0.)
        self.assertEqual(transition_mask(dict(label, terminated=False, truncated=True, mask=1.)), 1.)
        for extra in (dict(injected=True), dict(diagnostic_only=True), dict(valid_for_training=False)):
            self.assertIsNone(transition_mask(dict(label, **extra)))
        with self.assertRaises(ValueError):
            transition_mask(dict(label, mask=1.))

    def test_terminal_target_does_not_bootstrap(self):
        agent = SAC(9, config(), "cpu", 12)
        b = dict(next_observations=torch.randn(3, 9), rewards=torch.tensor([[1.], [0.], [-1.]]),
                 masks=torch.zeros(3, 1))
        self.assertTrue(torch.equal(agent.target_value(b), b["rewards"]))
        b["masks"].fill_(1.)
        self.assertFalse(torch.equal(agent.target_value(b), b["rewards"]))

    def test_replay_wrap_and_sampling_resume(self):
        replay = Replay(3, 9, 4)
        for i in range(5):
            replay.append(np.full(9, i), np.full(7, i), i, np.full(9, i+1), 1.)
        restored = Replay(3, 9, 19)
        restored.load_state_dict(replay.state_dict())
        self.assertEqual((restored.size, restored.position, restored.total), (3, 2, 5))
        self.assertEqual(set(restored.arrays["rewards"].ravel()), {2, 3, 4})
        b1, b2 = replay.sample(20, "cpu"), restored.sample(20, "cpu")
        for k in b1:
            self.assertTrue(torch.equal(b1[k], b2[k]))

    def test_sac_updates_and_exact_resume(self):
        agent = SAC(9, config(), "cpu", 12)
        rng = np.random.default_rng(1)
        for _ in range(24):
            o, a, nxt = rng.normal(size=9), rng.uniform(-1, 1, 7), rng.normal(size=9)
            agent.replay.append(o, a, -np.square(a).sum(), nxt, 1.)
        before = [p.detach().clone() for p in agent.actor.parameters()]
        q_before = [p.detach().clone() for p in agent.critic.parameters()]
        for _ in range(4):
            result = agent.update()
            self.assertTrue(all(np.isfinite(v) for v in result.values()))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, agent.actor.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(q_before, agent.critic.parameters())))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "checkpoint.pt"
            torch.save(agent.state_dict(), cp)
            expected = agent.update()
            restored = SAC(9, config(), "cpu", 99)
            restored.load_state_dict(torch.load(cp, map_location="cpu", weights_only=False))
            actual = restored.update()
        self.assertEqual(expected, actual)
        for a, b in zip(agent.actor.parameters(), restored.actor.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_config_rejects_unusable_batches(self):
        c = config()
        validate_config(c)
        with self.assertRaises(ValueError):
            validate_config(dict(c, replay_capacity=1))
        with self.assertRaises(ValueError):
            validate_config(dict(c, state_dim=31))


if __name__ == "__main__":
    unittest.main()
