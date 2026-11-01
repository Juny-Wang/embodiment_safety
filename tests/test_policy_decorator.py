import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from algos.dp_sac.policy_decorator import PolicyDecorator
from runners.train_dp_sac import validate_config

ROOT = Path(__file__).resolve().parents[1]


def config():
    c = json.loads((ROOT / "configs/experiments/task1_policy_decorator.json").read_text())
    c.update(device="cpu", hidden_dim=32, batch_size=8, replay_capacity=32,
             learning_starts=8, progressive_exploration_steps=20)
    return c


@unittest.skipIf(torch is None, "Requires the Diffusion Policy PyTorch environment")
class PolicyDecoratorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_actor_uses_latest_observation_without_base_action(self):
        agent = PolicyDecorator(25, config(), "cpu", 12)
        x = torch.randn(4, 25)
        changed = x.clone()
        changed[:, :9] += 20
        changed[:, -7:] -= 20
        expected, _ = agent.actor.sample(x, deterministic=True)
        actual, _ = agent.actor.sample(changed, deterministic=True)
        self.assertTrue(torch.equal(expected, actual))
        changed[:, 9:-7] += 20
        actual, _ = agent.actor.sample(changed, deterministic=True)
        self.assertFalse(torch.equal(expected, actual))

    def test_critic_depends_on_sum_and_backpropagates_residual_scale(self):
        agent = PolicyDecorator(25, config(), "cpu", 12)
        x = torch.randn(4, 25)
        residual = torch.randn(4, 7, requires_grad=True)
        seen = []
        handle = agent.critic.q1.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
        agent.critic(x, residual)
        expected = torch.cat((x[:, 9:-7], x[:, -7:] + .05 * residual), -1)
        torch.testing.assert_close(seen[0], expected)
        seen[0][:, -7:].sum().backward()
        torch.testing.assert_close(residual.grad, torch.full_like(residual, .05))
        handle.remove()
        alternate = x.clone()
        alternate[:, -7:] += .025
        q1 = agent.critic(x, residual)[0]
        q2 = agent.critic(alternate, residual - .5)[0]
        torch.testing.assert_close(q1, q2)

    def test_td_target_composes_next_base_action_and_honors_terminal_mask(self):
        agent = PolicyDecorator(25, config(), "cpu", 12)
        class FixedActor:
            def sample(self, obs):
                return torch.full((len(obs), 7), .4), torch.full((len(obs), 1), -2.)
        agent.actor = FixedActor()
        x = torch.zeros(2, 25)
        x[:, -7:] = .3
        seen = []
        hook = agent.target.q1.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
        b = dict(next_observations=x, rewards=torch.tensor([[1.], [.05]]),
                 masks=torch.tensor([[0.], [1.]]))
        actual = agent.target_value(b)
        torch.testing.assert_close(seen[0][:, -7:], torch.full((2, 7), .32))
        q1, q2 = agent.target(x, torch.full((2, 7), .4))
        expected = b["rewards"] + .99 * b["masks"] * (torch.minimum(q1, q2) + .02)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual[0].item(), 1.)
        hook.remove()

    def test_progressive_exploration_eval_and_exact_resume(self):
        agent = PolicyDecorator(25, config(), "cpu", 12)
        obs = np.linspace(-.2, .2, 25, dtype=np.float32)
        for _ in range(24):
            agent.replay.append(obs, np.zeros(7), .05, obs, 1.)
        before = [p.detach().clone() for p in agent.actor.parameters()]
        agent.update()
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, agent.actor.parameters())))
        for i in range(10):
            residual, info = agent.behavior_action(obs, True)
            self.assertEqual(info["residual_probability"], (i + 1) / 20)
            if not info["residual_enabled"]:
                np.testing.assert_array_equal(residual, np.zeros(7))
        saved = copy.deepcopy(agent.state_dict())
        expected_update = agent.update()
        expected = [agent.behavior_action(obs, True) for _ in range(15)]
        restored = PolicyDecorator(25, config(), "cpu", 99)
        restored.load_state_dict(saved)
        self.assertEqual(expected_update, restored.update())
        actual = [restored.behavior_action(obs, True) for _ in range(15)]
        for (a, info_a), (b, info_b) in zip(expected, actual):
            np.testing.assert_array_equal(a, b)
            self.assertEqual(info_a, info_b)
        self.assertEqual(actual[-1][1]["residual_probability"], 1.)
        restored.behavior_steps = 0
        rng_before = copy.deepcopy(restored.exploration_rng.bit_generator.state)
        _, info = restored.behavior_action(obs, False)
        self.assertTrue(info["residual_enabled"])
        self.assertEqual(restored.behavior_steps, 0)
        self.assertEqual(restored.exploration_rng.bit_generator.state, rng_before)

    def test_initial_exploration_samples_bounded_uniform_residuals(self):
        cfg = dict(config(), progressive_exploration_steps=1)
        agent = PolicyDecorator(25, cfg, "cpu", 12)
        residual, info = agent.behavior_action(np.zeros(25, np.float32), True)
        self.assertEqual(info["sampling_mode"], "progressive_uniform")
        self.assertTrue(np.all(np.abs(residual) <= 1))
        self.assertGreater(np.max(np.abs(residual)), .1)

    def test_config_and_checkpoint_algorithm_separation(self):
        cfg = config()
        validate_config(cfg)
        with self.assertRaises(ValueError):
            validate_config(dict(cfg, progressive_exploration_steps=0))
        agent = PolicyDecorator(25, cfg, "cpu", 12)
        saved = copy.deepcopy(agent.state_dict())
        saved["config"]["algorithm"] = "dp_residual_sac"
        with self.assertRaises(ValueError):
            agent.load_state_dict(saved)

    def test_worker_training_checkpoint_and_evaluation_protocol(self):
        from algos.dp_sac import worker
        class FixtureDP:
            def __init__(self, checkpoint, device, seed):
                self.path, self.feature_dim = Path(checkpoint), 25

            def predict(self, observation):
                return np.zeros(25, np.float32), np.zeros(7, np.float32)

        def requests(out):
            out.mkdir()
            for i in range(4):
                np.savez(out / f"obs_{i:03d}.npz", fixture=np.zeros(1))
                if i == 3:
                    break
                (out / f"request_{i:03d}.json").write_text('{}')
                label = dict(reward=float(i == 2), valid_for_training=True,
                             terminated=i == 2, truncated=False, mask=0. if i == 2 else 1.)
                (out / f"stage_transition_{i:03d}.json").write_text(json.dumps(label))
            (out / 'policy_stop.json').write_text(json.dumps(dict(final_observation_index=3)))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = dict(config(), learning_starts=2, batch_size=2,
                       checkpoint_every=2, progressive_exploration_steps=2)
            (root / 'config.json').write_text(json.dumps(cfg))
            env = dict(DP_SAC_CONFIG=str(root / 'config.json'), TASK1_ONLINE_LEARN='1',
                       TASK1_SAFETY_MODE='stop', TASK1_EPISODE_SEED='12',
                       SERL_INPUT_CHECKPOINT=str(root / 'base.pt'))
            training, evaluation = root / 'training', root / 'evaluation'
            requests(training)
            with patch.dict(os.environ, env, clear=True), patch.object(worker, 'FrozenDP', FixtureDP):
                worker.run(training)
            result = json.loads((training / 'worker_result.json').read_text())
            self.assertEqual(result['algorithm'], 'policy_decorator')
            self.assertEqual(result['network_updates'], 2)
            saved = torch.load(training / 'checkpoint.pt', weights_only=False)
            self.assertEqual(saved['sac']['behavior_steps'], 3)
            self.assertEqual(saved['sac']['replay']['total'], 3)
            for i in range(3):
                action = json.loads((training / f'action_{i:03d}.json').read_text())
                np.testing.assert_allclose(saved['sac']['replay']['arrays']['actions'][i],
                                           action['residual_action'])
            requests(evaluation)
            env.update(TASK1_ONLINE_LEARN='0', TASK1_EVAL='1',
                       DP_SAC_CHECKPOINT=str(training / 'checkpoint.pt'))
            with patch.dict(os.environ, env, clear=True), patch.object(worker, 'FrozenDP', FixtureDP):
                worker.run(evaluation)
            result = json.loads((evaluation / 'worker_result.json').read_text())
            self.assertEqual(result['network_updates'], 0)
            self.assertEqual(result['replay_added'], 0)
            self.assertFalse((evaluation / 'checkpoint.pt').exists())
            action = json.loads((evaluation / 'action_000.json').read_text())
            self.assertEqual(action['sampling_mode'], 'residual_mean')
            self.assertTrue(action['residual_enabled'])


if __name__ == "__main__":
    unittest.main()
