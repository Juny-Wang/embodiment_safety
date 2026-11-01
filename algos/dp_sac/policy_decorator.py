"""Policy Decorator with frozen DP features and progressive residual exploration.

Method: https://arxiv.org/abs/2412.13630
Reference: tongzhoumu/policy_decorator, online/pi_dec_diffusion_maniskill2_rgbd.py.
"""
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .sac import Replay, SAC


def network(input_dim, output_dim, width):
    layers = []
    for _ in range(3):
        layers.extend((nn.Linear(input_dim, width), nn.ReLU()))
        input_dim = width
    head = nn.Linear(width, output_dim)
    nn.init.orthogonal_(head.weight, gain=.01)
    nn.init.zeros_(head.bias)
    return nn.Sequential(*layers, head)


def latest_observation(features):
    """Two observation embeddings followed by the seven-dimensional DP action."""
    frame_dim = (features.shape[-1] - 7) // 2
    return features[..., frame_dim:-7]


class DecoratorActor(nn.Module):
    def __init__(self, frame_dim, width):
        super().__init__()
        self.net = network(frame_dim, 14, width)

    def sample(self, features, deterministic=False):
        mean, raw_std = self.net(latest_observation(features)).chunk(2, dim=-1)
        log_std = -20. + 11. * (raw_std.tanh() + 1.)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        latent = mean if deterministic else distribution.rsample()
        residual = latent.tanh()
        log_prob = (distribution.log_prob(latent) -
                    2 * (math.log(2) - latent - F.softplus(-2 * latent))).sum(-1, keepdim=True)
        return residual, log_prob


class DecoratorQ(nn.Module):
    def __init__(self, frame_dim, width, scale):
        super().__init__()
        self.q1 = network(frame_dim + 7, 1, width)
        self.q2 = network(frame_dim + 7, 1, width)
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, features, residual):
        # Q evaluates the proposed sum; the environment clips its action bounds.
        action = features[..., -7:] + self.scale * residual
        value_input = torch.cat((latest_observation(features), action), dim=-1)
        return self.q1(value_input), self.q2(value_input)


class PolicyDecorator(SAC):
    def __init__(self, feature_dim, cfg, device="cuda", seed=0):
        self.cfg, self.device, self.feature_dim = cfg, torch.device(device), feature_dim
        if feature_dim <= 7 or (feature_dim - 7) % 2:
            raise ValueError("Expected two observation frames followed by a seven-dimensional base action")
        torch.manual_seed(seed)
        frame_dim = (feature_dim - 7) // 2
        self.actor = DecoratorActor(frame_dim, cfg["hidden_dim"]).to(device)
        self.critic = DecoratorQ(frame_dim, cfg["hidden_dim"], cfg["residual_scale"]).to(device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = torch.tensor(math.log(cfg["initial_alpha"]), device=device, requires_grad=True)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["learning_rate"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg["learning_rate"])
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg["learning_rate"])
        self.replay = Replay(cfg["replay_capacity"], feature_dim, seed)
        self.updates = self.behavior_steps = 0
        self.exploration_rng = np.random.default_rng(seed)

    def behavior_action(self, features, learn):
        if not learn:
            return self.action(features, deterministic=True), dict(
                residual_probability=1., residual_enabled=True, sampling_mode="residual_mean")
        self.behavior_steps += 1
        probability = min(self.behavior_steps / self.cfg["progressive_exploration_steps"], 1.)
        enabled = bool(self.exploration_rng.random() < probability)
        if not enabled:
            residual, mode = np.zeros(7, np.float32), "progressive_dp_only"
        elif self.updates == 0:
            residual = self.exploration_rng.uniform(-1, 1, 7).astype(np.float32)
            mode = "progressive_uniform"
        else:
            residual, mode = self.action(features), "progressive_residual_sample"
        return residual, dict(residual_probability=probability, residual_enabled=enabled,
                              sampling_mode=mode, behavior_steps=self.behavior_steps)

    def state_dict(self):
        return dict(super().state_dict(), behavior_steps=self.behavior_steps,
                    exploration_rng=self.exploration_rng.bit_generator.state)

    def load_state_dict(self, saved):
        super().load_state_dict(saved)
        self.behavior_steps = saved["behavior_steps"]
        self.exploration_rng.bit_generator.state = saved["exploration_rng"]
