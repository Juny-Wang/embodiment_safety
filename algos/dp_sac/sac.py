"""SAC in residual-action space with resumable online feature replay."""
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def mlp(input_dim, output_dim, width):
    return nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width), nn.ReLU(),
                         nn.Linear(width, width), nn.LayerNorm(width), nn.ReLU(),
                         nn.Linear(width, output_dim))


class Actor(nn.Module):
    def __init__(self, input_dim, width):
        super().__init__()
        self.net = mlp(input_dim, 14, width)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[7:] = -3.

    def sample(self, obs, deterministic=False):
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 1)
        dist = torch.distributions.Normal(mean, log_std.exp())
        latent = mean if deterministic else dist.rsample()
        action = latent.tanh()
        log_prob = (dist.log_prob(latent) -
                    2 * (math.log(2) - latent - F.softplus(-2 * latent))).sum(-1, keepdim=True)
        return action, log_prob


class TwinQ(nn.Module):
    def __init__(self, input_dim, width):
        super().__init__()
        self.q1, self.q2 = mlp(input_dim + 7, 1, width), mlp(input_dim + 7, 1, width)

    def forward(self, obs, action):
        x = torch.cat((obs, action), dim=-1)
        return self.q1(x), self.q2(x)


def compose_action(base, residual, scale):
    return np.clip(np.asarray(base) + np.asarray(scale) * np.asarray(residual), -1, 1).astype(np.float32)


def transition_mask(label):
    if label.get("injected") or label.get("diagnostic_only"):
        return None
    if not label.get("valid_for_training", label.get("evidence_valid", False)):
        return None
    expected = 0. if label["terminated"] else 1.
    if float(label["mask"]) != expected or (label["terminated"] and label["truncated"]):
        raise ValueError("Inconsistent episode termination mask")
    return expected


class Replay:
    def __init__(self, capacity, feature_dim, seed):
        self.capacity = capacity
        self.size = self.position = self.total = 0
        self.rng = np.random.default_rng(seed)
        self.arrays = {
            "observations": np.empty((capacity, feature_dim), np.float32),
            "next_observations": np.empty((capacity, feature_dim), np.float32),
            "actions": np.empty((capacity, 7), np.float32),
            "rewards": np.empty((capacity, 1), np.float32),
            "masks": np.empty((capacity, 1), np.float32),
        }

    def append(self, observations, actions, rewards, next_observations, masks):
        values = dict(observations=observations, actions=actions, rewards=rewards,
                      next_observations=next_observations, masks=masks)
        for key, value in values.items():
            self.arrays[key][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total += 1

    def sample(self, batch_size, device):
        ids = self.rng.integers(self.size, size=batch_size)
        return {k: torch.as_tensor(v[ids], device=device) for k, v in self.arrays.items()}

    def state_dict(self):
        return dict(size=self.size, position=self.position, total=self.total,
                    rng=self.rng.bit_generator.state,
                    arrays={k: v[:self.size].copy() for k, v in self.arrays.items()})

    def load_state_dict(self, state):
        if state["size"] > self.capacity:
            raise ValueError("Replay capacity smaller than checkpoint")
        self.size, self.position, self.total = state["size"], state["position"], state["total"]
        self.rng.bit_generator.state = state["rng"]
        for key, values in state["arrays"].items():
            self.arrays[key][:self.size] = values


class SAC:
    def __init__(self, feature_dim, cfg, device="cuda", seed=0):
        self.cfg, self.device, self.feature_dim = cfg, torch.device(device), feature_dim
        torch.manual_seed(seed)
        self.actor = Actor(feature_dim, cfg["hidden_dim"]).to(device)
        self.critic = TwinQ(feature_dim, cfg["hidden_dim"]).to(device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = torch.tensor(math.log(cfg["initial_alpha"]), device=device, requires_grad=True)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["learning_rate"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg["learning_rate"])
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg["learning_rate"])
        self.replay = Replay(cfg["replay_capacity"], feature_dim, seed)
        self.updates = 0

    @torch.no_grad()
    def action(self, obs, deterministic=False):
        x = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        action, _ = self.actor.sample(x, deterministic)
        return action[0].cpu().numpy()

    @torch.no_grad()
    def target_value(self, batch):
        action, log_prob = self.actor.sample(batch["next_observations"])
        q1, q2 = self.target(batch["next_observations"], action)
        return batch["rewards"] + self.cfg["discount"] * batch["masks"] * (
            torch.minimum(q1, q2) - self.log_alpha.exp() * log_prob)

    def update(self):
        b = self.replay.sample(self.cfg["batch_size"], self.device)
        target = self.target_value(b)
        q1, q2 = self.critic(b["observations"], b["actions"])
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        self.critic.requires_grad_(False)
        action, log_prob = self.actor.sample(b["observations"])
        q1, q2 = self.critic(b["observations"], action)
        actor_loss = (self.log_alpha.exp().detach() * log_prob - torch.minimum(q1, q2)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        self.critic.requires_grad_(True)
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.cfg["target_entropy"])).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            for target_param, param in zip(self.target.parameters(), self.critic.parameters()):
                target_param.lerp_(param, self.cfg["tau"])
        self.updates += 1
        metrics = dict(update=self.updates, critic_loss=critic_loss.item(),
                       actor_loss=actor_loss.item(), alpha=self.log_alpha.exp().item(),
                       entropy=-log_prob.mean().item(), mean_reward=b["rewards"].mean().item(),
                       replay_size=self.replay.size)
        if not all(np.isfinite(v) for v in metrics.values()):
            raise ValueError("Nonfinite SAC update")
        return metrics

    def state_dict(self):
        return dict(feature_dim=self.feature_dim, config=self.cfg,
                    actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    target=self.target.state_dict(), log_alpha=self.log_alpha.detach(),
                    actor_opt=self.actor_opt.state_dict(), critic_opt=self.critic_opt.state_dict(),
                    alpha_opt=self.alpha_opt.state_dict(), replay=self.replay.state_dict(),
                    updates=self.updates, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None)

    def load_state_dict(self, saved):
        if saved["config"] != self.cfg or saved["feature_dim"] != self.feature_dim:
            raise ValueError("SAC checkpoint configuration mismatch")
        for name in ("actor", "critic", "target", "actor_opt", "critic_opt", "alpha_opt"):
            getattr(self, name).load_state_dict(saved[name])
        with torch.no_grad():
            self.log_alpha.copy_(saved["log_alpha"])
        self.replay.load_state_dict(saved["replay"])
        self.updates = saved["updates"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if self.device.type == "cuda" and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), self.device)
