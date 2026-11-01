"""Reuse the trained DP's observation features and single-step action."""
import json
from pathlib import Path

import numpy as np
import torch

from algos.diffusion_policy.common import CAMERAS, build
from algos.diffusion_policy.representation import ROT6D, encode_state, validate_config


class FrozenDP:
    def __init__(self, checkpoint, device="cuda", seed=0):
        self.path = Path(checkpoint).resolve()
        meta = json.loads(self.path.with_suffix(".json").read_text())
        if meta.get("algorithm") != "diffusion_policy":
            raise ValueError("Expected a trained Diffusion Policy checkpoint")
        saved = torch.load(self.path, map_location="cpu", weights_only=False)
        self.config = saved["config"]
        contract = validate_config(self.config)
        if contract["state_representation"] != ROT6D or self.config["n_obs_steps"] != 2:
            raise ValueError("DP-SAC requires rotation6d33_v002 and two observation frames")
        if self.config["n_action_steps"] != 1:
            raise ValueError("DP-SAC executes one base action per observation")
        self.model = build(self.config)
        self.model.load_state_dict(saved["ema"])
        self.model.to(device).eval().requires_grad_(False)
        self.device = torch.device(device)
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.history = []
        self.feature_dim = self.model.obs_feature_dim * 2 + 7

    def reset(self):
        self.history.clear()

    def batch(self, observation):
        self.history.append({k: np.asarray(v)[0].copy() for k, v in observation.items()})
        self.history = self.history[-2:]
        pair = self.history if len(self.history) == 2 else self.history * 2
        batch = {
            k: torch.as_tensor(np.stack([o[k] for o in pair]), device=self.device)
            .permute(0, 3, 1, 2)[None].float() / 255
            for k in CAMERAS
        }
        state = encode_state(np.stack([o["state"] for o in pair]), ROT6D)
        batch["state"] = torch.as_tensor(state, device=self.device)[None]
        return batch

    @torch.no_grad()
    def predict(self, observation):
        batch = self.batch(observation)
        normalized = self.model.normalizer.normalize(batch)
        flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in normalized.items()}
        features = self.model.obs_encoder(flat).reshape(1, -1)
        condition = torch.zeros((1, self.model.horizon, 7), device=self.device)
        sample = self.model.conditional_sample(
            condition, torch.zeros_like(condition, dtype=torch.bool),
            global_cond=features, generator=self.generator, **self.model.kwargs,
        )
        actions = self.model.normalizer["action"].unnormalize(sample)
        base = actions[:, self.model.n_obs_steps - 1].clamp(-1, 1)
        features = torch.cat((features, base), dim=-1)
        if not torch.isfinite(features).all():
            raise ValueError("Nonfinite DP features or action")
        return features[0].cpu().numpy(), base[0].cpu().numpy()
