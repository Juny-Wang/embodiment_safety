"""Task1 plain DirectRLEnv as a single-env SERL interface.

Import only AFTER Isaac Lab AppLauncher has started.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tasks.task1_dimm_insert.env.make_env import ACT_DIM, OBS_DIM
from tasks.task1_dimm_insert.env.plain_direct_env import Task1PlainInsertEnv
from tasks.task1_dimm_insert.env.plain_direct_env_cfg import Task1PlainInsertEnvCfg
from tasks.task1_dimm_insert.env.register_gym import register_task1_plain_gym

ES_ROOT = Path(__file__).resolve().parents[2]


def _scalar(value) -> float:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return float(np.reshape(value, -1)[0])


def flat_obs(obs) -> np.ndarray:
    if isinstance(obs, dict):
        obs = obs.get("policy", next(iter(obs.values())))
    if torch.is_tensor(obs):
        obs = obs.detach().cpu().numpy()
    array = np.asarray(obs, dtype=np.float32).reshape(-1)
    if array.shape != (OBS_DIM,):
        raise RuntimeError(f"expected {OBS_DIM}-dim policy observation, got {array.shape}")
    return array


class IsaacTask1Env:
    """num_envs=1 wrapper: 30-D state in, 7-D action out, plain success reward."""

    def __init__(
        self,
        device: str = "cuda:0",
        control_mode: str = "diff_ik",
        scene_path: str | None = None,
    ):
        register_task1_plain_gym()
        cfg = Task1PlainInsertEnvCfg()
        cfg.scene.num_envs = 1
        cfg.sim.device = device
        cfg.control_mode = str(control_mode)
        if scene_path:
            self.scene_path = Path(scene_path)
        else:
            self.scene_path = ES_ROOT / "tasks/task1_dimm_insert/scene"
        self.env = Task1PlainInsertEnv(cfg=cfg)
        self.device = self.env.device
        self.control_mode = control_mode
        if int(self.env.num_envs) != 1:
            raise RuntimeError(f"SERL wrapper expects 1 env, got {self.env.num_envs}")

    def reset(self):
        obs_dict, info = self.env.reset()
        return flat_obs(obs_dict), self._info(info)

    def step(self, action):
        action = np.array(np.asarray(action, dtype=np.float32).reshape(ACT_DIM), copy=True)
        action = np.clip(action, -1.0, 1.0)
        obs_dict, reward, terminated, truncated, info = self.env.step(
            torch.as_tensor(action, device=self.device).view(1, -1)
        )
        extras = {}
        if hasattr(self.env, "extras") and isinstance(self.env.extras, dict):
            extras = self.env.extras
        merged = dict(info) if isinstance(info, dict) else {}
        merged.update({k: v for k, v in extras.items() if k != "log"})
        return (
            flat_obs(obs_dict),
            _scalar(reward),
            bool(_scalar(terminated)),
            bool(_scalar(truncated)),
            self._info(merged),
        )

    def close(self):
        self.env.close()

    def _info(self, info) -> dict:
        out = {}
        if not isinstance(info, dict):
            return out
        for key, value in info.items():
            if key == "log":
                continue
            try:
                if torch.is_tensor(value):
                    arr = value.detach().cpu().numpy()
                    out[key] = float(arr.reshape(-1)[0]) if arr.size == 1 else arr
                elif isinstance(value, (bool, int, float, np.floating, np.integer)):
                    out[key] = float(value) if not isinstance(value, bool) else bool(value)
                elif isinstance(value, np.ndarray) and value.size == 1:
                    out[key] = float(value.reshape(-1)[0])
            except Exception:
                continue
        return out
