"""Task1 plain DirectRLEnv — DIMM insert success only (no safety constraints).

Subclass of Stage 1A ``Task1DimmInsertEnv``. Overrides reward and termination
only; scene, control, and OBS (30) stay the same. Contact forces may remain
in the observation vector but do **not** contribute to reward or episode fail.
"""

from __future__ import annotations

import torch

from tasks.task1_dimm_insert.env.direct_env import Task1DimmInsertEnv
from tasks.task1_dimm_insert.env.plain_direct_env_cfg import Task1PlainInsertEnvCfg


class Task1PlainInsertEnv(Task1DimmInsertEnv):
    """Plain PPO env: terminate/reward on insert success (+ light shaping)."""

    cfg: Task1PlainInsertEnvCfg

    def _success_mask(self) -> torch.Tensor:
        """Geometric success: seated depth + XY + both latches closed."""
        self._refresh_intermediates()
        seated = self._insert_depth >= self.cfg.success_depth_m
        xy_ok = torch.linalg.norm(self._rel_pos[:, 0:2], dim=-1) < self.cfg.xy_tol_m * 2.0
        latched = (self._latch_angles.abs() < 0.05).all(dim=-1)
        return seated & xy_ok & latched

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        success = self._success_mask()
        # Intentionally ignore overforce / latch_overforce / grip limits.
        terminated = success
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        self.extras["success"] = success
        self.extras["overforce_ignored"] = True
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        """Task-completion shaping + success bonus; **no** force_penalty term."""
        self._refresh_intermediates()
        dist = torch.linalg.norm(self._rel_pos, dim=-1)
        approach = -dist

        xy_err = torch.linalg.norm(self._rel_pos[:, 0:2], dim=-1)
        w = self._rel_quat[:, 0].abs().clamp(0.0, 1.0)
        tilt = 2.0 * torch.acos(w)

        xy_scale = max(float(self.cfg.align_xy_scale_m), 1e-6)
        tilt_scale = max(float(self.cfg.align_tilt_scale_rad), 1e-6)
        align_raw = -(xy_err / xy_scale) - (tilt / tilt_scale)
        align = align_raw.clamp(
            min=float(self.cfg.align_clip_min),
            max=float(self.cfg.align_clip_max),
        )

        near = (xy_err < float(self.cfg.near_align_xy_m)) & (
            tilt < float(self.cfg.near_align_tilt_rad)
        )
        near_f = near.to(dtype=torch.float32)
        insert_depth = self._insert_depth * near_f
        dual_latch = (
            (self._latch_angles.abs() < 0.05).to(dtype=torch.float32).sum(dim=-1) * near_f
        )

        success = self._success_mask().to(dtype=torch.float32)
        w_success = float(getattr(self.cfg, "w_success", 20.0))

        reward = (
            self.cfg.w_approach * approach
            + self.cfg.w_align * align
            + self.cfg.w_insert_depth * insert_depth
            + self.cfg.w_dual_latch * dual_latch
            + w_success * success
        )
        # Explicit: force_penalty always zero in this trainer.
        force_penalty = torch.zeros_like(reward)

        self.extras["log"] = {
            "approach": (self.cfg.w_approach * approach).mean(),
            "align": (self.cfg.w_align * align).mean(),
            "insert_depth": (self.cfg.w_insert_depth * insert_depth).mean(),
            "dual_latch": (self.cfg.w_dual_latch * dual_latch).mean(),
            "success_bonus": (w_success * success).mean(),
            "force_penalty": force_penalty.mean(),
            "reward": reward.mean(),
            "xy_err": xy_err.mean(),
            "tilt": tilt.mean(),
            "align_raw": align_raw.mean(),
            "near_align_frac": near_f.mean(),
            "success_frac": success.mean(),
        }
        self.extras["reward_terms"] = {
            "approach": approach,
            "align": align,
            "insert_depth": insert_depth,
            "dual_latch": dual_latch,
            "success_bonus": success * w_success,
            "force_penalty": force_penalty,
            "xy_err": xy_err,
            "tilt": tilt,
            "near_align": near,
        }
        self.extras["reward_mode"] = "plain"
        return reward
