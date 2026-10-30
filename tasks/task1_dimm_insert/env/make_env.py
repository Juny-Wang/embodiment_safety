"""Stage 1A env factory — low-dim state obs, no vision.

- backend=fake → FakeEnv (dry-run / --dry-run)
- backend=isaac_lab|isaac|lab → LabEnvHandle (stub without Kit; DirectRLEnv
  when Kit/pxr is up and try_direct succeeds — see lab_env.py)

Force / latch limits conceptually mirror scene/safety_config.json (Plan D demo
thresholds — NOT material limits). Do not import franka_sim/t1_rl.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

# ---- Space docs (dims match OBS_SPEC.md / REWARD_SPEC.md) -----------------

OBS_JOINT_POS = 7
OBS_JOINT_VEL = 7
OBS_GRIPPER = 1          # width [m] or normalized open fraction
OBS_DIMM_SOCKET_REL = 7  # xyz (3) + quat wxyz (4) of DIMM in socket frame
OBS_LATCH_ANGLES = 2     # left/right latch hinge [rad]
OBS_CONTACT_FORCES = 6   # axial, lateral, left/right latch press, finger L/R [N]
OBS_DIM = (
    OBS_JOINT_POS
    + OBS_JOINT_VEL
    + OBS_GRIPPER
    + OBS_DIMM_SOCKET_REL
    + OBS_LATCH_ANGLES
    + OBS_CONTACT_FORCES
)  # 30

# Action: EE delta pose (xyz + rpy) + gripper cmd — Stage 1A default
ACT_EE_DELTA = 6  # dx,dy,dz,droll,dpitch,dyaw
ACT_GRIPPER = 1
ACT_DIM = ACT_EE_DELTA + ACT_GRIPPER  # 7

SAFETY_LIMITS_REF = {
    "latch_overforce_N": 3.0,
    "grip_force_limit_N": 15.0,
    "axial_contact_N": 5.0,
    "lateral_contact_N": 2.0,
    "source": "tasks/task1_dimm_insert/scene/safety_config.json",
}


@dataclass
class SpaceSpec:
    """Minimal stand-in for gymnasium.spaces (no gym required for smoke)."""

    shape: Tuple[int, ...]
    dtype: str = "float32"
    low: Optional[float] = None
    high: Optional[float] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "low": self.low,
            "high": self.high,
            "description": self.description,
        }


@dataclass
class FakeEnv:
    """Dry-run env: reset/step return zeros; documents Stage 1A contract."""

    cfg: Mapping[str, Any] = field(default_factory=dict)
    scene_path: Optional[Path] = None
    observation_space: SpaceSpec = field(
        default_factory=lambda: SpaceSpec(
            shape=(OBS_DIM,),
            description=(
                "joint_pos(7)+joint_vel(7)+gripper(1)+"
                "dimm_socket_rel(7)+latch_angles(2)+contact_forces(6)"
            ),
        )
    )
    action_space: SpaceSpec = field(
        default_factory=lambda: SpaceSpec(
            shape=(ACT_DIM,),
            low=-1.0,
            high=1.0,
            description="ee_delta_xyz_rpy(6)+gripper(1), normalized",
        )
    )
    _step_count: int = 0
    _max_episode_steps: int = 500
    backend: str = "fake"

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._step_count = 0
        obs = [0.0] * OBS_DIM
        mode = str(self.cfg.get("reward_mode") or self.cfg.get("stage") or "stage1a").lower()
        info = {
            "backend": self.backend,
            "scene_path": str(self.scene_path) if self.scene_path else None,
            "reward_mode": "plain" if mode in ("plain", "plain_insert", "task1_plain_ppo") else "stage1a",
            "safety_limits_ref": dict(SAFETY_LIMITS_REF),
        }
        return obs, info

    def step(self, action: Sequence[float]):
        if self.backend != "fake":
            raise NotImplementedError(
                "Real Isaac Lab step not wired yet; use backend='fake' for smoke "
                "or implement Lab env under tasks/task1_dimm_insert/env/"
            )
        self._step_count += 1
        obs = [0.0] * OBS_DIM
        mode = str(self.cfg.get("reward_mode") or self.cfg.get("stage") or "stage1a").lower()
        if mode in ("plain", "plain_insert", "task1_plain_ppo"):
            reward_terms = {
                "approach": 0.0,
                "align": 0.0,
                "insert_depth": 0.0,
                "dual_latch": 0.0,
                "success_bonus": 0.0,
                # force_penalty intentionally zero for plain trainer
                "force_penalty": 0.0,
            }
            reward_mode = "plain"
        else:
            reward_terms = {
                "approach": 0.0,
                "align": 0.0,
                "insert_depth": 0.0,
                "dual_latch": 0.0,
                "force_penalty": 0.0,
            }
            reward_mode = "stage1a"
        reward = float(sum(reward_terms.values()))
        terminated = False
        truncated = self._step_count >= self._max_episode_steps
        info = {
            "reward_terms": reward_terms,
            "reward_mode": reward_mode,
            "step": self._step_count,
            "backend": self.backend,
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        return None


def _as_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    return Path(str(value))


EnvLike = Union["FakeEnv", Any]


def make_env(cfg: Optional[Mapping[str, Any]] = None) -> EnvLike:
    """Factory required by runners.train.

    cfg keys (optional):
      scene_path: path to tasks/.../scene/
      backend: 'fake' (default) | 'isaac_lab' (LabEnvHandle; Direct when Kit up)
      max_episode_steps: int
      num_envs, seed, device: forwarded to lab_env when backend is Lab
      try_direct: bool — attempt DirectRLEnv when Kit available (default: auto)
    """
    cfg = dict(cfg or {})
    backend = str(cfg.get("backend", "fake")).lower()
    scene_path = _as_path(cfg.get("scene_path"))
    max_steps = int(cfg.get("max_episode_steps", 500))

    if backend in ("isaac_lab", "isaac", "lab", "isaac_lab_direct"):
        from tasks.task1_dimm_insert.env.lab_env import make_lab_env

        return make_lab_env(cfg)

    cfg.setdefault("stage", cfg.get("stage", "stage1a"))
    if "reward_mode" not in cfg and str(cfg.get("stage", "")).lower() in (
        "plain", "plain_insert", "task1_plain_ppo",
    ):
        cfg["reward_mode"] = "plain"
    return FakeEnv(
        cfg=cfg,
        scene_path=scene_path,
        _max_episode_steps=max_steps,
        backend="fake",
    )
