"""Gymnasium registration for task1 DirectRLEnv (Kit-time import)."""

from __future__ import annotations

_REGISTERED = False
GYM_ID = "Embodied-Task1-DimmInsert-Direct-v0"


def register_task1_gym(force: bool = False) -> str:
    """Register Isaac Lab DirectRLEnv with gymnasium. Idempotent."""
    global _REGISTERED
    if _REGISTERED and not force:
        return GYM_ID
    import gymnasium as gym

    # Avoid duplicate registration errors
    if GYM_ID in gym.envs.registry:
        _REGISTERED = True
        return GYM_ID

    gym.register(
        id=GYM_ID,
        entry_point="tasks.task1_dimm_insert.env.direct_env:Task1DimmInsertEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "tasks.task1_dimm_insert.env.direct_env_cfg:Task1DimmInsertEnvCfg"
            ),
        },
    )
    _REGISTERED = True
    return GYM_ID


# ---- Plain PPO gym id (task completion only; parallel to Stage 1A) ---------
_PLAIN_REGISTERED = False
GYM_ID_PLAIN = "Embodied-Task1-DimmInsert-Plain-v0"


def register_task1_plain_gym(force: bool = False) -> str:
    """Register plain DirectRLEnv (no safety terminate/reward). Idempotent."""
    global _PLAIN_REGISTERED
    if _PLAIN_REGISTERED and not force:
        return GYM_ID_PLAIN
    import gymnasium as gym

    if GYM_ID_PLAIN in gym.envs.registry:
        _PLAIN_REGISTERED = True
        return GYM_ID_PLAIN

    gym.register(
        id=GYM_ID_PLAIN,
        entry_point="tasks.task1_dimm_insert.env.plain_direct_env:Task1PlainInsertEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                "tasks.task1_dimm_insert.env.plain_direct_env_cfg:Task1PlainInsertEnvCfg"
            ),
        },
    )
    _PLAIN_REGISTERED = True
    return GYM_ID_PLAIN
