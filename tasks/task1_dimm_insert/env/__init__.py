"""Env package: obs/act/reward contracts + make_env factory + Lab stub/direct."""

from __future__ import annotations

from tasks.task1_dimm_insert.env.make_env import (
    ACT_DIM,
    OBS_DIM,
    FakeEnv,
    make_env,
)

__all__ = ["FakeEnv", "make_env", "OBS_DIM", "ACT_DIM"]

try:
    from tasks.task1_dimm_insert.env.lab_env import (  # noqa: F401
        LabEnvHandle,
        make_lab_env,
        probe_lab_stack,
    )

    __all__ += ["LabEnvHandle", "make_lab_env", "probe_lab_stack"]
except Exception:
    pass

# Gym id constant (registration happens in Kit via register_gym / kit_smoke)
try:
    from tasks.task1_dimm_insert.env.register_gym import GYM_ID  # noqa: F401

    __all__ += ["GYM_ID"]
except Exception:
    pass
