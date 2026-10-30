"""Isaac Lab–facing env adapter for Stage 1A (OBS 30 / ACT 7).

Design
------
- FakeEnv (make_env.py): gym-like single env for dry-run smoke — no Isaac.
- LabEnvHandle / make_lab_env (this file): adapter toward Isaac Lab DirectRLEnv
  + rsl_rl VecEnv. Import-guarded; does not launch Kit on import.
- FakeVecEnv: minimal rsl_rl.env.VecEnv stand-in so OnPolicyRunner can be
  *constructed* without SimulationApp (wiring smoke only).
- try_make_direct_vec_env: when Kit/pxr is up, builds Task1DimmInsertEnv
  (Franka + cuboid DIMM/socket placeholders) + RslRlVecEnvWrapper.

Blockers for real learn() outside Kit
-------------------------------------
1. Isaac Sim must provide ``pxr`` / Kit runtime (conda env_isaaclab alone is
   not enough — need AppLauncher / isaaclab.sh / kit python).
2. Teleop ``manual_franka_scene.usda`` is NOT cloned; DirectRLEnv uses Lab
   asset cfgs + cuboid placeholders (see direct_env_cfg.py asset_notes).
3. Latch DOFs / contact sensors still placeholders in OBS slots 23..29.

This module never imports franka_sim/t1_rl.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from tasks.task1_dimm_insert.env.make_env import (
    ACT_DIM,
    OBS_DIM,
    SAFETY_LIMITS_REF,
)

DEFAULT_SCENE_USDA = "manual_franka_scene.usda"
DEFAULT_SAFETY_JSON = "safety_config.json"


def probe_lab_stack() -> Dict[str, Any]:
    """Report what is importable in the current interpreter (no Kit launch)."""
    out: Dict[str, Any] = {
        "rsl_rl": None,
        "OnPolicyRunner": None,
        "isaaclab": None,
        "DirectRLEnv": None,
        "pxr": None,
        "torch": None,
        "tensordict": None,
        "Task1DimmInsertEnv": None,
        "errors": {},
    }
    try:
        import rsl_rl  # type: ignore

        out["rsl_rl"] = getattr(rsl_rl, "__file__", True)
        from rsl_rl.runners import OnPolicyRunner  # type: ignore

        out["OnPolicyRunner"] = str(OnPolicyRunner)
    except Exception as e:  # noqa: BLE001
        out["errors"]["rsl_rl"] = f"{type(e).__name__}: {e}"

    try:
        import isaaclab  # type: ignore

        out["isaaclab"] = getattr(isaaclab, "__file__", True)
    except Exception as e:  # noqa: BLE001
        out["errors"]["isaaclab"] = f"{type(e).__name__}: {e}"

    try:
        from isaaclab.envs import DirectRLEnv  # type: ignore

        out["DirectRLEnv"] = str(DirectRLEnv)
    except Exception as e:  # noqa: BLE001
        out["errors"]["DirectRLEnv"] = f"{type(e).__name__}: {e}"

    try:
        import pxr  # type: ignore

        out["pxr"] = getattr(pxr, "__file__", True)
    except Exception as e:  # noqa: BLE001
        out["errors"]["pxr"] = f"{type(e).__name__}: {e}"

    try:
        import torch  # type: ignore

        out["torch"] = getattr(torch, "__version__", True)
    except Exception as e:  # noqa: BLE001
        out["errors"]["torch"] = f"{type(e).__name__}: {e}"

    try:
        import tensordict  # type: ignore

        out["tensordict"] = getattr(tensordict, "__version__", True)
    except Exception as e:  # noqa: BLE001
        out["errors"]["tensordict"] = f"{type(e).__name__}: {e}"

    # Only resolvable under Kit (imports isaaclab.sim → pxr)
    if out.get("pxr"):
        try:
            from tasks.task1_dimm_insert.env.direct_env import Task1DimmInsertEnv  # type: ignore

            out["Task1DimmInsertEnv"] = str(Task1DimmInsertEnv)
        except Exception as e:  # noqa: BLE001
            out["errors"]["Task1DimmInsertEnv"] = f"{type(e).__name__}: {e}"

    return out


def load_safety_limits(scene_path: Optional[Path]) -> Dict[str, Any]:
    """Merge SAFETY_LIMITS_REF with scene/safety_config.json when present."""
    limits = dict(SAFETY_LIMITS_REF)
    if scene_path is None:
        return limits
    cfg_path = Path(scene_path) / DEFAULT_SAFETY_JSON
    if not cfg_path.is_file():
        return limits
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return limits
    plan_d = data.get("plan_d") or {}
    if isinstance(plan_d, dict):
        if "latch_overforce_N" in plan_d:
            limits["latch_overforce_N"] = float(plan_d["latch_overforce_N"])
        if "grip_force_limit_N" in plan_d:
            limits["grip_force_limit_N"] = float(plan_d["grip_force_limit_N"])
    rules = data.get("rules")
    if isinstance(rules, dict):
        for key, dest in (
            ("axial_contact_N", "axial_contact_N"),
            ("lateral_contact_N", "lateral_contact_N"),
        ):
            val = rules.get(key)
            if isinstance(val, (int, float)):
                limits[dest] = float(val)
            elif isinstance(val, dict) and "limit" in val:
                limits[dest] = float(val["limit"])
    elif isinstance(rules, list):
        for item in rules:
            if not isinstance(item, dict):
                continue
            name = str(item.get("id") or item.get("name") or item.get("key") or "")
            lim = item.get("limit", item.get("threshold_N", item.get("value")))
            if lim is None:
                continue
            if "axial" in name.lower():
                limits["axial_contact_N"] = float(lim)
            elif "lateral" in name.lower():
                limits["lateral_contact_N"] = float(lim)
            elif "latch_overforce" in name.lower() or name.lower() == "latch_overforce_N":
                limits["latch_overforce_N"] = float(lim)
            elif "grip" in name.lower():
                limits["grip_force_limit_N"] = float(lim)
    limits["source"] = str(cfg_path)
    return limits


@dataclass
class LabEnvCfg:
    """Config for DirectRLEnv / adapter (Stage 1A)."""

    scene_path: Path
    scene_usda: str = DEFAULT_SCENE_USDA
    num_envs: int = 64
    seed: int = 42
    device: str = "cuda:0"
    max_episode_steps: int = 500
    obs_dim: int = OBS_DIM
    act_dim: int = ACT_DIM
    usd_path: Optional[Path] = None
    safety_limits: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def resolved_usd(self) -> Path:
        if self.usd_path is not None:
            return Path(self.usd_path)
        return Path(self.scene_path) / self.scene_usda

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_path": str(self.scene_path),
            "scene_usda": self.scene_usda,
            "usd_path": str(self.resolved_usd()),
            "num_envs": self.num_envs,
            "seed": self.seed,
            "device": self.device,
            "max_episode_steps": self.max_episode_steps,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "safety_limits": dict(self.safety_limits),
            "extra": dict(self.extra),
        }


def build_lab_env_cfg(cfg: Mapping[str, Any]) -> LabEnvCfg:
    scene = Path(str(cfg.get("scene_path") or ".")).resolve()
    lab = LabEnvCfg(
        scene_path=scene,
        scene_usda=str(cfg.get("scene_usda", DEFAULT_SCENE_USDA)),
        num_envs=int(cfg.get("num_envs", 64)),
        seed=int(cfg.get("seed", 42)),
        device=str(cfg.get("device", "cuda:0")),
        max_episode_steps=int(cfg.get("max_episode_steps", 500)),
        usd_path=Path(cfg["usd_path"]).resolve() if cfg.get("usd_path") else None,
        safety_limits=load_safety_limits(scene),
        extra={k: v for k, v in cfg.items() if k not in {
            "scene_path", "scene_usda", "num_envs", "seed", "device",
            "max_episode_steps", "usd_path", "backend", "try_direct",
        }},
    )
    return lab


@dataclass
class LabEnvHandle:
    """Handle returned by make_lab_env.

    ``backend``:
      - isaac_lab_stub: no VecEnv (plain conda / no Kit)
      - isaac_lab_direct: Task1DimmInsertEnv wrapped for rsl_rl
    """

    cfg: LabEnvCfg
    backend: str = "isaac_lab_stub"
    vec_env: Any = None
    blocker: str = ""
    probe: Dict[str, Any] = field(default_factory=dict)

    @property
    def observation_dim(self) -> int:
        return self.cfg.obs_dim

    @property
    def action_dim(self) -> int:
        return self.cfg.act_dim

    def get_observations(self):  # noqa: ANN201
        if self.vec_env is not None:
            return self.vec_env.get_observations()
        raise NotImplementedError(self.blocker or "Lab env not built")

    def step(self, actions):  # noqa: ANN001, ANN201
        if self.vec_env is not None:
            return self.vec_env.step(actions)
        raise NotImplementedError(self.blocker or "Lab env not built")

    def close(self) -> None:
        if self.vec_env is not None and hasattr(self.vec_env, "close"):
            self.vec_env.close()
        return None


def _blocker_message(probe: Dict[str, Any], usd: Path) -> str:
    parts = [
        "Isaac Lab DirectRLEnv needs Kit (pxr). Without Kit this is a stub.",
        f"Teleop USD (reference only, not cloned): {usd} exists={usd.is_file()}.",
        "DirectRLEnv scene = Franka + cuboid DIMM/socket placeholders "
        "(see tasks/.../env/direct_env_cfg.py).",
        "Stack probe (current interpreter):",
    ]
    for k in (
        "rsl_rl", "OnPolicyRunner", "isaaclab", "DirectRLEnv", "pxr",
        "torch", "Task1DimmInsertEnv",
    ):
        parts.append(f"  {k}={probe.get(k)!r}")
    if probe.get("errors"):
        parts.append("Import errors:")
        for name, err in probe["errors"].items():
            parts.append(f"  {name}: {err}")
    parts.append(
        "Launch Kit smoke: see WIRING.md § Kit short smoke "
        "(runners/kit_smoke.py via isaaclab.sh -p … --headless)."
    )
    return "\n".join(parts)


def kit_runtime_available() -> bool:
    """True when Isaac Sim App / pxr is importable (Kit process)."""
    try:
        import pxr  # noqa: F401
    except Exception:
        return False
    try:
        import omni.kit.app  # type: ignore

        app = omni.kit.app.get_app()
        return app is not None
    except Exception:
        return True


def try_make_direct_vec_env(cfg: Mapping[str, Any]):
    """Build Task1DimmInsertEnv + RslRlVecEnvWrapper when Kit is up.

    Returns (vec_env, env_cfg, error_str). vec_env is None on failure.
    """
    if not kit_runtime_available():
        return None, None, "Kit/pxr not available in this interpreter"

    try:
        from tasks.task1_dimm_insert.env.direct_env_cfg import (
            Task1DimmInsertEnvCfg,
            resolve_franka_usd_path,
        )
        from tasks.task1_dimm_insert.env.direct_env import Task1DimmInsertEnv
        from tasks.task1_dimm_insert.env.register_gym import register_task1_gym
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    except Exception as e:  # noqa: BLE001
        return None, None, f"import DirectRLEnv stack failed: {type(e).__name__}: {e}"

    try:
        register_task1_gym()
        env_cfg = Task1DimmInsertEnvCfg()
        num_envs = int(cfg.get("num_envs", 4))
        env_cfg.scene.num_envs = num_envs
        if cfg.get("device"):
            env_cfg.sim.device = str(cfg["device"])
        scene_path = cfg.get("scene_path")
        if scene_path:
            limits = load_safety_limits(Path(str(scene_path)))
            for k in (
                "latch_overforce_N",
                "grip_force_limit_N",
                "axial_contact_N",
                "lateral_contact_N",
            ):
                if k in limits:
                    setattr(env_cfg, k, float(limits[k]))
            env_cfg._scene_path_override = str(scene_path)  # type: ignore[attr-defined]
        env_cfg.franka_usd_resolved = resolve_franka_usd_path()
        env = Task1DimmInsertEnv(cfg=env_cfg)
        vec = RslRlVecEnvWrapper(env)
        return vec, env_cfg, ""
    except Exception as e:  # noqa: BLE001
        return None, None, f"DirectRLEnv construct failed: {type(e).__name__}: {e}"


def make_lab_env(cfg: Optional[Mapping[str, Any]] = None) -> LabEnvHandle:
    """Build Lab-facing handle. Does not start SimulationApp.

    Without Kit → stub. With Kit and try_direct → Task1DimmInsertEnv wrap.
    """
    cfg = dict(cfg or {})
    lab_cfg = build_lab_env_cfg(cfg)
    probe = probe_lab_stack()
    usd = lab_cfg.resolved_usd()
    blocker = _blocker_message(probe, usd)

    try_direct = bool(cfg.get("try_direct", kit_runtime_available()))
    if try_direct and kit_runtime_available():
        vec, env_cfg, err = try_make_direct_vec_env(cfg)
        if vec is not None:
            if env_cfg is not None:
                lab_cfg.num_envs = int(env_cfg.scene.num_envs)
                lab_cfg.extra["franka_usd"] = getattr(env_cfg, "franka_usd_resolved", "")
                lab_cfg.extra["asset_notes"] = getattr(env_cfg, "asset_notes", "")
            return LabEnvHandle(
                cfg=lab_cfg,
                backend="isaac_lab_direct",
                vec_env=vec,
                blocker="",
                probe=probe,
            )
        blocker = f"DirectRLEnv attempted but failed.\n  error: {err}\n" + blocker

    return LabEnvHandle(cfg=lab_cfg, probe=probe, blocker=blocker)


def try_make_fake_vec_env(cfg: Mapping[str, Any]):
    """Return a FakeVecEnv if torch+tensordict+rsl_rl are importable, else None."""
    try:
        import torch
        from tensordict import TensorDict
        from rsl_rl.env import VecEnv
    except Exception:
        return None

    lab_cfg = build_lab_env_cfg(cfg)
    num_envs = int(cfg.get("num_envs", lab_cfg.num_envs))
    device = str(cfg.get("fake_vec_device", cfg.get("device", "cpu")))

    class FakeVecEnv(VecEnv):
        """Zeros obs; satisfies OnPolicyRunner.__init__ / get_observations."""

        def __init__(self) -> None:
            self.num_envs = num_envs
            self.num_actions = ACT_DIM
            self.max_episode_length = lab_cfg.max_episode_steps
            self.device = device
            self.cfg = lab_cfg.to_dict()
            self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
            self._obs = TensorDict(
                {"policy": torch.zeros(num_envs, OBS_DIM, device=device)},
                batch_size=[num_envs],
                device=device,
            )

        def get_observations(self) -> TensorDict:
            return self._obs

        def step(self, actions: torch.Tensor):
            rewards = torch.zeros(self.num_envs, device=self.device)
            dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            extras: Dict[str, Any] = {"observations": self._obs}
            return self._obs, rewards, dones, extras

        def close(self) -> None:
            return None

        def seed(self, seed: int = -1) -> int:  # noqa: A003
            return int(seed)

    return FakeVecEnv()
