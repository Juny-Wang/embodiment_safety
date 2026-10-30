#!/usr/bin/env python3
"""Record successful Task1 DIMM inserts as SERL demo transitions (.pkl).

Uses Task1PlainInsertEnv (30-D obs, 7-D diff_ik action, plain reward) — same
backend as plain PPO / SERL train (Isaac Lab → Isaac Sim 4.5 via isaaclab.sh).

IMPORTANT: DirectRLEnv does NOT open_stage(manual_franka_scene.usda). Lab owns
/World/envs/env_0/{Robot,DIMM,Socket}. This recorder composes the confirmed
Task1 Floor+Table backdrop under /World/Task1Backdrop so the GUI is not an
empty grid. MatchedSocket full-asset overlays are NOT attached (they create
static-static PhysicsUSD joint errors on LeftHinge).

Scene authority (teleop / franka_manual_safety_v002 copy):
  tasks/task1_dimm_insert/scene/manual_franka_scene.usda

Isaac Sim 5.0 Script-Editor teleop of that USDA is a different entry (see
scene/README.md). SERL demos must stay on Lab/4.5 for 30-D obs compatibility.

Launch via run_record_demos.sh (GUI — not headless).
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

ES_ROOT = Path("/mnt/data2/wjy_SERL/Embodied_Safety")
DEFAULT_SCENE = ES_ROOT / "tasks/task1_dimm_insert/scene"
DEFAULT_SCENE_USDA = DEFAULT_SCENE / "manual_franka_scene.usda"
BACKDROP_USD = ES_ROOT / "tasks/task1_dimm_insert/env/assets/lab_task1_backdrop.usda"
TELEOP_PREVIEW_USD = Path("/mnt/data2/wjy_SERL/franka_sim/outputs/T1/riser_io_preview.usda")
# Reference only — do not attach; joints between static bodies break PhysX.
SOCKET_VISUAL_USD = (
    ES_ROOT / "tasks/task1_dimm_insert/env/assets/lab_socket_visual_only.usda"
)

parser = argparse.ArgumentParser(description="Record Task1 SERL demos via keyboard teleop")
parser.add_argument("--success-needed", type=int, default=10)
parser.add_argument("--max-attempts", type=int, default=80)
parser.add_argument("--control-mode", choices=("diff_ik", "joint"), default="diff_ik")
parser.add_argument(
    "--out-dir",
    type=str,
    default=str(ES_ROOT / "outputs/demos/task1_dimm_insert"),
)
parser.add_argument("--move-scale", type=float, default=1.0)
parser.add_argument("--rot-scale", type=float, default=0.7)
parser.add_argument(
    "--scene-path",
    type=str,
    default=str(DEFAULT_SCENE),
    help="Confirmed Task1 scene dir (manual_franka_scene.usda + safety_config.json)",
)
parser.add_argument(
    "--idle-action",
    action="store_true",
    help="If set, zero action when no keys held (default). Kept for clarity.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Force GUI for human teleop
if getattr(args_cli, "headless", False):
    print("[record_demos] WARNING: headless=True — keyboard may not work. Prefer GUI.", flush=True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path[:0] = [str(ES_ROOT), str(ES_ROOT / "algos/serl")]
import os

os.chdir(ES_ROOT)

import numpy as np
import torch

from isaaclab.envs import ViewerCfg

from tasks.task1_dimm_insert.env.make_env import ACT_DIM, OBS_DIM
from tasks.task1_dimm_insert.env.plain_direct_env import Task1PlainInsertEnv
from tasks.task1_dimm_insert.env.plain_direct_env_cfg import Task1PlainInsertEnvCfg
from tasks.task1_dimm_insert.env.register_gym import register_task1_plain_gym
from tasks.task1_dimm_insert.env.direct_env_cfg import (
    resolve_dimm_usd_path,
    resolve_franka_usd_path,
    resolve_socket_usd_path,
)
from keyboard_intervene import KeyboardIntervene


def flat_obs(obs) -> np.ndarray:
    if isinstance(obs, dict):
        obs = obs.get("policy", next(iter(obs.values())))
    if torch.is_tensor(obs):
        obs = obs.detach().cpu().numpy()
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    if arr.shape != (OBS_DIM,):
        raise RuntimeError(f"obs dim {arr.shape} != {OBS_DIM}")
    return arr


def scalar(x) -> float:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return float(np.reshape(x, -1)[0])


def is_success(env, terminated: bool, info: dict) -> bool:
    if isinstance(info, dict) and "success" in info:
        try:
            return bool(scalar(info["success"]))
        except Exception:
            pass
    extras = getattr(env, "extras", {}) or {}
    if "success" in extras:
        try:
            return bool(scalar(extras["success"]))
        except Exception:
            pass
    return bool(terminated)


def _assert_task1_prims() -> dict:
    """Require Lab Robot+DIMM+Socket under env_0 AND Task1 table backdrop."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    required = {
        "robot": "/World/envs/env_0/Robot",
        "dimm": "/World/envs/env_0/DIMM",
        "socket": "/World/envs/env_0/Socket",
        "backdrop_table": "/World/Task1Backdrop/Table",
        "backdrop_floor": "/World/Task1Backdrop/Floor",
    }
    found = {}
    missing = []
    for key, path in required.items():
        prim = stage.GetPrimAtPath(path)
        ok = bool(prim) and prim.IsValid()
        found[key] = {"path": path, "valid": ok}
        if not ok:
            missing.append(path)
    world = stage.GetPrimAtPath("/World")
    children = []
    if world and world.IsValid():
        children = [str(c.GetPath()) for c in world.GetChildren()]
    found["world_children"] = children
    if missing:
        raise RuntimeError(
            "Task1 demo scene missing prims: "
            + ", ".join(missing)
            + f" | /World children={children}. "
            "Lab must spawn Franka+DIMM+Socket and compose Task1Backdrop "
            f"(table). scene_usda={DEFAULT_SCENE_USDA}"
        )
    return found


def _strip_physics(prim_root) -> None:
    """Disable collision/rigid and deactivate joints under a visual backdrop."""
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    for p in Usd.PrimRange(prim_root):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            try:
                UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
            except Exception:
                pass
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            try:
                UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
            except Exception:
                pass
        tname = str(p.GetTypeName())
        if "Joint" in tname:
            try:
                p.SetActive(False)
            except Exception:
                pass


def _compose_task1_backdrop() -> dict:
    """Compose confirmed Task1 Floor+Table under /World/Task1Backdrop (visual).

    Root cause of empty-grid GUI: Task1PlainInsertEnv never open_stage()s
    manual_franka_scene.usda (lab_env.py documents teleop USDA is NOT cloned).
    """
    import omni.usd
    from pxr import UsdGeom, Gf

    stage = omni.usd.get_context().get_stage()
    root = "/World/Task1Backdrop"
    info = {
        "backdrop_usd": str(BACKDROP_USD),
        "teleop_preview": str(TELEOP_PREVIEW_USD),
        "path": root,
        "ok": False,
        "method": None,
    }
    if not BACKDROP_USD.is_file():
        raise FileNotFoundError(f"Missing Task1 backdrop USDA: {BACKDROP_USD}")

    existing = stage.GetPrimAtPath(root)
    if existing and existing.IsValid() and stage.GetPrimAtPath(f"{root}/Table").IsValid():
        info["ok"] = True
        info["method"] = "already_present"
        return info

    xform = UsdGeom.Xform.Define(stage, root)
    prim = xform.GetPrim()
    # Clear stale refs then add backdrop
    try:
        prim.GetReferences().ClearReferences()
    except Exception:
        pass
    prim.GetReferences().AddReference(str(BACKDROP_USD), "/Backdrop")
    info["method"] = "reference_lab_task1_backdrop"
    _strip_physics(prim)

    table = stage.GetPrimAtPath(f"{root}/Table")
    if not (table and table.IsValid()):
        if not TELEOP_PREVIEW_USD.is_file():
            raise RuntimeError(f"Backdrop Table missing after reference: {BACKDROP_USD}")
        # Clear failed backdrop ref, then pull teleop Table+Floor with Lab Z offset.
        try:
            prim.GetReferences().ClearReferences()
        except Exception:
            pass
        alias = UsdGeom.Xform.Define(stage, f"{root}/Table")
        alias.GetPrim().GetReferences().AddReference(
            str(TELEOP_PREVIEW_USD), "/World/Table"
        )
        alias.ClearXformOpOrder()
        alias.AddTranslateOp().Set(Gf.Vec3d(0.25, 0.0, -0.75))
        _strip_physics(alias.GetPrim())
        floor = UsdGeom.Xform.Define(stage, f"{root}/Floor")
        floor.GetPrim().GetReferences().AddReference(
            str(TELEOP_PREVIEW_USD), "/World/Floor"
        )
        floor.ClearXformOpOrder()
        floor.AddTranslateOp().Set(Gf.Vec3d(0.25, 0.0, -0.75))
        _strip_physics(floor.GetPrim())
        info["method"] = "fallback_teleop_preview_table"

    info["ok"] = True
    print(
        f"[record_demos] composed Task1 backdrop at {root} via {info['method']}",
        flush=True,
    )
    return info


def _disable_matched_socket_visual_if_present() -> None:
    """Deactivate prior SocketVisualMatched overlays (static-static joint errors)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    for path in (
        "/World/envs/env_0/SocketVisualMatched",
        "/World/SocketVisualMatched",
    ):
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            try:
                prim.SetActive(False)
                print(f"[record_demos] deactivated broken overlay {path}", flush=True)
            except Exception as exc:
                print(f"[record_demos] WARN: could not deactivate {path}: {exc!r}", flush=True)


def _frame_viewport_to_task1() -> None:
    """Nudge Kit perspective camera toward socket/DIMM workspace (env_0)."""
    try:
        import omni.kit.viewport.utility as vp_util

        viewport = vp_util.get_active_viewport()
        if viewport is None:
            return
        try:
            vp_util.frame_viewport_to_selection(viewport)
        except Exception:
            pass
    except Exception as exc:
        print(f"[record_demos] viewport frame skip: {exc!r}", flush=True)


def main():
    scene_path = Path(args_cli.scene_path).resolve()
    scene_usda = scene_path / "manual_franka_scene.usda"
    if not scene_usda.is_file():
        raise FileNotFoundError(
            f"Confirmed Task1 scene USDA missing: {scene_usda}. "
            "Expected Embodied_Safety copy of franka_manual_safety_v002."
        )
    if not BACKDROP_USD.is_file():
        raise FileNotFoundError(
            f"Task1 Lab backdrop missing: {BACKDROP_USD}. "
            "Expected Floor+Table visual for demo GUI."
        )

    register_task1_plain_gym()
    cfg = Task1PlainInsertEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args_cli.device
    cfg.control_mode = str(args_cli.control_mode)
    # Safety JSON from confirmed Task1 scene pack (USDA itself is NOT open_stage'd)
    cfg._scene_path_override = str(scene_path)  # type: ignore[attr-defined]
    # Frame GUI on env_0 workspace; ALWAYS env_index=0 with num_envs=1
    cfg.viewer = ViewerCfg(
        eye=(1.05, 0.85, 0.70),
        lookat=(0.50, 0.0, 0.24),
        origin_type="env",
        env_index=0,
    )

    franka_usd = resolve_franka_usd_path()
    dimm_usd = resolve_dimm_usd_path()
    socket_usd = resolve_socket_usd_path()
    print(
        "RECORD_DEMOS_SCENE",
        json.dumps(
            {
                "scene_path": str(scene_path),
                "scene_usda_authority": str(scene_usda),
                "franka_usd": franka_usd,
                "dimm_usd": dimm_usd,
                "socket_usd": socket_usd,
                "backdrop_usd": str(BACKDROP_USD),
                "socket_visual_usd_disabled": str(SOCKET_VISUAL_USD),
                "kit_note": (
                    "IsaacLab->_isaac_sim 4.5.0 (same as plain PPO). "
                    "Teleop USDA opens in Isaac Sim 5.0 separately (scene/README.md)."
                ),
                "compose_note": (
                    "Lab does not open_stage(manual_franka_scene); "
                    "Task1Backdrop supplies Floor+Table from confirmed scene look."
                ),
                "viewer": {
                    "eye": list(cfg.viewer.eye),
                    "lookat": list(cfg.viewer.lookat),
                    "origin_type": cfg.viewer.origin_type,
                    "env_index": cfg.viewer.env_index,
                },
                "tag": "TASK1_SERL_RECORD_DEMOS_TABLE_BACKDROP_20261026",
            }
        ),
        flush=True,
    )

    env = Task1PlainInsertEnv(cfg=cfg, render_mode="human")
    for _ in range(20):
        simulation_app.update()

    backdrop_info = _compose_task1_backdrop()
    for _ in range(10):
        simulation_app.update()
    _disable_matched_socket_visual_if_present()
    prim_info = _assert_task1_prims()
    print("RECORD_DEMOS_PRIMS_OK", json.dumps(prim_info), flush=True)
    print("RECORD_DEMOS_BACKDROP_OK", json.dumps(backdrop_info), flush=True)
    for _ in range(5):
        simulation_app.update()
    _frame_viewport_to_task1()
    print(
        "[record_demos] GUI check: Stage should list /World/Task1Backdrop "
        "(Table+Floor) and /World/envs/env_0. Keep IsaacLab Environment Index=0. "
        "Kit title bar will say Isaac Sim 4.5.0 (IsaacLab; same as plain PPO).",
        flush=True,
    )

    kb = KeyboardIntervene(move_scale=args_cli.move_scale, rot_scale=args_cli.rot_scale)
    kb.enable()

    out_dir = Path(args_cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = out_dir / f"serl_{args_cli.success_needed}_demos_{stamp}.pkl"
    meta_path = out_dir / f"serl_{args_cli.success_needed}_demos_{stamp}.json"

    print(
        "RECORD_DEMOS_START",
        json.dumps(
            {
                "success_needed": args_cli.success_needed,
                "control_mode": args_cli.control_mode,
                "obs_dim": OBS_DIM,
                "act_dim": ACT_DIM,
                "out_path": str(out_path),
                "scene_usda_authority": str(scene_usda),
                "backdrop_usd": str(BACKDROP_USD),
                "tag": "TASK1_SERL_RECORD_DEMOS_TABLE_BACKDROP_20261026",
            }
        ),
        flush=True,
    )
    print(
        "Controls: Arrow keys / PgUp / PgDn translate; hold Shift to rotate; "
        "[ or O open gripper; ] or P close; R reset episode. "
        "Click the Isaac viewport so it has keyboard focus. "
        "IsaacLab Environment Index must stay 0 (num_envs=1).",
        flush=True,
    )

    transitions: list = []
    success_count = 0
    attempt = 0
    obs_dict, info = env.reset()
    obs = flat_obs(obs_dict)
    ep_transitions: list = []
    ep_ret = 0.0
    ep_len = 0

    while success_count < args_cli.success_needed and attempt < args_cli.max_attempts:
        if not simulation_app.is_running():
            break
        simulation_app.update()

        if kb.consume_reset():
            print(f"[record] manual reset (discard episode len={ep_len})", flush=True)
            ep_transitions.clear()
            ep_ret = 0.0
            ep_len = 0
            obs_dict, info = env.reset()
            obs = flat_obs(obs_dict)
            continue

        action = kb.action()
        next_obs_dict, reward, terminated, truncated, step_info = env.step(
            torch.as_tensor(action, device=env.device).view(1, -1)
        )
        next_obs = flat_obs(next_obs_dict)
        rew = scalar(reward)
        term = bool(scalar(terminated))
        trunc = bool(scalar(truncated))
        done = term or trunc
        if kb.consume_skip():
            trunc = True
            done = True

        extras = getattr(env, "extras", {}) or {}
        merged = dict(step_info) if isinstance(step_info, dict) else {}
        merged.update({k: v for k, v in extras.items() if k != "log"})

        tr = {
            "observations": obs.copy(),
            "actions": action.copy(),
            "next_observations": next_obs.copy(),
            "rewards": np.float32(rew),
            "masks": np.float32(1.0 - float(term)),
            "dones": bool(done),
        }
        ep_transitions.append(copy.deepcopy(tr))
        ep_ret += rew
        ep_len += 1
        obs = next_obs

        if done:
            attempt += 1
            ok = is_success(env, term, merged)
            if ok:
                transitions.extend(ep_transitions)
                success_count += 1
                print(
                    f"[record] SUCCESS {success_count}/{args_cli.success_needed} "
                    f"attempt={attempt} len={ep_len} ret={ep_ret:.2f} "
                    f"transitions_total={len(transitions)}",
                    flush=True,
                )
            else:
                print(
                    f"[record] fail/timeout attempt={attempt} len={ep_len} "
                    f"ret={ep_ret:.2f} term={term} trunc={trunc} — discarded",
                    flush=True,
                )
            ep_transitions = []
            ep_ret = 0.0
            ep_len = 0
            obs_dict, info = env.reset()
            obs = flat_obs(obs_dict)

    kb.disable()
    env.close()

    if success_count < args_cli.success_needed:
        print(
            f"[record] STOPPED early: got {success_count}/{args_cli.success_needed} "
            f"after {attempt} attempts. Writing partial pickle.",
            flush=True,
        )

    if transitions:
        sample = transitions[0]
        assert sample["observations"].shape == (OBS_DIM,)
        assert sample["actions"].shape == (ACT_DIM,)
        assert sample["next_observations"].shape == (OBS_DIM,)

    with out_path.open("wb") as handle:
        pickle.dump(transitions, handle)

    meta = {
        "success_count": success_count,
        "success_needed": args_cli.success_needed,
        "attempts": attempt,
        "num_transitions": len(transitions),
        "obs_dim": OBS_DIM,
        "act_dim": ACT_DIM,
        "control_mode": args_cli.control_mode,
        "out_path": str(out_path),
        "reward": "Task1PlainInsertEnv._get_rewards",
        "scene_path": str(scene_path),
        "scene_usda_authority": str(scene_usda),
        "backdrop_usd": str(BACKDROP_USD),
        "franka_usd": franka_usd,
        "dimm_usd": dimm_usd,
        "socket_usd": socket_usd,
        "keys": ["observations", "actions", "next_observations", "rewards", "masks", "dones"],
        "created": stamp,
        "tag": "TASK1_SERL_RECORD_DEMOS_TABLE_BACKDROP_20261026",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print("RECORD_DEMOS_EXIT:0", json.dumps(meta), flush=True)
    return 0 if success_count > 0 else 2


if __name__ == "__main__":
    code = 0
    try:
        code = main() or 0
    except Exception as exc:
        print("RECORD_DEMOS_FAIL", repr(exc), flush=True)
        code = 1
        raise
    finally:
        simulation_app.close()
    raise SystemExit(code)
