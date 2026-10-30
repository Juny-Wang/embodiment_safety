#!/usr/bin/env python3
"""Kit-headless short smoke for Stage 1A DirectRLEnv + RSL-RL.

MUST be launched via Isaac Lab isaaclab.sh so ``pxr`` exists.

--no-learn: construct + timed reset/step only (single-step timeout test).
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="Embodied_Safety Stage 1A Kit smoke")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--max_iterations", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--do-learn", action="store_true", default=True)
parser.add_argument("--no-learn", action="store_true", help="Construct + timed step only; skip learn()")
parser.add_argument("--num-steps", type=int, default=3, help="Env steps to run in --no-learn mode")
parser.add_argument("--step-timeout-s", type=float, default=60.0, help="Warn if a single step exceeds this (soft)")
parser.add_argument(
    "--control-mode",
    type=str,
    default="joint",
    choices=("joint", "diff_ik"),
    help="Action mapping: joint (default) or DifferentialIK",
)
parser.add_argument(
    "--scene_path",
    type=str,
    default=str(ROOT / "tasks/task1_dimm_insert/scene"),
)
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.no_learn:
    args_cli.do_learn = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

os.chdir(ROOT)

from tasks.task1_dimm_insert.env.direct_env_cfg import (  # noqa: E402
    Task1DimmInsertEnvCfg,
    resolve_dimm_usd_path,
    resolve_franka_usd_path,
    resolve_socket_usd_path,
)
from tasks.task1_dimm_insert.env.direct_env import Task1DimmInsertEnv  # noqa: E402
from tasks.task1_dimm_insert.env.register_gym import register_task1_gym  # noqa: E402
from tasks.task1_dimm_insert.env.lab_env import load_safety_limits  # noqa: E402
from tasks.task1_dimm_insert.env.make_env import ACT_DIM  # noqa: E402


def _p(msg: str) -> None:
    print(f"[kit_smoke {time.strftime('%H:%M:%S')}] {msg}", flush=True)
    sys.stdout.flush()


def _crumb(msg: str) -> None:
    path = Path("/tmp/embodied_kit_smoke/crumbs.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    _p(f"CRUMB {msg}")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _action_dim(env) -> int:
    """DirectRLEnv has no num_actions; use single_action_space / ACT_DIM."""
    try:
        import gymnasium as gym
        return int(gym.spaces.flatdim(env.single_action_space))
    except Exception:
        try:
            shape = env.single_action_space.shape
            return int(shape[-1] if shape else ACT_DIM)
        except Exception:
            return int(ACT_DIM)


def main() -> int:
    Path("/tmp/embodied_kit_smoke").mkdir(parents=True, exist_ok=True)
    Path("/tmp/embodied_kit_smoke/crumbs.txt").write_text("")

    _p("Embodied_Safety Stage 1A")
    _p(f"ROOT={ROOT}")
    _p(f"franka_usd={resolve_franka_usd_path()}")
    _p(f"dimm_usd={resolve_dimm_usd_path()}")
    _p(f"socket_usd={resolve_socket_usd_path()}")
    _p(f"num_envs={args_cli.num_envs} max_iterations={args_cli.max_iterations}")
    _p(f"do_learn={args_cli.do_learn} control_mode={args_cli.control_mode}")

    gym_id = register_task1_gym()
    _p(f"gym_id={gym_id}")

    cfg = Task1DimmInsertEnvCfg()
    cfg.scene.num_envs = int(args_cli.num_envs)
    cfg.seed = int(args_cli.seed)
    cfg.sim.device = str(getattr(args_cli, "device", "cuda:0"))
    cfg.control_mode = str(args_cli.control_mode)
    cfg.wait_for_textures = False  # avoid headless texture wait loops
    limits = load_safety_limits(Path(args_cli.scene_path))
    for k in (
        "latch_overforce_N",
        "grip_force_limit_N",
        "axial_contact_N",
        "lateral_contact_N",
    ):
        if k in limits:
            setattr(cfg, k, float(limits[k]))
    cfg._scene_path_override = args_cli.scene_path  # type: ignore[attr-defined]

    _p("building Task1DimmInsertEnv …")
    t_build = time.perf_counter()
    env = Task1DimmInsertEnv(cfg=cfg)
    # Force headless GUI flag if AppLauncher left it unset (was None → weird close/render).
    if getattr(env.sim, "_has_gui", None) is None:
        env.sim._has_gui = False
        _crumb("forced sim._has_gui=False")
    _crumb(f"has_gui={env.sim.has_gui()} has_rtx={env.sim.has_rtx_sensors()}")
    _p(
        f"env ok num_envs={env.num_envs} max_episode_length={env.max_episode_length} "
        f"build={(time.perf_counter()-t_build):.2f}s"
    )

    _p("reset …")
    t0 = time.perf_counter()
    obs = env.reset()
    _crumb(f"reset ok type={type(obs)} dt={(time.perf_counter()-t0):.3f}s")

    import torch

    act_dim = _action_dim(env)
    device = env.device
    _crumb(f"act_dim={act_dim} device={device}")
    zeros = torch.zeros(env.num_envs, act_dim, device=device)
    _crumb(f"zeros ready shape={tuple(zeros.shape)}")

    n_steps = int(args_cli.num_steps) if args_cli.no_learn else 1
    soft_to = float(args_cli.step_timeout_s)

    for i in range(n_steps):
        _crumb(f"step[{i}] ENTER")
        t1 = time.perf_counter()
        step_out = env.step(zeros)
        dt = time.perf_counter() - t1
        flag = " SLOW" if dt > soft_to else ""
        _crumb(f"step[{i}] EXIT dt={dt:.3f}s out_type={type(step_out)}{flag}")
        if dt > soft_to:
            _p(f"WARNING: step exceeded soft timeout {soft_to}s")

    if args_cli.no_learn:
        _crumb("closing env (no-learn)")
        env.close()
        _p("DONE kit_step_smoke_ok")
        return 0

    from algos.rsl_rl_ppo.runner import Runner  # noqa: E402
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from datetime import datetime

    vec = RslRlVecEnvWrapper(env)
    _p(f"RslRlVecEnvWrapper num_envs={vec.num_envs} num_actions={vec.num_actions}")

    log_dir = (
        ROOT
        / "outputs/runs/task1_dimm_insert/rsl_rl_ppo/stage1a"
        / f"kit_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    algo_cfg = {
        "num_envs": args_cli.num_envs,
        "seed": args_cli.seed,
        "device": getattr(args_cli, "device", "cuda:0"),
        "max_iterations": min(int(args_cli.max_iterations), 2),
        "num_steps_per_env": 24,
        "experiment_name": "kit_smoke_stage1a",
    }
    runner = Runner(vec, algo_cfg, log_dir=str(log_dir), device=str(getattr(args_cli, "device", "cuda:0")))
    _p(f"OnPolicyRunner ready={runner.ready} log_dir={log_dir}")

    status = "kit_construct_ok"
    if args_cli.do_learn and runner.ready:
        iters = min(int(args_cli.max_iterations), 2)
        _p(f"learn({iters}) …")
        runner.learn(num_learning_iterations=iters)
        status = "kit_learn_smoke_ok"
        _p(f"learn finished status={status}")
    else:
        _p(f"learn skipped status={status}")

    env.close()
    _p(f"DONE {status}")
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception as exc:
        _p(f"FATAL {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        try:
            simulation_app.close()
        except Exception as close_exc:
            _p(f"simulation_app.close error: {close_exc}")
    raise SystemExit(code)
