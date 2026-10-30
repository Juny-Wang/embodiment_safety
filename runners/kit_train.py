#!/usr/bin/env python3
"""Kit-headless Stage 1A PPO long train (DirectRLEnv + RSL-RL).

Launched via isaaclab.sh. Loads algos/rsl_rl_ppo/cfg/stage1a.yaml by default.
Requires --allow-long-run when max_iterations > 5 (same guard as algos/.../train.py).
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ALGO_CFG = ROOT / "algos/rsl_rl_ppo/cfg/stage1a.yaml"


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        data: dict = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or ":" not in s:
                continue
            if s.startswith("notes:"):
                break
            k, _, v = s.partition(":")
            k, v = k.strip(), v.strip().strip("\"'")
            if not k or not v or v.startswith(("|", "[", "{")):
                continue
            if v.lower() in ("true", "false"):
                data[k] = v.lower() == "true"
            else:
                try:
                    data[k] = int(v) if "." not in v and "e" not in v.lower() else float(v)
                except ValueError:
                    data[k] = v
        return data


_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--algo-cfg", type=str, default=str(DEFAULT_ALGO_CFG))
_pre_args, _ = _pre.parse_known_args()
_algo_defaults = _load_yaml(Path(_pre_args.algo_cfg)) if Path(_pre_args.algo_cfg).is_file() else {}

parser = argparse.ArgumentParser(description="Embodied_Safety Stage 1A Kit long train")
parser.add_argument("--algo-cfg", type=str, default=str(DEFAULT_ALGO_CFG))
parser.add_argument("--num_envs", type=int, default=int(_algo_defaults.get("num_envs", 64)))
parser.add_argument("--max_iterations", type=int, default=int(_algo_defaults.get("max_iterations", 2000)))
parser.add_argument("--seed", type=int, default=int(_algo_defaults.get("seed", 42)))
parser.add_argument(
    "--allow-long-run",
    action="store_true",
    help="Required when max_iterations > 5 (refuses accidental multi-hour runs otherwise)",
)
parser.add_argument(
    "--control-mode",
    type=str,
    default="joint",
    choices=("joint", "diff_ik"),
)
parser.add_argument(
    "--scene_path",
    type=str,
    default=str(ROOT / "tasks/task1_dimm_insert/scene"),
)
parser.add_argument(
    "--run-name",
    type=str,
    default="",
    help="Optional suffix under stage1a output dir",
)
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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
from algos.rsl_rl_ppo.runner import Runner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402


def _p(msg: str) -> None:
    print(f"[kit_train {time.strftime('%H:%M:%S')}] {msg}", flush=True)
    sys.stdout.flush()


def _action_dim(env) -> int:
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
    algo_path = Path(args_cli.algo_cfg)
    if not algo_path.is_absolute():
        cand = ROOT / algo_path
        if cand.is_file():
            algo_path = cand
    algo_cfg = _load_yaml(algo_path) if algo_path.is_file() else dict(_algo_defaults)
    algo_cfg["num_envs"] = int(args_cli.num_envs)
    algo_cfg["seed"] = int(args_cli.seed)
    algo_cfg["device"] = str(getattr(args_cli, "device", algo_cfg.get("device", "cuda:0")))
    algo_cfg["max_iterations"] = int(args_cli.max_iterations)
    if args_cli.run_name:
        algo_cfg["run_name"] = args_cli.run_name

    max_iters = int(algo_cfg["max_iterations"])
    if max_iters > 5 and not bool(args_cli.allow_long_run):
        raise RuntimeError(
            f"Refusing learn(max_iterations={max_iters}). Pass --allow-long-run "
            "or set --max_iterations<=5 for a short smoke."
        )

    _p("Embodied_Safety Stage 1A LONG TRAIN")
    _p(f"ROOT={ROOT}")
    _p(f"algo_cfg={algo_path}")
    _p(f"franka_usd={resolve_franka_usd_path()}")
    _p(f"dimm_usd={resolve_dimm_usd_path()}")
    _p(f"socket_usd={resolve_socket_usd_path()}")
    _p(
        f"num_envs={algo_cfg['num_envs']} max_iterations={max_iters} "
        f"save_interval={algo_cfg.get('save_interval')} "
        f"num_steps_per_env={algo_cfg.get('num_steps_per_env')} "
        f"allow_long_run={args_cli.allow_long_run}"
    )
    _p(f"control_mode={args_cli.control_mode} seed={algo_cfg['seed']}")

    gym_id = register_task1_gym()
    _p(f"gym_id={gym_id}")

    cfg = Task1DimmInsertEnvCfg()
    cfg.scene.num_envs = int(algo_cfg["num_envs"])
    cfg.seed = int(algo_cfg["seed"])
    cfg.sim.device = str(algo_cfg["device"])
    cfg.control_mode = str(args_cli.control_mode)
    cfg.wait_for_textures = False
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
    if getattr(env.sim, "_has_gui", None) is None:
        env.sim._has_gui = False
        _p("forced sim._has_gui=False")
    _p(
        f"env ok num_envs={env.num_envs} obs_space dim check act={_action_dim(env)} "
        f"max_episode_length={env.max_episode_length} build={(time.perf_counter()-t_build):.2f}s"
    )

    _p("reset …")
    obs = env.reset()
    _p(f"reset ok type={type(obs)}")

    import torch

    act_dim = _action_dim(env)
    zeros = torch.zeros(env.num_envs, act_dim, device=env.device)
    _ = env.step(zeros)
    _p(f"warmup step ok act_dim={act_dim} device={env.device}")

    vec = RslRlVecEnvWrapper(env)
    _p(f"RslRlVecEnvWrapper num_envs={vec.num_envs} num_actions={vec.num_actions}")

    out_base = algo_cfg.get("output_dir") or "outputs/runs/task1_dimm_insert/rsl_rl_ppo/stage1a"
    out_base_p = Path(str(out_base))
    if not out_base_p.is_absolute():
        out_base_p = ROOT / out_base_p
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_suffix = args_cli.run_name.strip() or f"train_{stamp}"
    if not run_suffix.startswith("train_"):
        run_suffix = f"train_{run_suffix}_{stamp}" if stamp not in run_suffix else f"train_{run_suffix}"
    log_dir = out_base_p / run_suffix
    # Prefer clean stamp dir when run-name empty
    if not args_cli.run_name:
        log_dir = out_base_p / f"train_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Persist resolved cfg for reproducibility
    try:
        import yaml  # type: ignore
        with (log_dir / "stage1a_resolved.yaml").open("w", encoding="utf-8") as fh:
            yaml.safe_dump(dict(algo_cfg), fh, sort_keys=False)
    except Exception:
        (log_dir / "stage1a_resolved.txt").write_text(repr(algo_cfg), encoding="utf-8")
    (log_dir / "launch_cmd.txt").write_text(" ".join(sys.argv), encoding="utf-8")

    runner = Runner(vec, algo_cfg, log_dir=str(log_dir), device=str(algo_cfg["device"]))
    _p(f"OnPolicyRunner ready={runner.ready} log_dir={log_dir}")
    if not runner.ready:
        raise RuntimeError("OnPolicyRunner not ready")

    _p(f"learn({max_iters}) …")
    t_learn = time.perf_counter()
    runner.learn(num_learning_iterations=max_iters)
    dt = time.perf_counter() - t_learn
    _p(f"learn finished dt={dt:.1f}s status=kit_train_ok log_dir={log_dir}")

    env.close()
    _p("DONE kit_train_ok")
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
