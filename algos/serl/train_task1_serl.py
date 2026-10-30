#!/usr/bin/env python3
"""Single-process SERL SAC on Task1 plain DIMM insert (Isaac Lab).

Online-only by default. With --demo-pkl, mixes 50/50 demo/online (RLPD-style).
Launch via isaaclab.sh after AppLauncher args.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Embodied_Safety Task1 SERL SAC")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_steps", type=int, default=35000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--start_steps", type=int, default=1000)
parser.add_argument("--ckpt_interval", type=int, default=5000)
parser.add_argument(
    "--offline-demo",
    choices=("none", "required"),
    default="none",
    help="none: online-only. required: need --demo-pkl with transitions.",
)
parser.add_argument("--demo-pkl", type=str, default="", help="Pickle list of SERL transitions")
parser.add_argument("--control-mode", choices=("diff_ik", "joint"), default="diff_ik")
parser.add_argument("--logdir", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

ES_ROOT = Path("/mnt/data2/wjy_SERL/Embodied_Safety")
SERL_ROOT = Path("/mnt/data2/wjy_SERL")
_JAX_SITE = str(SERL_ROOT / "SERL-runs/task1_serl_py310_jax")
sys.path[:0] = [
    _JAX_SITE,
    str(SERL_ROOT / "serl_launcher"),
    str(ES_ROOT),
    str(ES_ROOT / "algos/serl"),
]
os.chdir(ES_ROOT)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_ROOT", "/usr/local/cuda")

import gym
import jax
import numpy as np

from tasks.task1_dimm_insert.env.make_env import ACT_DIM, OBS_DIM
from isaac_task1_env import IsaacTask1Env
from replay import Task1ReplayBuffer, sample_half_batch
from serl_launcher.utils.launcher import make_sac_agent


def load_demo_transitions(path: Path) -> list:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list) or not data:
        raise SystemExit(f"demo pkl must be a nonempty list of transitions: {path}")
    required = {"observations", "actions", "next_observations", "rewards", "masks", "dones"}
    sample = data[0]
    missing = required - set(sample)
    if missing:
        raise SystemExit(f"demo transitions missing keys {missing}")
    return data


def save_checkpoint(path: Path, agent, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(
            {
                "agent": "sac",
                "step": step,
                "seed": args_cli.seed,
                "offline_demo": args_cli.offline_demo,
                "params": jax.device_get(agent.state.params),
                "target_params": jax.device_get(agent.state.target_params),
            },
            handle,
        )


def to_numpy_batch(batch):
    return {key: np.asarray(batch[key]) for key in batch.keys()}


def to_jax(batch):
    return jax.tree_map(lambda value: jax.device_put(np.asarray(value)), batch)


def main():
    demo_path = Path(args_cli.demo_pkl) if args_cli.demo_pkl else None
    demos = []
    if args_cli.offline_demo == "required":
        if demo_path is None or not demo_path.is_file():
            raise SystemExit(
                "offline-demo required, but --demo-pkl is missing. "
                "Online-only was not started as a fallback. See DEMO_REQUIREMENTS.md."
            )
        demos = load_demo_transitions(demo_path)
    elif demo_path is not None and demo_path.is_file():
        demos = load_demo_transitions(demo_path)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    logdir = Path(args_cli.logdir) if args_cli.logdir else (
        ES_ROOT
        / "outputs/runs/task1_dimm_insert/serl/task1_serl_sac_plain"
        / f"seed{args_cli.seed}_{stamp}"
    )
    logdir.mkdir(parents=True, exist_ok=True)
    # Allow launcher to pre-create LOGDIR with stdout/pid; block only if a prior run left config.json.
    if (logdir / "config.json").is_file():
        raise SystemExit(f"refusing to overwrite existing run {logdir}")

    np.random.seed(args_cli.seed)
    import torch

    torch.manual_seed(args_cli.seed)

    env = IsaacTask1Env(device=args_cli.device, control_mode=args_cli.control_mode)
    obs, info = env.reset()
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACT_DIM,), dtype=np.float32)
    online = Task1ReplayBuffer(observation_space, action_space, capacity=200_000)
    demo_buf = Task1ReplayBuffer(observation_space, action_space, capacity=max(len(demos), 1))
    for tr in demos:
        demo_buf.insert(
            {
                "observations": np.asarray(tr["observations"], dtype=np.float32).reshape(OBS_DIM),
                "actions": np.asarray(tr["actions"], dtype=np.float32).reshape(ACT_DIM),
                "next_observations": np.asarray(tr["next_observations"], dtype=np.float32).reshape(OBS_DIM),
                "rewards": np.float32(tr["rewards"]),
                "masks": np.float32(tr["masks"]),
                "dones": bool(tr["dones"]),
            }
        )
    demo_transitions = len(demos)

    sample_obs = np.zeros((1, OBS_DIM), np.float32)
    sample_action = np.zeros((1, ACT_DIM), np.float32)
    agent = make_sac_agent(args_cli.seed, sample_obs, sample_action, discount=0.99)
    rng = jax.random.PRNGKey(args_cli.seed)

    config = {
        "agent": "sac",
        "seed": args_cli.seed,
        "max_steps": args_cli.max_steps,
        "start_steps": args_cli.start_steps,
        "batch_size": args_cli.batch_size,
        "offline_demo": args_cli.offline_demo,
        "demo_transitions": demo_transitions,
        "demo_fraction_requested": 0.5 if demo_transitions > 0 else 0.0,
        "observation": "task1_plain_policy_float32_30",
        "action": f"task1_{args_cli.control_mode}_float32_7",
        "control_mode": args_cli.control_mode,
        "discount": 0.99,
        "safety_constraints": False,
        "tag": "TASK1_SERL_SAC_PLAIN_20261025",
    }
    (logdir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print("TASK1_SERL_CONFIG", json.dumps(config), flush=True)

    ep_path = logdir / "episodes.csv"
    with ep_path.open("w", newline="") as handle:
        csv.writer(handle).writerow(
            ["learner_step", "ep_return", "ep_len", "success", "terminated", "truncated"]
        )

    updates = 0
    ep_ret = 0.0
    ep_len = 0
    success_count = 0
    episode_count = 0

    for step in range(args_cli.max_steps):
        if step < args_cli.start_steps:
            action = np.random.uniform(-1, 1, size=(ACT_DIM,)).astype(np.float32)
        else:
            rng, key = jax.random.split(rng)
            action = agent.sample_actions(
                observations=jax.device_put(obs), seed=key, deterministic=False
            )
            action = np.asarray(jax.device_get(action), dtype=np.float32).reshape(ACT_DIM)

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        online.insert(
            {
                "observations": obs,
                "actions": action,
                "next_observations": next_obs,
                "rewards": np.float32(reward),
                "masks": np.float32(1.0 - float(terminated)),
                "dones": bool(terminated or truncated),
            }
        )
        ep_ret += float(reward)
        ep_len += 1
        obs = next_obs

        if len(online) >= args_cli.batch_size and step >= args_cli.start_steps:
            if demo_transitions > 0:
                batch, mix = sample_half_batch(online, demo_buf, args_cli.batch_size)
            else:
                batch = to_numpy_batch(online.sample(batch_size=args_cli.batch_size))
                mix = {"demo_fraction": 0.0, "demo_count": 0, "online_count": args_cli.batch_size}
            agent, _info = agent.update(to_jax(batch))
            updates += 1
            if updates == 1:
                print("FIRST_UPDATE", json.dumps({"mix": mix}), flush=True)

        if terminated or truncated:
            success = bool(next_info.get("success", terminated))
            success_count += int(success)
            episode_count += 1
            with ep_path.open("a", newline="") as handle:
                csv.writer(handle).writerow(
                    [step + 1, ep_ret, ep_len, int(success), int(terminated), int(truncated)]
                )
            if episode_count % 10 == 0:
                print(
                    f"ep={episode_count} step={step+1} ret={ep_ret:.2f} "
                    f"success_rate={success_count/max(episode_count,1):.3f} updates={updates}",
                    flush=True,
                )
            ep_ret = 0.0
            ep_len = 0
            obs, _ = env.reset()

        if args_cli.ckpt_interval > 0 and (step + 1) % args_cli.ckpt_interval == 0:
            save_checkpoint(logdir / "checkpoints" / f"step_{step + 1}.pkl", agent, step + 1)
        if (step + 1) % 1000 == 0:
            print(f"progress step {step + 1} updates {updates}", flush=True)

    save_checkpoint(logdir / "checkpoints" / "latest.pkl", agent, args_cli.max_steps)
    save_checkpoint(logdir / "agent.pkl", agent, args_cli.max_steps)
    summary = {
        "env_steps": args_cli.max_steps,
        "updates": updates,
        "episodes": episode_count,
        "success_count": success_count,
        "success_rate": success_count / max(episode_count, 1),
        "demo_transitions": demo_transitions,
        "logdir": str(logdir),
    }
    (logdir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    env.close()
    print("TASK1_SERL_TRAIN_EXIT:0", logdir, flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
