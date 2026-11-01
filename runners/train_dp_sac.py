"""Train or evaluate frozen DP plus SAC residuals with fresh Isaac resets."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import ROOT, resolve
from .eval_diffusion import label_summary, reconcile_result, stop_group, write_json

DEFAULT_BASE = "outputs/runs/task1/diffusion_policy/full_flow_bc_v001/train/last.pt"


def validate_config(cfg):
    if (cfg["algorithm"] not in ("dp_residual_sac", "policy_decorator") or cfg["state_dim"] != 33 or
        cfg["state_representation"] != "rotation6d33_v002" or cfg["n_obs_steps"] != 2):
        raise ValueError("Expected DP-SAC with two rotation6d33 frames")
    if cfg["algorithm"] == "policy_decorator":
        horizon = cfg["progressive_exploration_steps"]
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("Positive progressive exploration horizon required")
    scale = cfg["residual_scale"]
    if len(scale) != 7 or not all(math.isfinite(x) and 0 < x <= 1 for x in scale):
        raise ValueError("Seven positive residual scales <=1 required")
    for key in ("batch_size", "learning_starts", "updates_per_step", "replay_capacity", "hidden_dim", "checkpoint_every"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError("Positive integer required: " + key)
    if cfg["replay_capacity"] < cfg["batch_size"]:
        raise ValueError("Replay capacity must hold a batch")
    for key in ("discount", "tau"):
        if not 0 < cfg[key] <= 1:
            raise ValueError(key)
    if not all(math.isfinite(cfg[k]) and cfg[k] > 0 for k in ("learning_rate", "initial_alpha")):
        raise ValueError("Positive learning rate and alpha required")
    if not math.isfinite(cfg["target_entropy"]):
        raise ValueError("Finite target entropy required")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint")
    p.add_argument("--checkpoint", help="Residual checkpoint, including optimizer and replay")
    p.add_argument("--config")
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=2400)
    p.add_argument("--wall-timeout-s", type=float, default=3600)
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--baseline", action="store_true", help="Evaluate DP alone with identical base inference")
    a = p.parse_args()
    evaluation = a.eval or a.baseline
    if a.episodes < 1 or a.max_steps < 1 or not math.isfinite(a.wall_timeout_s) or a.wall_timeout_s <= 0:
        raise ValueError("Positive episode, step and wall-time budgets required")
    if a.eval and not a.baseline and not a.checkpoint:
        raise ValueError("Residual evaluation requires --checkpoint; use --baseline for DP alone")
    checkpoint = resolve(a.checkpoint).resolve() if a.checkpoint else None
    metadata = json.loads(checkpoint.with_suffix(".json").read_text()) if checkpoint else None
    if checkpoint and (not checkpoint.is_file() or metadata["algorithm"] not in ("dp_residual_sac", "policy_decorator")):
        raise ValueError("Completed residual SAC checkpoint required")
    base = resolve(a.base_checkpoint or (metadata["base_checkpoint"] if metadata else DEFAULT_BASE)).resolve()
    base_meta = json.loads(base.with_suffix(".json").read_text())
    if (not base.is_file() or base_meta["algorithm"] != "diffusion_policy" or
        base_meta["interface_version"] != "task1_rgb_state33_rot6d_reward_history_v002"):
        raise ValueError("Trained rotation6d33 DP checkpoint required")
    if a.config:
        cfg = json.loads(resolve(a.config).read_text())
    elif metadata:
        cfg = metadata["config"]
    else:
        cfg = json.loads(resolve("configs/experiments/task1_dp_sac.json").read_text())
    validate_config(cfg)
    if metadata and (metadata["base_checkpoint"] != str(base) or metadata["config"] != cfg):
        raise ValueError("Resume must retain base checkpoint and SAC configuration")
    out = resolve(a.output).resolve()
    if not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Repository outputs only")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", cfg)
    report = dict(algorithm=cfg["algorithm"], base_checkpoint=str(base),
                  initial_checkpoint=str(checkpoint) if checkpoint else None,
                  learning=not evaluation, baseline=a.baseline, status="running",
                  config=cfg, episodes=[], network_updates=0, new_transitions=0)
    def save():
        write_json(out / "summary.json", report)
    save()
    try:
        for i in range(a.episodes):
            dest = out / f"episode_{i+1:03d}"
            env = os.environ.copy()
            env.update(TASK1_POLICY_BACKEND="dp_sac", TASK1_SAFETY_MODE="stop",
                       TASK1_EVAL="1" if evaluation else "0", TASK1_DIAGNOSTIC_STOP_STAGE="",
                       DP_SAC_CONFIG=str(out / "config.json"), DP_SAC_BASELINE="1" if a.baseline else "0")
            env.pop("DP_SAC_CHECKPOINT", None)
            if checkpoint:
                env["DP_SAC_CHECKPOINT"] = str(checkpoint)
            cmd = [sys.executable, "-B", "-m", "runners.rollout", "--checkpoint", str(base),
                   "--output", str(dest), "--max-steps", str(a.max_steps), "--seed", str(a.seed+i)]
            cmd.append("--deterministic" if evaluation else "--learn")
            row = dict(index=i+1, output=str(dest), seed=a.seed+i, completed=False)
            report["episodes"].append(row)
            save()
            start = time.monotonic()
            with (out / f"episode_{i+1:03d}.log").open("w") as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                row["pid"] = proc.pid
                save()
                try:
                    rc = proc.wait(timeout=a.wall_timeout_s)
                except subprocess.TimeoutExpired:
                    stop_group(proc)
                    row.update(wall_timeout=True, wall_s=time.monotonic()-start)
                    raise RuntimeError("Episode wall-time budget reached: " + str(dest))
                except BaseException:
                    stop_group(proc)
                    raise
            if rc:
                raise RuntimeError("DP-SAC episode failed: " + str(dest))
            result = json.loads((dest / "result.json").read_text())
            row.update(reconcile_result(result, label_summary(dest)), completed=True,
                       wall_s=time.monotonic()-start)
            worker = json.loads((dest / "worker_result.json").read_text())
            if evaluation and (worker["network_updates"] or worker["replay_added"]):
                raise ValueError("Evaluation changed training state")
            if not evaluation:
                checkpoint = dest / "checkpoint.pt"
            report.update(network_updates=report["network_updates"]+worker["network_updates"],
                          new_transitions=report["new_transitions"]+worker["replay_added"],
                          latest_checkpoint=str(checkpoint) if checkpoint else None)
            save()
        report.update(status="finished", successes=sum(x["full_task_success"] for x in report["episodes"]))
    except BaseException as exc:
        report.update(status="error", error=str(exc))
        raise
    finally:
        save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
