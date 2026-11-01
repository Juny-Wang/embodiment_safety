"""Native Task1 inference and online residual SAC updates."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from algos.dp_sac.base import FrozenDP
from algos.dp_sac.sac import SAC, compose_action, transition_mask


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def run(out):
    torch.set_num_threads(2)
    cfg = json.loads(Path(os.environ["DP_SAC_CONFIG"]).read_text())
    learn = os.environ.get("TASK1_ONLINE_LEARN") == "1"
    baseline = os.environ.get("DP_SAC_BASELINE") == "1"
    if os.environ.get("TASK1_SAFETY_MODE", "stop") != "stop" or os.environ.get("TASK1_SMOKE") == "1":
        raise ValueError("DP-SAC uses ordinary task actions and stop-mode transitions")
    if learn and (baseline or os.environ.get("TASK1_EVAL") == "1"):
        raise ValueError("Evaluation cannot update SAC")
    seed = int(os.environ["TASK1_EPISODE_SEED"])
    base = FrozenDP(os.environ["SERL_INPUT_CHECKPOINT"], cfg["device"], seed)
    agent = SAC(base.feature_dim, cfg, cfg["device"], seed)
    resume = os.environ.get("DP_SAC_CHECKPOINT")
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["base_checkpoint"] != str(base.path):
            raise ValueError("Residual checkpoint belongs to another DP base")
        agent.load_state_dict(saved["sac"])
    start_updates, start_total = agent.updates, agent.replay.total
    write_json(out / "loaded_checkpoint.json", dict(
        algorithm="dp_residual_sac", base_checkpoint=str(base.path), weights="ema",
        residual_checkpoint=resume, base_frozen=True, state_dim=33, history_frames=2,
        feature_dim=base.feature_dim, residual_scale=cfg["residual_scale"], learning=learn,
        initial_updates=start_updates, initial_online=start_total, base_seed=seed))

    def save():
        metadata = dict(algorithm="dp_residual_sac", base_checkpoint=str(base.path),
                        config=cfg, updates=agent.updates, online_transitions=agent.replay.total,
                        replay_size=agent.replay.size, base_frozen=True,
                        interface_version="task1_rgb_state33_rot6d_reward_history_v002",
                        reward_version="task1_stage_once_v001")
        tmp = out / "checkpoint.pt.tmp"
        torch.save(dict(metadata, sac=agent.state_dict()), tmp)
        tmp.replace(out / "checkpoint.pt")
        write_json(out / "checkpoint.json", metadata)

    def observation(index):
        with np.load(out / f"obs_{index:03d}.npz", allow_pickle=False) as f:
            return {key: f[key].copy() for key in f.files}

    accepted = rejected = 0
    metrics = (out / "metrics.jsonl").open("x", buffering=1)
    transitions = (out / "residual_transitions.jsonl").open("x", buffering=1)

    def insert(previous, next_features):
        nonlocal accepted, rejected
        index, features, residual = previous
        label = json.loads((out / f"stage_transition_{index:03d}.json").read_text())
        mask = transition_mask(label)
        transitions.write(json.dumps(dict(index=index, reward=label["reward"], mask=mask,
                                          eligible=mask is not None, learning=learn,
                                          terminated=label["terminated"], truncated=label["truncated"])) + "\n")
        if mask is None:
            rejected += 1
            return
        accepted += 1
        if not learn:
            return
        agent.replay.append(features, residual, label["reward"], next_features, mask)
        if agent.replay.total >= cfg["learning_starts"] and agent.replay.size >= cfg["batch_size"]:
            for _ in range(cfg["updates_per_step"]):
                row = agent.update()
                row.update(transition=index, online_transitions=agent.replay.total)
                metrics.write(json.dumps(row, allow_nan=False) + "\n")
        if agent.replay.total % cfg["checkpoint_every"] == 0:
            save()

    if learn:
        save()
    previous = None
    n = 0
    try:
        while True:
            request = out / f"request_{n:03d}.json"
            deadline = time.monotonic() + 600
            while not request.exists() and not (out / "policy_stop.json").exists():
                if (out / "actor_error.json").exists():
                    raise RuntimeError("Isaac actor failed")
                if time.monotonic() > deadline:
                    raise TimeoutError("Waiting for Task1 observation")
                time.sleep(.02)
            if not request.exists():
                break
            features, action_base = base.predict(observation(n))
            if previous is not None:
                insert(previous, features)
            warming = learn and agent.replay.total < cfg["learning_starts"]
            residual = np.zeros(7, np.float32) if baseline or warming else agent.action(features, not learn)
            action = compose_action(action_base, residual, cfg["residual_scale"])
            write_json(out / f"action_{n:03d}.json", dict(
                index=n, base_action=action_base.tolist(), residual_action=residual.tolist(),
                residual_scale=cfg["residual_scale"], applied_residual=(action-action_base).tolist(),
                proposed=(action_base + np.asarray(cfg["residual_scale"])*residual).tolist(),
                executed=action.tolist(), network_updates=agent.updates-start_updates,
                sampling_mode="dp_only" if baseline or warming else "residual_sample" if learn else "residual_mean",
                base_sampling="seeded_ddim_eta0", base_seed=seed))
            previous = n, features, residual
            n += 1
        final = json.loads((out / "policy_stop.json").read_text())["final_observation_index"]
        if previous is not None and final == previous[0] + 1:
            next_features, _ = base.predict(observation(final))
            insert(previous, next_features)
        elif previous is not None:
            raise ValueError("Missing final observation for executed action")
        if learn:
            save()
        write_json(out / "worker_result.json", dict(
            passed=True, algorithm="dp_residual_sac", learning=learn, base_frozen=True,
            network_updates=agent.updates-start_updates, cumulative_updates=agent.updates,
            initial_online=start_total, final_online=agent.replay.total,
            new_transitions=accepted, rejected_transitions=rejected,
            replay_added=agent.replay.total-start_total, policy_actions=n,
            checkpoint=str(out / "checkpoint.pt") if learn else None))
    finally:
        metrics.close()
        transitions.close()


if __name__ == "__main__":
    output = Path(sys.argv[1])
    try:
        run(output)
    except BaseException:
        write_json(output / "worker_error.json", dict(error=traceback.format_exc()))
        raise
