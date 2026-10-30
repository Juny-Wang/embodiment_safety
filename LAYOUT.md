# Layout principles

## Separation

| Area | Owns | Must not own |
|------|------|--------------|
| `tasks/<id>/` | Scene, env (obs/act/reward/done/reset), task teleop, task-only configs | Training loops, optimizer, networks |
| `algos/<name>/` | Optimizer, buffer, networks, algo hyperparams | Task reward formulas, USD paths |
| `runners/` | CLI, registry, glue that wires task env → algo | Business logic |
| `lib/` | Truly shared interfaces (robot / cameras / safety) | "Temporarily shared" task details |
| `configs/experiments/` | One YAML per experiment (task × algo × stage) | Source code |
| `outputs/` | Runs, scene snapshots, baselines (not large git blobs) | Source |

## Contracts

1. Task exposes `make_env(cfg) -> env`; obs/act spaces documented in the task README.
2. Algo consumes env + algo cfg only; no hard `import tasks.task1_...`.
3. A run identity is `task_id + algo_id + experiment_yaml`.
4. Artifacts land at: `outputs/runs/<task>/<algo>/<exp>/`

## Adding

- New task: copy `tasks/_template/`, register in `runners/register.py`.
- New algo: copy `algos/_template/`, register the same way.
- Train: `python -m runners.train --task <id> --algo <name> --config configs/experiments/...yaml`

## Stage 1A (current)

- Obs: low-dim sim state only (`tasks/.../env/OBS_SPEC.md`); no camera in policy.
- Reward/terminate: `REWARD_SPEC.md`; force limits from `scene/safety_config.json` (Plan D demo).
- Algo: `algos/rsl_rl_ppo` dry-runs FakeEnv; Lab + RSL-RL learn later.
- Experiment: `configs/experiments/task1_rsl_rl_ppo_stage1a.yaml`
