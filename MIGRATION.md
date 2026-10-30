# Migration status (Embodied_Safety)

Root: `/mnt/data2/wjy_SERL/Embodied_Safety`

| Item | Source (franka_sim) | Target | Status |
| --- | --- | --- | --- |
| Scene package v002 | `/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_manual_safety_v002` | `tasks/task1_dimm_insert/scene` (**real copy**) | **DONE** (copied 2026-10-24; original kept as backup) |
| Training code t1_rl | `/mnt/data2/wjy_SERL/franka_sim/t1_rl` | — | **DROPPED** from Embodied_Safety (not based on final v002; rewrite later in `algos/rsl_rl_ppo/`) |
| Shared robot/camera | (later) | `lib/` | deferred |

## Scene copy (done)

```bash
test ! -L /mnt/data2/wjy_SERL/Embodied_Safety/tasks/task1_dimm_insert/scene && echo NOT_SYMLINK
du -sh /mnt/data2/wjy_SERL/Embodied_Safety/tasks/task1_dimm_insert/scene
test -f /mnt/data2/wjy_SERL/Embodied_Safety/tasks/task1_dimm_insert/scene/keyboard_panel.py && echo KP_OK
test -d /mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_manual_safety_v002 && echo V002_BACKUP_INTACT
```

## Algo (DROPPED)

- Removed `algos/rsl_rl_ppo/legacy` symlink.
- `algos/rsl_rl_ppo/` is a **clean stub** for rewrite (no import of `t1_rl`).
- `/mnt/data2/wjy_SERL/franka_sim/t1_rl` left in place as historical reference only.

```bash
test ! -e /mnt/data2/wjy_SERL/Embodied_Safety/algos/rsl_rl_ppo/legacy && echo LEGACY_GONE
test -d /mnt/data2/wjy_SERL/franka_sim/t1_rl && echo T1RL_KEPT
```

## Stage 1A PPO scaffold (2026-10-24)

| Item | Path | Status |
|------|------|--------|
| Env + FakeEnv + make_env | `tasks/task1_dimm_insert/env/` | DONE (scaffold) |
| OBS / REWARD specs | `env/OBS_SPEC.md`, `env/REWARD_SPEC.md` | DONE |
| Algo cfg + train dry-run | `algos/rsl_rl_ppo/cfg/stage1a.yaml`, `train.py` | DONE (scaffold) |
| Experiment YAML | `configs/experiments/task1_rsl_rl_ppo_stage1a.yaml` | DONE |
| runners.train wire | imports make_env + algo.train | DONE (no Isaac) |

Smoke:

```bash
cd /mnt/data2/wjy_SERL/Embodied_Safety
PYTHONPATH=. python -m runners.train \
  --task task1_dimm_insert --algo rsl_rl_ppo \
  --config configs/experiments/task1_rsl_rl_ppo_stage1a.yaml
```

## Residual (not done)

1. Isaac Lab VecEnv wrapping `tasks/task1_dimm_insert/scene/`.
2. Real RSL-RL `OnPolicyRunner.learn()` (replace FakeEnv dry-run).
3. Tune reward weights from teleop baselines; load limits from `safety_config.json` at runtime.
