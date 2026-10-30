# task1_dimm_insert

DIMM insert + Plan-D latch/thin-contact safety (Franka). Stage 1A = low-dim state RL.

## Scene

- **Authoritative:** `scene/` — slim real copy of final `franka_manual_safety_v002` (~1.7–1.8M)
- Absolute: `/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task1_dimm_insert/scene`
- Backup: `/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_manual_safety_v002/`
- Details: `SCENE.md`

## Env (Stage 1A scaffold)

| Path | Role |
|------|------|
| `env/make_env.py` | `make_env(cfg)` → `FakeEnv` (smoke); Isaac Lab later |
| `env/OBS_SPEC.md` | Joint / gripper / DIMM–socket / latch / forces (dim 30) |
| `env/REWARD_SPEC.md` | approach / align / insert / dual_latch / force_penalty + over-force terminate |

Obs are **sim state reads** (no vision in Stage 1A). Force limits conceptually from `scene/safety_config.json`.

## Layout

- `scene/` — USD + teleop/safety scripts (slim)
- `env/` — Gym-like contract + FakeEnv
- `teleop/` — placeholder
- `configs/` — task knobs (placeholder)

## Algo

Rewrite under `algos/rsl_rl_ppo/` (clean stub). **`franka_sim/t1_rl` is not used.**

```bash
cd /mnt/data2/wjy_SERL/Embodied_Safety
PYTHONPATH=. python -m runners.train \
  --task task1_dimm_insert --algo rsl_rl_ppo \
  --config configs/experiments/task1_rsl_rl_ppo_stage1a.yaml
```
