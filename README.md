# Embodied_Safety

Multi-task / multi-algo layout for embodied-safety training (Isaac).

- Tasks: `tasks/<id>/` (scene + env contract)
- Algos: `algos/<name>/` (learning only; rewrite in-tree)
- Glue: `runners/`
- Scene for task1: **real copy** at `tasks/task1_dimm_insert/scene/` (final v002)
- `franka_sim/t1_rl` is **not** used here (DROPPED); see `MIGRATION.md`

## Stage 1A smoke

```bash
cd /mnt/data2/wjy_SERL/Embodied_Safety
PYTHONPATH=. python -m runners.train \
  --task task1_dimm_insert --algo rsl_rl_ppo \
  --config configs/experiments/task1_rsl_rl_ppo_stage1a.yaml
```
