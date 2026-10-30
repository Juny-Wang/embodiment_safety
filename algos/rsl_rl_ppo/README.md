# rsl_rl_ppo (Stage 1A)

RSL-RL PPO for low-dim DIMM insert. Clean rewrite under Embodied_Safety —
does **not** import `franka_sim/t1_rl`.

See repo-root **`WIRING.md`** for install discovery, launch commands, and blockers.

## Plug-in shape

```
Isaac Lab VecEnv (batched task1)   ← not implemented yet (lab_env stub)
        ↓  obs/act per OBS_SPEC / ACT
rsl_rl OnPolicyRunner  ←  cfg/stage1a.yaml via runner.build_rsl_train_cfg
        ↓
algos.rsl_rl_ppo.train / Runner.learn
        ↓
outputs/runs/task1_dimm_insert/rsl_rl_ppo/stage1a/<stamp>/
```

## Smoke

```bash
cd /mnt/data2/wjy_SERL/Embodied_Safety
PYTHONPATH=. python -m runners.train \
  --task task1_dimm_insert --algo rsl_rl_ppo \
  --config configs/experiments/task1_rsl_rl_ppo_stage1a.yaml --dry-run
```

## Plain PPO (no safety constraints)

See **`PLAIN_PPO.md`** and algo id `rsl_rl_ppo_plain` / `runners/kit_train_plain.py`. Stage 1A above is unchanged.
