# serl — Task1 DIMM insert

Status (2026-10-25 Asia/Shanghai): **scaffolded, not yet training**.

## What exists on disk

| Path | Role |
|------|------|
| `/mnt/data2/wjy_SERL/serl_launcher` | Upstream SERL JAX agents (SAC / DrQ / …) |
| `/mnt/data2/wjy_SERL/serl_robot_infra` | Real Franka gym + SpaceMouse (PCB/peg insert) |
| `/mnt/data2/wjy_SERL/examples/async_pcb_insert_drq` | Real-robot SERL PCB insert (closest task analogy) |
| `/mnt/data2/wjy_SERL/examples/drawer_open` | **Isaac Lab + SERL SAC** single-process pattern we copy |
| `/mnt/data2/wjy_SERL/SERL-runs/` | Prior drawer SERL runs + `py311_jax` prefix |
| `Embodied_Safety/algos/serl/` | **This package** (was placeholder; now Task1 wiring) |

## Minimal path for Task1

1. **Env**: `Task1PlainInsertEnv` (30-D state, 7-D action, no safety terminate) via `isaac_task1_env.py`.
2. **Algo**: `serl_launcher` SAC (online-only first). Optional later: demo buffer → 50/50 RLPD.
3. **Control**: prefer `diff_ik` (EE delta + gripper) to match SERL relative actions; `joint` kept as fallback.
4. **Demos**: **none on disk yet** for Task1. See `DEMO_REQUIREMENTS.md`.
5. **Train entry**: `train_task1_serl.py` + `run_sac_online.sh`.

## Blockers (do not start train until cleared)

1. **GPU**: plain PPO resume may still hold the 4090 — do **not** kill without user OK.
2. **Demos**: no Task1 `.pkl` / HDF5 demo transitions. Online-only SAC can smoke without demos, but sample-efficient SERL needs ~10–20 successful teleop inserts.
3. **JAX + Isaac**: reuse `SERL-runs/task1_serl_py310_jax` prefix (same as drawer). First smoke must confirm import in Kit python.

## Commands (after GPU free)

Online-only smoke (no demos):

```bash
cd /mnt/data2/workspace_pyx/LNP_Locomotion-main/IsaacLab
bash /mnt/data2/wjy_SERL/Embodied_Safety/algos/serl/run_sac_online.sh
```

Full SERL (after demos): set `OFFLINE_DEMO=required` and `DEMO_PKL=/path/to/demos.pkl`.

## Out of scope here

- Real Franka `FrankaPCBInsert-Vision-v0` (different stack; keep under `examples/`).
- Stage1A force penalties (use plain env until safety SERL-Lagrangian is wired).

## Record demos (human teleop)

```bash
bash /mnt/data2/wjy_SERL/Embodied_Safety/algos/serl/run_record_demos.sh
# SUCCESS_NEEDED=20 for a full set
```

See `DEMO_REQUIREMENTS.md` for keys, output path, and how to pass `--demo-pkl` into training.
