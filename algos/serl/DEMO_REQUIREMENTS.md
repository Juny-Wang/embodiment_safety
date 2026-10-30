# Task1 SERL demo requirements

Updated 2026-10-25 Asia/Shanghai.

## Schema (must match train_task1_serl.py)

| Field | Spec |
|-------|------|
| Count | Prefer **10–20 successful** inserts (start with 3 for smoke) |
| Obs | `float32` length **30** — `OBS_SPEC.md` / `make_env.OBS_DIM` |
| Action | `float32` length **7**, clipped `[-1, 1]` — EE delta + gripper, `control_mode=diff_ik` |
| Transition keys | `observations`, `actions`, `next_observations`, `rewards`, `masks` (`1-terminated`), `dones` |
| Reward | From **Task1PlainInsertEnv** at record time (same as SERL train) |
| Format | Pickle **list** of transition dicts |
| Default path | `Embodied_Safety/outputs/demos/task1_dimm_insert/serl_<N>_demos_<timestamp>.pkl` |

Failed / timeout episodes are **discarded** by the recorder.

## Scene (required)

Confirmed Task1 training / teleop scene pack (franka_manual_safety_v002 copy):

`/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task1_dimm_insert/scene/manual_franka_scene.usda`

Recorder uses **Task1PlainInsertEnv** (same Franka / DIMM / Socket **physics** as plain PPO)
via Isaac Lab → **Isaac Sim 4.5** (`isaaclab.sh`). Lab does **not** `open_stage` the USDA
(see `env/lab_env.py`); `--scene-path` feeds safety JSON only.

For GUI (so it is not an empty grid), the recorder composes Floor+Table from
`env/assets/lab_task1_backdrop.usda` under `/World/Task1Backdrop` (teleop table look,
Lab-aligned). Do **not** attach full MatchedSocket visual overlays (static-static
`LeftHinge` PhysicsUSD errors).

Isaac Sim **5.0** Script-Editor teleop of `manual_franka_scene.usda` (scene/README.md)
is a separate entry and does **not** emit 30-D Lab obs for SERL.

Keep **IsaacLab Environment Index = 0** (`num_envs=1`).

## How to collect (keyboard teleop)

On isaac-b (display / remote desktop so you can see Isaac GUI):

```bash
# 10 successful demos (default)
bash /mnt/data2/wjy_SERL/Embodied_Safety/algos/serl/run_record_demos.sh
# equivalent:
# SCENE_PATH=/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task1_dimm_insert/scene \
#   bash /mnt/data2/wjy_SERL/Embodied_Safety/algos/serl/run_record_demos.sh

# or 20
SUCCESS_NEEDED=20 bash /mnt/data2/wjy_SERL/Embodied_Safety/algos/serl/run_record_demos.sh

# smoke with 3
SUCCESS_NEEDED=3 MAX_ATTEMPTS=20 bash /mnt/data2/wjy_SERL/Embodied_Safety/algos/serl/run_record_demos.sh
```

### Controls (after Isaac opens — **click the main viewport** for focus)

| Key | Effect |
|-----|--------|
| ↑ ↓ ← → | Translate EE |
| PgUp / PgDn | Translate EE Z |
| Shift + arrows / Pg | Rotate EE |
| `[` or `O` | Open gripper |
| `]` or `P` | Close gripper |
| `R` | Reset episode (discard current) |

Console prints `SUCCESS k/N` when an insert seats (depth + XY + dual latch). When `N` successes are reached, a `.pkl` + `.json` meta file are written under `outputs/demos/task1_dimm_insert/`.

### Validate

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
python /mnt/data2/wjy_SERL/Embodied_Safety/algos/serl/validate_demo_pkl.py \
  /mnt/data2/wjy_SERL/Embodied_Safety/outputs/demos/task1_dimm_insert/serl_10_demos_YYYY-MM-DD_HH-MM-SS.pkl
```

### Train with demos

```bash
OFFLINE_DEMO=required \
DEMO_PKL=/mnt/data2/wjy_SERL/Embodied_Safety/outputs/demos/task1_dimm_insert/serl_10_demos_....pkl \
bash /mnt/data2/wjy_SERL/Embodied_Safety/algos/serl/run_sac_online.sh
```

## Files

| Path | Role |
|------|------|
| `algos/serl/record_demos.py` | Kit entry: env + keyboard → SERL transitions |
| `algos/serl/keyboard_intervene.py` | Carb keyboard → 7-D action |
| `algos/serl/run_record_demos.sh` | GUI launcher |
| `algos/serl/validate_demo_pkl.py` | Offline schema check |

## Why not reuse scene Script-Editor teleop logs

Scene teleop (`keyboard_panel` / `franka_teleop`) drives joint targets in the USDA scene and does **not** emit Task1PlainInsertEnv 30-D obs / plain reward / `masks`. SERL RLPD needs transitions from **this** DirectRLEnv (see drawer demo audit).
