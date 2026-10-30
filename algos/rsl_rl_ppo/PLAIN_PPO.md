# Task1 Plain PPO — task completion only

Separate training path from Stage 1A. **No safety constraints** in reward or
termination: the policy is trained only to finish DIMM insert.

Stage 1A (`cfg/stage1a.yaml`, `runners/kit_train.py`, `REWARD_SPEC.md`) is
unchanged and still includes force penalties / overforce fail.

## Success definition

Episode success (and `terminated=True`) when **all** hold:

1. Insert depth ≥ `success_depth_m` (default 0.012 m)
2. XY error &lt; `2 * xy_tol_m` (default 5 mm)
3. Both latch hinge angles `|θ| &lt; 0.05` rad

Timeout → `truncated` at `max_episode_steps` (default 500).  
**Over-force / latch-break / grip limits do not end the episode** in this trainer.

## Reward (plain)

| Term | Role |
|------|------|
| `approach` | −‖DIMM–socket‖ · `w_approach` |
| `align` | dense XY + tilt (cm-scale) · `w_align` |
| `insert_depth` | depth progress when near-aligned · `w_insert_depth` |
| `dual_latch` | closed hinges when near-aligned · `w_dual_latch` |
| `success_bonus` | `w_success` (default 20) when success |
| `force_penalty` | **always 0** |

Contact sensors may still fill OBS dims 24–29; they are **not** training constraints.

## Files

| Path | Role |
|------|------|
| `algos/rsl_rl_ppo/cfg/plain_insert.yaml` | PPO hyperparams + output dir |
| `algos/rsl_rl_ppo/train_plain.py` | FakeEnv / Lab-stub train entry |
| `configs/experiments/task1_rsl_rl_ppo_plain.yaml` | Experiment YAML |
| `tasks/.../env/plain_direct_env.py` | DirectRLEnv subclass |
| `tasks/.../env/plain_direct_env_cfg.py` | Cfg: `w_force_penalty=0`, `disable_safety_terminate` |
| `runners/kit_train_plain.py` | Kit long / smoke learn |
| Gym id | `Embodied-Task1-DimmInsert-Plain-v0` |

Artifacts: `outputs/runs/task1_dimm_insert/rsl_rl_ppo/task1_plain_ppo/`

## Launch

### 1) FakeEnv dry-run (no Isaac)

```bash
cd /mnt/data2/wjy_SERL/Embodied_Safety
PYTHONPATH=. python -m runners.train \
  --task task1_dimm_insert \
  --algo rsl_rl_ppo_plain \
  --config configs/experiments/task1_rsl_rl_ppo_plain.yaml \
  --dry-run
```

Expect `train_status=scaffold_ok` and `reward_mode=plain`.

### 2) Kit short smoke (≤2 iters, headless)

**Do not start if Isaac GUI is already using the GPU (e.g. Task2 teleop).**

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd ~/workspace_pyx/LNP_Locomotion-main/IsaacLab
./isaaclab.sh -p /mnt/data2/wjy_SERL/Embodied_Safety/runners/kit_train_plain.py \
  --headless --num_envs 4 --max_iterations 2
```

### 3) Kit long train

```bash
./isaaclab.sh -p /mnt/data2/wjy_SERL/Embodied_Safety/runners/kit_train_plain.py \
  --headless --num_envs 64 --max_iterations 2000 --allow-long-run
```

## vs Stage 1A

| | Plain (this) | Stage 1A |
|--|--------------|----------|
| Force soft penalty | off | on (`w_force_penalty=0.1`) |
| Overforce terminate | off | on |
| Success bonus | on (`w_success`) | none (success only terminates) |
| Entry | `kit_train_plain.py` / `rsl_rl_ppo_plain` | `kit_train.py` / `rsl_rl_ppo` |


## 2026-10-25 follow-up (longer + stronger early align)

After the first 2000-iter kit run (`train_20261025_215327`) ended with
`success_fracpprox0`, the plain trainer was lengthened and early align shaping
strengthened — **still no force_penalty / no overforce terminate**:

| Knob | Was | Now |
|------|-----|-----|
| `max_iterations` | 2000 | **10000** |
| `w_align` | 1.0 | **2.5** |
| `w_approach` | 1.0 | **1.5** |
| `near_align_xy_m` | 0.03 | **0.04** |

Resume: `runners/kit_train_plain.py --checkpoint <model_*.pt>` (loads weights
via `Runner.load` before `learn`). See
`outputs/runs/task1_dimm_insert/rsl_rl_ppo/task1_plain_ppo/NOTE_LONGER_ALIGN_*.md`.
