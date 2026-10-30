# Stage 1A wiring — Isaac Lab × RSL-RL PPO

Root: `/mnt/data2/wjy_SERL/Embodied_Safety`

## What is installed on isaac-b (discovered 2026-10-24)

| Component | Location / version | Notes |
|-----------|--------------------|-------|
| Conda env | `env_isaaclab` (`~/anaconda3/envs/env_isaaclab`) | Python 3.10.20 |
| `rsl-rl-lib` | **5.0.1** (site-packages) | `OnPolicyRunner` OK without Kit; cfg uses **actor/critic MLPModel** (not legacy ActorCritic) |
| `isaaclab` | **0.54.3** editable → `~/workspace_pyx/LNP_Locomotion-main/IsaacLab/source/isaaclab` | Importing `DirectRLEnv` needs **`pxr`** |
| Isaac Sim 4.5 | `/mnt/data2/workspace_pyx/isaacsim-4.5.0` | Linked as LNP IsaacLab `_isaac_sim` |
| Isaac Sim 5.0 | `/mnt/data2/isaacsim-5.0.0` | Standalone Kit tree |
| IsaacLab5 | `/mnt/data2/workspace_pyx/ASAP-code/IsaacLab5` (VERSION 2.3.2) | Separate checkout |

**Important:** `conda activate env_isaaclab` alone is **not** enough for Lab envs — `pxr` comes from Isaac Sim's Kit runtime (AppLauncher / `isaaclab.sh` / kit python). Plain `python -c "from isaaclab.envs import DirectRLEnv"` fails with `No module named 'pxr'`.

## Contracts

- Obs **30** / Act **7** — `tasks/task1_dimm_insert/env/OBS_SPEC.md`, `make_env.py`
- Reward / terminate — `REWARD_SPEC.md` + `scene/safety_config.json`
- Teleop USD (human demo, **not** Lab clone) — `tasks/task1_dimm_insert/scene/manual_franka_scene.usda`
- Lab DirectRLEnv — `tasks/task1_dimm_insert/env/direct_env.py` + `direct_env_cfg.py`

## Lab scene strategy (asset paths)

## Lab scene strategy (asset paths)

Teleop USDA cannot be multi-env-cloned as-is (absolute deps, GUI stage). Stage 1A DirectRLEnv builds:

| Asset | Source | Notes |
|-------|--------|-------|
| Franka Panda | Local cache if present: `franka_sim/outputs/T1_rl/franka_latch_open_test_v1/robot_cache/.../panda_instanceable.usd`; else Nucleus | `resolve_franka_usd_path()` |
| DIMM | `tasks/.../env/assets/lab_dimm_dynamic.usda` → `franka_sim/assets/task1_pcb_insert/usd/DIMM/2308107_4_matched_work/physics_contact_bench_v1/assets/dimm_2308107_matched_physics_proxy_pending.usda` | Same physics proxy as teleop `MatchedDIMM` |
| Socket | `tasks/.../env/assets/lab_socket_articulation.usda` (self-contained) | **Articulation**: Base + LeftHinge/RightHinge (±40°); `RootFixed` world pin; `fix_root_link=False` |
| Latches | Socket artic `LeftHinge` / `RightHinge` joint_pos [rad] | OBS indices 22–23; init open15 (±15°); legacy kinematic wrap kept as fallback |
| Contacts | DIMM→Socket; Latch→Robot/DIMM; Finger→DIMM | OBS `[axial, lateral, Lpress, Rpress, fingerL, fingerR]` |

Gym id: `Embodied-Task1-DimmInsert-Direct-v0` (`register_gym.py`).
OBS dim **30** unchanged.

## Commands

Always from repo root with `PYTHONPATH=.` (Kit scripts also `chdir` there).

### 1) FakeEnv smoke (no Isaac, default)

```bash
cd /mnt/data2/wjy_SERL/Embodied_Safety
PYTHONPATH=. python -m runners.train \
  --task task1_dimm_insert \
  --algo rsl_rl_ppo \
  --config configs/experiments/task1_rsl_rl_ppo_stage1a.yaml \
  --dry-run
```

Expect `train_status=scaffold_ok`.

### 2) Lab stub path (probe + optional OnPolicyRunner construct on FakeVecEnv)

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /mnt/data2/wjy_SERL/Embodied_Safety
PYTHONPATH=. python -m runners.train \
  --task task1_dimm_insert \
  --algo rsl_rl_ppo \
  --config configs/experiments/task1_rsl_rl_ppo_stage1a.yaml \
  --backend isaac_lab
```

Expect `train_status=lab_stub_ok` (no Kit → no DirectRLEnv). Prints stack probe and may construct `OnPolicyRunner` on CPU `FakeVecEnv`. **Does not** call `learn()` and **does not** open a GUI.

### 3) Kit short smoke (DirectRLEnv + optional learn ≤2 iters, headless)

Preferred (LNP IsaacLab 0.54.3 + linked Sim 4.5):

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd ~/workspace_pyx/LNP_Locomotion-main/IsaacLab
./isaaclab.sh -p /mnt/data2/wjy_SERL/Embodied_Safety/runners/kit_smoke.py \
  --headless --num_envs 2 --max_iterations 2
```

Alternate (Sim 4.5 kit python — ensure `PYTHONPATH` includes Embodied_Safety + IsaacLab source):

```bash
export PYTHONPATH=/mnt/data2/wjy_SERL/Embodied_Safety:$PYTHONPATH
/mnt/data2/workspace_pyx/isaacsim-4.5.0/python.sh \
  /mnt/data2/wjy_SERL/Embodied_Safety/runners/kit_smoke.py \
  --headless --num_envs 4 --max_iterations 2
```

Expect `kit_learn_smoke_ok` or `kit_construct_ok`. Logs under:

`outputs/runs/task1_dimm_insert/rsl_rl_ppo/stage1a/kit_smoke_<timestamp>/`

`--no-learn` skips `OnPolicyRunner.learn()` (construct + one env step only).

### 4) Longer train (only when intentional)

Use kit_smoke with higher `--max_iterations` **or** wire `runners.train --backend isaac_lab --do-learn` inside an AppLauncher process, and set `allow_long_run` / `max_iterations<=5` guard awareness in `algos/rsl_rl_ppo/train.py`.

## Code map

| Path | Role |
|------|------|
| `tasks/.../env/make_env.py` | FakeEnv + `make_env` router |
| `tasks/.../env/lab_env.py` | Lab cfg, probe, stub / Direct handle, FakeVecEnv |
| `tasks/.../env/direct_env_cfg.py` | `Task1DimmInsertEnvCfg` (Franka + cuboids) |
| `tasks/.../env/direct_env.py` | `Task1DimmInsertEnv` DirectRLEnv |
| `tasks/.../env/register_gym.py` | Gymnasium id registration |
| `runners/kit_smoke.py` | AppLauncher headless short smoke |
| `algos/rsl_rl_ppo/runner.py` | `build_rsl_train_cfg` + `OnPolicyRunner` wrap |
| `algos/rsl_rl_ppo/train.py` | Fake dry-run vs Lab stub vs learn guard |
| `algos/rsl_rl_ppo/cfg/stage1a.yaml` | Hyperparams + `output_dir` |
| `runners/train.py` | `--dry-run` / `--backend` / `--do-learn` |

## Blockers / known gaps

1. **Latch Articulation + contact filters wired** (hinges + axial/lateral/press/finger → OBS).
2. **Real DIMM + socket artic** wired (Lab wraps).
3. **`pxr` / Kit** required to instantiate DirectRLEnv; plain conda → stub only.
4. Default configs **refuse** long `learn()` unless `allow_long_run` or `max_iterations<=5` (kit_smoke caps at 2).
5. If Nucleus is offline and local Franka cache missing, robot spawn fails — check `resolve_franka_usd_path()`.

`franka_sim/t1_rl` is intentionally unused.

## Kit smoke attempts (2026-10-24 Asia/Shanghai)

### Attempt 1 (earlier, GPU busy)
GUI Isaac Sim 5.0 still on GPU (~3.2 GB). Scene init then hang; learn() not reached.

### Attempt 2 (11:44 Asia/Shanghai) — **FAIL** (hang after reset)
GPU was free (only Xorg/gnome/cursor). Command:

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
export PYTHONPATH=/mnt/data2/wjy_SERL/Embodied_Safety:$PYTHONPATH
cd ~/workspace_pyx/LNP_Locomotion-main/IsaacLab
PYTHONUNBUFFERED=1 ./isaaclab.sh -p /mnt/data2/wjy_SERL/Embodied_Safety/runners/kit_smoke.py \
  --headless --num_envs 2 --max_iterations 2
```

**Succeeded:**
- AppLauncher headless Kit **4.5** OK (`cuda:0`)
- Gym id `Embodied-Task1-DimmInsert-Direct-v0`
- Local Franka USD resolved:
  `franka_sim/outputs/T1_rl/.../robot_cache/.../panda_instanceable.usd`
- Scene create ~1.1s, sim start ~0.45s, 2 envs
- `env.reset()` returned (`reset type=<class  ' tuple ' >`)

**Failed / hung:**
- Stuck on first `env.step(zeros)` in `kit_smoke.py` (never printed `step ok`)
- No further stdout for **>5 min** after scene init (log mtime frozen at 11:44:42)
- Python held ~1177 MiB GPU; killed by policy (TERM then KILL) after hang timeout
- `learn()` / `OnPolicyRunner` **never reached**
- Not GPU contention this time (GUI was closed)

Log: `/tmp/embodied_kit_smoke/kit_smoke_20261024_114435.log`

**Likely next debug:** Instrument `_apply_action` / DifferentialIK / PhysX step in `direct_env.py`; try `--no-learn` with a timeout around `env.step`; check Jacobian body index / joint targets for Franka cuboid scene.

Status: **kit_smoke short learn = FAIL** (construct+reset OK, first step hang).

## Debug note (2026-10-24) — first env.step "hang"

**Root cause (not IK / not PhysX):** `kit_smoke.py` used `env.num_actions`, but `DirectRLEnv` has **no** `num_actions` attribute (only `single_action_space` / wrapper sets it). That raised `AttributeError` right after `reset()`; `simulation_app.close()` in `finally` then spun infinite `sim.render` (looked like a step hang). Also `sim._has_gui` was left as `None` under headless.

**Fix:**
- Resolve act dim via `gym.spaces.flatdim(env.single_action_space)` or `ACT_DIM`
- Force `env.sim._has_gui = False` when None; `cfg.wait_for_textures = False`
- Default `control_mode=joint` (diff_ik still available via `--control-mode diff_ik`)
- `--no-learn` timed step smoke + crumb file `/tmp/embodied_kit_smoke/crumbs.txt`

**Verified:** `kit_step_smoke_ok` (1 env, 3 steps) and `kit_learn_smoke_ok` (2 env, 2 iters) headless.

## Real-asset wiring (2026-10-24 12:24 Asia/Shanghai)

Replaced cuboid DIMM/Socket with physics proxies used by teleop/train USDA.

**Paths used**
- DIMM source: `/mnt/data2/wjy_SERL/franka_sim/assets/task1_pcb_insert/usd/DIMM/2308107_4_matched_work/physics_contact_bench_v1/assets/dimm_2308107_matched_physics_proxy_pending.usda`
- Socket source: `.../socket_2308107_4_physics_proxy_pending.usda` (Base only → Lab wrap)
- Lab wraps: `Embodied_Safety/tasks/task1_dimm_insert/env/assets/lab_dimm_dynamic.usda`, `lab_socket_kinematic.usda`
- Teleop refs (not cloned): `tasks/task1_dimm_insert/scene/manual_franka_scene.usda` MatchedDIMM/MatchedSocket; train: `franka_sim/outputs/T1_rl/t1_dimm_insert_train.usda`

**OBS / control**
- OBS dim 30 kept; latch = heuristic; contact = ContactSensor net force when available
- Default `control_mode=joint`

**Verify commands**
```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
export PYTHONPATH=/mnt/data2/wjy_SERL/Embodied_Safety:$PYTHONPATH
cd ~/workspace_pyx/LNP_Locomotion-main/IsaacLab
./isaaclab.sh -p /mnt/data2/wjy_SERL/Embodied_Safety/runners/kit_smoke.py \
  --headless --num_envs 1 --no-learn --num-steps 3
# then if stable:
./isaaclab.sh -p /mnt/data2/wjy_SERL/Embodied_Safety/runners/kit_smoke.py \
  --headless --num_envs 2 --max_iterations 1
```

Smoke results recorded below.
### Smoke results (2026-10-24 12:28 Asia/Shanghai)

| Run | Command | Result |
|-----|---------|--------|
| step | `--headless --num_envs 1 --no-learn --num-steps 3` | **kit_step_smoke_ok** (~1.8s build, steps 0.07/0.04/0.01s) |
| learn | `--headless --num_envs 2 --max_iterations 1` | **kit_learn_smoke_ok** (48 steps, ~93 sps, OnPolicyRunner.learn OK) |

Logs:
- `/tmp/embodied_kit_smoke/kit_step_real_20261024_122442.log`
- `/tmp/embodied_kit_smoke/kit_learn_real_20261024_122600.log`
- learn log_dir: `outputs/runs/task1_dimm_insert/rsl_rl_ppo/stage1a/kit_smoke_20261024_122607`

Verified at init:
- ContactSensor `dimm_contact` registered (`contact_ok=True`)
- `latch_source=heuristic_insert_depth`
- OBS 30 / ACT 7 / `control_mode=joint`
- Real USD paths printed (lab_dimm_dynamic / lab_socket_kinematic)

**Still deferred:** latch hinge Articulation (raw multi-body socket); finger contact filters; long train.




## Latch Articulation wiring (2026-10-24 13:00 Asia/Shanghai)

Replaced heuristic latch angles with real hinge DOFs.

**Representation**
- Lab wrap: `tasks/task1_dimm_insert/env/assets/lab_socket_articulation.usda`
  - `/Asset` — `PhysicsArticulationRootAPI` only (no RigidBody)
  - `/Asset/Base` — dynamic root link (physics_proxy Base cube collisions)
  - `/Asset/LeftLatch`, `/Asset/RightLatch` — dynamic latch bodies (proxy hook cubes)
  - `/Asset/Joints/RootFixed` — `PhysicsFixedJoint` Base→world (Lab `fix_root_link` avoided)
  - `/Asset/Joints/LeftHinge` — Y revolute, limits **−40…0°** (v008)
  - `/Asset/Joints/RightHinge` — Y revolute, limits **0…40°** (v008)
- Teleop reference (not spawned as Lab multi-env): `franka_sim/outputs/T1_rl/socket_franka_install_v008/socket_print_v007_installed.usda`
- Tradeoff: proxy-cube collisions (smoke-light) + v008 hinge kinematics; full v008 SDF latch meshes deferred if needed for contact fidelity.

**OBS / env**
- OBS dim **30** unchanged; indices 22–23 = latch angles [rad]
- `latch_source=articulation_LeftHinge_RightHinge`; `latch_ids` resolve to `LeftHinge`,`RightHinge`
- Init joint_pos open15 (±15° rad); ImplicitActuator stiffness=0, damping=0.001
- Socket registered as `scene.articulations["socket"]` (not rigid_objects)

**Smoke (2026-10-24 13:00 Asia/Shanghai)**
| Run | Result | Log |
|-----|--------|-----|
| step 1env×3 | kit_step_smoke_ok | `/tmp/embodied_kit_smoke/kit_step_latch_20261024_1301.log` |
| learn 2env×1iter | **kit_learn_smoke_ok** (48 steps, ~103 sps) | `/tmp/embodied_kit_smoke/kit_learn_latch_20261024_1301.log` |

**Still deferred:** latch press / finger contact filters; v008 SDF latch meshes; long train; thresh-4 baseline.


## Finger / latch-press contact wiring (2026-10-24 13:03 Asia/Shanghai)

Filled OBS `contact_forces` slots 26–29 (indices 24–29 total block).

**Sensors** (`direct_env_cfg.py` / `_setup_scene`)
- `dimm_contact` — `/…/DIMM` filter Socket → axial/lateral
- `latch_l_contact` / `latch_r_contact` — `/…/Socket/{Left,Right}Latch` filter Robot+DIMM → press
- `finger_l_contact` / `finger_r_contact` — `/…/Robot/panda_{left,right}finger` filter DIMM
- Franka: `activate_contact_sensors=True`

**OBS dim 30** unchanged.

**Smokes**
| Run | Log |
|-----|-----|
| step 1env×3 | `/tmp/embodied_kit_smoke/kit_step_finger_20261024_130322.log` |
| learn 2env×1 | `/tmp/embodied_kit_smoke/kit_learn_finger_20261024_130322.log` |

**Still deferred:** long train; thresh-4 baseline; v008 SDF latch meshes.
## Stage 1A first long train (2026-10-24 13:06 Asia/Shanghai)

Launcher: `runners/kit_train.py` (AppLauncher + DirectRLEnv; `--allow-long-run` required when iters>5).
Config: `algos/rsl_rl_ppo/cfg/stage1a.yaml` as designed — **num_envs=64**, **max_iterations=2000**, save_interval=50, num_steps_per_env=24, joint control, OBS 30 / ACT 7.

```bash
./isaaclab.sh -p /mnt/data2/wjy_SERL/Embodied_Safety/runners/kit_train.py \
  --headless --allow-long-run --num_envs 64 --max_iterations 2000 \
  --algo-cfg /mnt/data2/wjy_SERL/Embodied_Safety/algos/rsl_rl_ppo/cfg/stage1a.yaml
# or: bash /tmp/run_kit_train_stage1a.sh  (nohup; log under /tmp/embodied_kit_smoke/)
```

Run dir: `outputs/runs/task1_dimm_insert/rsl_rl_ppo/stage1a/train_20261024_130656`
Console log: `/tmp/embodied_kit_smoke/kit_train_stage1a_20261024_130650.log`
Contacts: all 5 sensors registered; latch Articulation kept.

**Still deferred:** thresh-4 baseline; v008 SDF latch meshes.

## Long train finished (2026-10-24 13:23 Asia/Shanghai)

Status: **kit_train_ok** (2000 iters, ~965s learn). Final mean episode reward ~-84.7; sps ~3.4k. Checkpoints: model_0..model_2000 every 50 under train_20261024_130656.

**Still deferred:** thresh-4 baseline; v008 SDF latch meshes.
