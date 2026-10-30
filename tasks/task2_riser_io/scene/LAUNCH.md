# Printable v001 scene update

The same riser_io_preview.usda now contains the custom coarse-thread printable package. Select object index 1–4 in the panel. Threshold policy is warning-only; torque and cable tension are unavailable. Geometry passes offline checks; Kit teleop/fastening not accepted. Print files and preview images: printable_v001/. See PRINT_AND_ASSEMBLE.md.

# Task2 launch (keyboard teleop + safety skeleton)

**Asia/Shanghai docs updated 20261024_175349.** Not full PPO. Goal: open scene + drive Franka like Task1 patterns.

## Open scene

In Isaac Sim 5.x:

`/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/riser_io_preview.usda`

Keep Phase1 wires and Phase2 proxies (`NutPlate_M4`, `ScrewTray`, `Screw_1..4`).

## Start keyboard + StepFSM + color-alarm skeleton

1. Open the USDA above.
2. **Pause** timeline.
3. `Window → Script Editor`, run:

```python
exec(compile(open("/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/launch_panel.py").read(), "launch_panel.py", "exec"))
```

Or:

```python
exec(compile(open("/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/keyboard_panel.py").read(), "keyboard_panel.py", "exec"))
```

4. **Play**, wait for settle (~2 physics frames).
5. Panel: **INITIALIZE** → **ACK thresholds** → **ENABLE**.
6. Click the viewport so Carb keyboard focus is on Isaac.
7. Keys (same idea as Task1):
   - Arrows: world X/Y; PageUp/PageDown: Z
   - Shift+those: world-axis rotation
   - `[` / `O` open gripper; `]` / `P` close
   - Esc: e-stop + pause

Step combo follows `TASK_FLOW.md` Phase1 then Phase2 ids (`home` … `repeat_screw_or_done`).

Logs: `scene/runs/<timestamp>/`.

## What this gives you

| Piece | Status |
|-------|--------|
| Openable mother USD | Yes |
| Franka keyboard drive (`franka_teleop` + `keyboard_input`) | Yes (ported) |
| StepFSM Phase1+2 | Yes (`step_fsm.py`) |
| SafetyEngine + traffic / RED latch | Yes (`safety_core` + rules in `safety_config.json`) |
| Force → displayColor alarm skeleton | Yes (`plan_d_runtime.apply_task2_alarms`) |
| Contact metrics | Coarse contact-report estimates (not wrist F/T) |
| Cameras / grasp diag / baseline CSV | Stubs only |
| PPO / DirectRLEnv | **Not yet** |

## Still blocking a real train env

1. No `env/make_env.py` / Isaac Lab `DirectRLEnv` for Task2
2. No calibrated EE↔hole / ring↔stud frames (geometry_calib is stub)
3. No true threaded screw dynamics / wrench sensing
4. Contact forces are impulse heuristics, not Mini40
5. Reward / obs spec not written for Phase1+2
6. Wire joint force not yet read from articulation drives into metrics
7. Full Kit teleop acceptance not run this pass (file port first)

## GPU note

Long training and GUI Isaac share one GPU — avoid both at once.

## Cameras (ported from Task1)
- Dual wrist (`WristCam_L` / `WristCam_R`) + `ExternalOverview` via `safety_config.json` → `cameras` (enabled).
- `keyboard_panel` on load calls `cameras_setup.apply_session_cameras` then `camera_viewports.open_camera_split`.
- In Isaac: Viewport → Cameras, or panel button **SHOW CAM SPLIT** to recreate the three mini windows.
- Ported: 20261024_192108. External look_world retuned toward Task2 seat centroid when geometry_calib seats exist.

# FIX3 2026-10-25 (wire_guides_v001)

Current scene: `riser_io_wire_guides_v001.usda` (not only preview).

1. **Reload USDA** (File → Reopen) so baked WristCam_L/R + ExternalOverview + wire3/4 ring joint fix load.
2. Play timeline.
3. Script Editor:
```python
exec(compile(open("/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/launch_panel.py").read(), "launch_panel.py", "exec"))
```
4. Console must show `FIX3_TELEOP_20261025`.
5. Click **▶ START TELEOP**, click **MAIN viewport**, arrows/PgUp/Dn.
6. Wrist L/R should show gripper; if not, click **SHOW CAM SPLIT**.

Wire tip: wire3 (2nd from +X/right) FixedJoint localRot0 aligned to ring; wire4 roll-180 also fixed.
Teleop: tracking envelope 0.10 m + velocity goal (Task1-style); old 3 mm stop was killing motion.

# FIX4 2026-10-25 input unpack
Root cause of events=0 with READY/enabled=True: Task2 `_on_input` treated `subscribe_to_input_events` payload as KeyboardEvent; Kit delivers InputEvent with `.deviceType` + `.event`. Ported Task1 unpack + optional `subscribe_to_keyboard_events`. Reload: re-run launch_panel.py only (no USDA reopen). Console must show FIX4_INPUT_20261025; keys_label events= must rise.


# BREAKABLE_NECK + SCREW_RESEAT 2026-10-25

Orange crimp-neck proxy + Screw_1..4 reseat/kinematic freeze.

- Details: `BREAKABLE_NECK.md`
- On `launch_panel.py` reload: console `BREAKABLE_NECK_20261025` / `SCREW_RESEAT_20261025`
- Thresholds: `safety_config.json` → `breakable_neck` (`F_warn_N=3`, `F_break_N=8` placeholders)
- USDA `physics:breakForce` on `/{wire}/FixedJoint` set to 8 N (was inf). **Reopen USDA** if Isaac still has old infinite breakForce in memory; necks/screw reseat work from Script Editor alone.
