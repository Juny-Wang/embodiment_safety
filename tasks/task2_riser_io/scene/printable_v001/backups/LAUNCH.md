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
