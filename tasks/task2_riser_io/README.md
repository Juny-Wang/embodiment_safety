# task2_riser_io

Franka **wire terminal (Phase 1) then M4 screw fasten ×4 (Phase 2)** from `franka_sim/outputs/T2`.

## Quick start (teleop)

See **`scene/LAUNCH.md`**.

Open: `scene/riser_io_preview.usda`  
Script: `scene/launch_panel.py` / `scene/keyboard_panel.py`

## Procedure

Authoritative: **`TASK_FLOW.md`** (Phase1 wire → Phase2 screw). Pure screw-only is obsolete.

## Control / safety (ported from Task1 patterns)

| File | Role |
|------|------|
| `scene/franka_teleop.py` | Rate-limited IK drive targets |
| `scene/keyboard_input.py` | Key → jog vectors |
| `scene/keyboard_panel.py` | UI + enable + StepFSM + alarms |
| `scene/step_fsm.py` | Phase1+2 step ids + traffic light gates |
| `scene/safety_core.py` | Shared SafetyEngine |
| `scene/plan_d_runtime.py` | Session hooks + color alarm |
| `scene/safety_config.json` | Paths, control speeds, force rules |
| `scene/geometry_calib.json` | Stub hole layout / soft align tol |

## Still missing for train

DirectRLEnv / obs-reward / calibrated frames / true screw torque — see LAUNCH.md.
