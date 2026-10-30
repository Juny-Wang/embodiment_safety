# Plan D landing status (2026-10-22 Asia/Shanghai)

## Implemented
- D-A1/A2: session-layer weak viscous latch (stiffness=0, damping=0.004, maxForce=0.025)
- D-A3: latch over-force -> displayColor red + latch_overforce metric
- D-B1/B2: thin contactOffset 5e-5 on PCB/TopPress/side pads; contact-limited grip; Gold_* collision disabled (visual-only)
- geometry_calib.json: LOCAL Socket/DIMM frames from live mesh; world pose snapshot informational only
- safety_config.json: plan_d + traffic_light + align/overforce rules; grasp_regions filled from calib at runtime
- StepFSM + TrafficLight + gates in step_fsm.py; wired in keyboard_panel.py
  - not aligned -> block world -Z (down)
  - not dual-latched -> block gripper open on LATCH_LOCK/RELEASE
  - RED latch reasons block down+release until ENABLE acknowledge
- ENABLE keyboard teleop kept (FREE 180 mm/s)

## Deferred
- D-B3 thicken: only if penetration remains after thin contact (not applied)
- Full hook-engagement geometry sensor (proxy: dual closed angles + seated_ok)
- Writing moved world poses back into t1_dimm_insert_train.usda (intentionally not; local calib only)

## Reload
Panel hot-reloaded via Isaac MCP execute_script. Or Script Editor:
```
exec(compile(open("/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_manual_safety_v002/keyboard_panel.py").read(), "keyboard_panel.py", "exec"))
```
Then Play -> ENABLE KEYBOARD. Session physics re-applies on panel construct.

## Backups
*.bak_plan_d_20261022_224657 for keyboard_panel.py and safety_config.json


## 2026-10-23 latch/grasp fix
- Soft closed hold k=0.08 d=0.02 maxF=0.06 + session velocity/pose reset (was viscous-only k=0).
- USDA latch overlay angVel/vel zeroed, orient identity.
- Grasp allow z 0.018->0.010 x+-0.062; gold forbid top 0.01->0.008.
- See LATCH_GRASP_FIX_20261023.md. Backup .bak_latch_grasp_*.
