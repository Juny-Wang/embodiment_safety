# Breakable neck (T17 crimp proxy) — hand test

**Asia/Shanghai · BREAKABLE_NECK_20261025**

## What it is
Orange short capsule (`/{wire1..4}/_0347300_ring_terminal/BreakableNeck`) between ring terminal and distal cable (`Segment_23`).
Exceed axial stress proxy → yellow → joint break → red + visible gap.

## Thresholds (`safety_config.json` → `breakable_neck`)
| Key | Default | Meaning |
|-----|---------|---------|
| `F_warn_N` | 3.0 | Yellow neck (risk, not yet failed) |
| `F_break_N` | 8.0 | PhysX `physics:breakForce` + software backup |
| `break_hold_frames` | 5 | Software break must hold this many ticks |
| `proxy_stiffness_N_per_m` | 200 | Stretch×k force proxy (**DEMO — calibrate**) |
| `rest_len_m` | 0.023 | Nominal ring↔Segment_23 gap |

> Placeholders only. Hand-pull with teleop until yellow appears at “feels tight”, red/break at “would damage crimp”. Then write real numbers back into `safety_config.json`.

## Reload (script-only — no USDA Reopen required for neck/screw freeze)
1. Keep Isaac open on `riser_io_wire_guides_v001.usda` (Play OK).
2. Script Editor:
```python
exec(compile(open("/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/launch_panel.py").read(), "launch_panel.py", "exec"))
```
3. Console should show `BREAKABLE_NECK_20261025` and `SCREW_RESEAT_20261025`.
4. Confirm 4 orange necks at ring exits + 4 screws seated in light-blue tray.
5. START TELEOP → pull a ring hard along the cable → neck yellow then snap red / separate.

## Screw note
`/Screw_1..4` already existed at nest poses; empty bottom-right hole was **Screw_4** falling after Play (dynamic RB). Panel load now reseats all four and sets `kinematicEnabled=true` so they stay in pockets.

## Residual risks
- Force is stretch-proxy, not true crimp stress / Mini40.
- PhysX may ignore `breakForce` on some FixedJoint builds → software disable is the backup.
- Kinematic screws will not react to gripper until you clear kinematic on grasp (not wired yet).
- FIX4 keyboard InputEvent unpack left untouched.
