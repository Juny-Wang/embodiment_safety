# Latch self-open + grasp zone fix (2026-10-23 Asia/Shanghai)

## Bug 1 — latches open by themselves at start

### Root cause
1. Socket asset and Plan D session both used **viscous-only** hinge drive: `stiffness=0`, light damping, small `maxForce`, authored as *not a closed-position hold*.
2. `targetPosition=0` with **k=0** does not hold closed — empty latches never restore.
3. `manual_franka_scene.usda` latch overlays baked non-zero `physics:angularVelocity` / `velocity` from a previous Play. On next Play those kicks coast open; with k=0 they stay open.
4. Gravity COM torque is tiny (~1e-5 Nm) — not primary opener, but prevents return once open.

### Fix (finger still openable)
- Soft closed hold: `latch_stiffness=0.08`, `latch_damping=0.02`, `latch_max_force=0.06` (Nm), `targetPosition=0`.
- On `apply_session_plan_d`: zero latch lin/ang velocity, snap orient to identity (closed).
- USDA overlays: zero residual velocities + closed orient.
- Finger tip ~1–3 N at ~15 mm → ~0.015–0.045 Nm still opens against soft hold / maxForce cap.

## Bug 2 — grasp forbidden zone too large (edge grasp drops)

### Root cause
`forbidden_contact` is true if contact is **in forbidden OR not in allowed**. Old allow only z in [0.018, 0.0325]; mid/edge PCB grasps were flagged forbidden -> unstable / drop. Old gold forbid up to z=0.01 ate lower PCB edge.

### Calib change (train ROI preserved)
| box | before | after |
|-----|--------|-------|
| allow | x+-0.055, y+-0.0035, z 0.018-0.0325 | x+-0.062, y+-0.004, z **0.010-0.0325** |
| forbid gold | z -0.001-**0.01** | z -0.001-**0.008** |

2 mm buffer between forbid top and allow bottom. Gold fingers remain forbidden.

## Reload
```
exec(compile(open("/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_manual_safety_v002/keyboard_panel.py", encoding="utf-8").read(), "keyboard_panel.py", "exec"))
```
Play -> ENABLE. Empty latches stay closed; finger can open; PCB edge grasp should not trip forbidden.

## Backup
`.bak_latch_grasp_*` under the package dir.
