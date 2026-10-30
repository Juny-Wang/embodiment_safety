# Scene pack (task1_dimm_insert)

## Status (2026-10-24)

**Real copy** of the final v002 scene into this task tree (not a symlink).

| Item | Path |
|------|------|
| Task scene (authoritative for Embodied_Safety) | `/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task1_dimm_insert/scene/` |
| Original backup (untouched) | `/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_manual_safety_v002/` |

Verify:

```bash
test ! -L /mnt/data2/wjy_SERL/Embodied_Safety/tasks/task1_dimm_insert/scene && echo NOT_SYMLINK
test -f /mnt/data2/wjy_SERL/Embodied_Safety/tasks/task1_dimm_insert/scene/keyboard_panel.py && echo KP_OK
du -sh /mnt/data2/wjy_SERL/Embodied_Safety/tasks/task1_dimm_insert/scene
```

Use the **Embodied_Safety** path for new work. The franka_sim copy remains as backup.

## Slim package (RL-ready)

Teleop dump tree `runs/`, shader/`runtime/cache`, and top-level validation logs/csv dumps were **stripped** from this Embodied_Safety copy so the package stays small for opening the stage and wiring RL.

- Kept: `manual_franka_scene.usda`, Python control/camera/safety scripts, `safety_config.json` / calib configs, meshes/assets referenced by the stage.
- Full historical teleop logs (if needed) remain only under the franka_sim backup: `franka_sim/outputs/T1_rl/franka_manual_safety_v002/runs/`.

Size after slim (2026-10-24): about **1.8M** (was ~22G).
