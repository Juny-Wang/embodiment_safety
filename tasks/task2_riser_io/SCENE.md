# Scene pack (task2_riser_io)

## Status (2026-10-24 Asia/Shanghai)

**Real slim copy** from franka_sim **T2**. **Not** pdb_c6320p.

| Item | Path |
|------|------|
| Mother USD | `scene/riser_io_preview.usda` |
| **Authoritative procedure** | **`TASK_FLOW.md` — Phase 1 wire → Phase 2 screw ×4** |
| Phase2 proxies | `scene/assets/task2_wire/usd/phase2/` (S2/S3/S4, M4) |
| Report | `scene/PHASE2_PROXY_REPORT.json` |
| Backup T2 | `/mnt/data2/wjy_SERL/franka_sim/outputs/T2/` |

**Not train-ready** (no StepFSM runtime / env yet). Pure “Task S only” is **wrong**.

### User intent (combined)

1. **Phase 1 接电线:** four wires / ring terminals → seat on terminals/studs.
2. **Phase 2 拧螺丝 (after wires):** PCB holes + nut plate + tray screws → pick → align → fasten ×4.

### Mother snapshot

- Phase 1: Robot, Table, wires/rings/TE, `PCB_4_Terminal` studs, `Task1PCB`, `PCB_holder` (P1–P6)
- Phase 2 proxies **added 20261024_174724**: `/NutPlate_M4` (S2), `/Screw_1..4` (S3), `/ScrewTray` (S4); `/ResetTray` alias → ScrewTray
- Hole layout from `pcb_4_terminal` MountHole: pitch **120×60 mm**, default **M4**
- Deferred: riser IO-card insert pack

### Open in Isaac

`/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/riser_io_preview.usda`

### Teleop / safety

See **`scene/LAUNCH.md`**. Keyboard panel: `scene/keyboard_panel.py` (Task1-pattern, Task2 StepFSM).
