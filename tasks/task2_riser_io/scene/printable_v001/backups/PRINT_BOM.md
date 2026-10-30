# PRINT_BOM — task2_riser_io (Phase 1 + Phase 2 proxies)

Aligned with **TASK_FLOW.md** (wire then screws). Proxy sim assets generated **20261024_174724** (default **M4**).

## Assumed hole layout (from USDA parse, no Kit)

Source: `assets/task2_wire/usd/pcb_4_terminal.usd` `MountHole_1..4`

| Index | Local XY (m) | Clearance r (asset) |
|-------|----------------|---------------------|
| 1 | (-0.06, -0.03) | 0.002 |
| 2 | (+0.06, -0.03) | 0.002 |
| 3 | (-0.06, +0.03) | 0.002 |
| 4 | (+0.06, +0.03) | 0.002 |

Pitch: **dX = 120 mm**, **dY = 60 mm**.
World: holes inherit `/PCB_4_Terminal` translate `(-0.050, -0.050, 0.750)` e.g. Hole1 world XY `(-0.110, -0.080)`.
`/NutPlate_M4` same XY, Z = `0.750 - 0.006`.

## Phase 2 printable / proxy dims (M4)

| ID | Part | Sim path | Print dims (mm) | Notes |
|----|------|----------|-----------------|-------|
| S2 | Nut plate + 4x M4 tap proxies | `scene/assets/task2_wire/usd/phase2/nut_plate_m4.usda` | Plate **140 x 80 x 6**; holes on 120x60 pitch; tap ~Ø3.3; clear guide Ø4.5 | Kinematic; not true helix |
| S3 | Screw x4 | `.../phase2/m4_screw.usda` -> `/Screw_1..4` | Shank Ø**4** x L**12**; pan head Ø**7** x H**3**; ~4 g | Dynamic RB on tray |
| S4 | Screw tray 4 nests | `.../phase2/screw_tray.usda` -> `/ScrewTray` | Outer **120 x 80 x 12**; nest Ø**12** x depth**6**; nest pitch 28 mm | World `(0.22, -0.28, 0.75)`; `/ResetTray` alias |
| S1 | PCB clearance | Partial Task1PCB + MountHole ref | Same 120x60 / Ø4-4.5 | Verify in Isaac |
| S5 | PCB holder | existing | — | Keep kinematic |

## Phase 1 (unchanged)

Wires / rings / TE left in mother.

## Open in Isaac

`/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/riser_io_preview.usda`
