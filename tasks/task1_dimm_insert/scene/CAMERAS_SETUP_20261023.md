# Cameras: dual wrist + external (2026-10-23 Asia/Shanghai) — 20261023_213226

## Deliverable
Three session-layer cameras created when `keyboard_panel.py` loads:

| name | parent | role |
|------|--------|------|
| `WristCam_L` | `/World/Robot/.../panda_hand` | left wrist, looks at grasp zone |
| `WristCam_R` | same | right wrist, looks at grasp zone |
| `ExternalOverview` | `/World/Cameras` | table / socket overview |

## How to view
1. Reload panel (Script Editor exec keyboard_panel.py).
2. Viewport → Cameras → pick `WristCam_L` / `WristCam_R` / `ExternalOverview`.
3. Console should print `[keyboard_panel] cameras {...} ok True`.

## Config
`safety_config.json` → `cameras` (offsets, focal length, look targets). Wrist cams follow the hand; external is world-fixed (retune `look_world` if Socket moves a lot).

## Out of scope this round
RGB train obs / Replicator writers / auto grasp classifier — next after you confirm framing.

## Backup
`.bak_cameras_*` on package dir.
