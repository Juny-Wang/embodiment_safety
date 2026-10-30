# Robot scene clean (2026-10-22 Asia/Shanghai)

## What was wrong
Layer stack: `manual_franka_scene.usda` -> `t1_dimm_insert_train.usda` -> `T1/riser_io_preview.usda`.
`riser_io_preview.usda` had simulation-baked world `xformOp:translate` / `xformOp:orient` on every `panda_link*`.
Top layer swapped Franka reference but did not clear those baked link xforms -> exploded arm.

## What we fixed (file-level)
1. `/mnt/data2/wjy_SERL/franka_sim/outputs/T1/riser_io_preview.usda`
   - Robot = single local Franka ref + translate (-0.43, 0, 0.75) only
   - Removed all baked link xforms / velocities
2. `/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_manual_safety_v002/manual_franka_scene.usda`
   - Same local Franka ref + same pose
   - Kept joint home drive/state overs only (no link xforms)

Backups: `*.bak_robot_clean_20261022_183102`

## How to reopen in Isaac (IMPORTANT)
Isaac PID may still hold the OLD in-memory stage.
1. Do NOT Save the current exploded stage (would overwrite clean files).
2. File > Open either:
   - `/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/t1_dimm_insert_train.usda` (train scene)
   - or `/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_manual_safety_v002/manual_franka_scene.usda` (manual+keyboard)
3. Confirm Franka looks connected at table pose before keyboard work.

## Keyboard status
Not verified yet this turn; keyboard needs a non-exploded Robot first.
