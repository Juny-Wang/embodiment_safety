# Observation spec — Stage 1A (low-dim state, no vision)

Source of truth for dims: `make_env.py` (`OBS_*` constants). Values are read
**directly from the simulator** (joint sensors, rigid transforms, contact
reports, latch DOFs). Cameras may record in parallel but **do not** enter the
policy in Stage 1A.

Total dim: **30** (`float32`).

| Block | Dim | Content | Sim source (conceptual) |
|-------|-----|---------|-------------------------|
| `joint_pos` | 7 | Franka arm joint positions [rad] | Articulation joint state |
| `joint_vel` | 7 | Franka arm joint velocities [rad/s] | Articulation joint state |
| `gripper` | 1 | Gripper width [m] or open fraction | Finger joints / gripper DOF |
| `dimm_socket_rel` | 7 | DIMM pose in socket frame: xyz (3) + quat wxyz (4) | Rigid transforms of `/World/MatchedDIMM` vs socket |
| `latch_angles` | 2 | Left / right latch hinge angles [rad] | Latch revolute DOFs |
| `contact_forces` | 6 | `[axial_N, lateral_N, left_press_N, right_press_N, finger_L_N, finger_R_N]` | Contact / Plan-D force reports |

## Notes

- Units align with `scene/safety_config.json` (`length=m`, `angle=rad`, `force=N`, quat `wxyz`).
- Relative pose: prefer socket-local frame so insert axis is +Z (see geometry in safety_config).
- Stage B+ may append pixels or embeddings; keep these first 30 dims stable for curriculum continuity.
- FakeEnv returns zeros of this shape for smoke tests.

## DirectRLEnv notes (2026-10-24 13:03 Asia/Shanghai, finger + latch-press contacts)

Lab env (`direct_env.py`) keeps the same **30-D** layout.

| Block | Source now |
|-------|------------|
| joints / gripper / rel pose | Franka + DIMM rigid / Socket artic root |
| `latch_angles` (idx 22–23) | Socket `LeftHinge`/`RightHinge` joint_pos [rad]; init open15 |
| `contact_forces` (idx 24–29) | See table below |

### `contact_forces[6]` wiring

| Index | Name | Sensor | Filter (thresh-2) |
|-------|------|--------|-------------------|
| 24 | `axial_N` | `dimm_contact` on DIMM | → Socket (world fz) |
| 25 | `lateral_N` | same | → Socket (hypot fx,fy) |
| 26 | `left_press_N` | `latch_l_contact` on `Socket/LeftLatch` | → Robot + DIMM (\|F\|) |
| 27 | `right_press_N` | `latch_r_contact` on `Socket/RightLatch` | → Robot + DIMM (\|F\|) |
| 28 | `finger_L_N` | `finger_l_contact` on `Robot/panda_leftfinger` | → DIMM (\|F\|) |
| 29 | `finger_R_N` | `finger_r_contact` on `Robot/panda_rightfinger` | → DIMM (\|F\|) |

Franka spawn uses `activate_contact_sensors=True`. Each sensor soft-fails independently; `contact_source` lists which parts registered (e.g. `dimm_socket+latch_L+latch_R+finger_L+finger_R`).
