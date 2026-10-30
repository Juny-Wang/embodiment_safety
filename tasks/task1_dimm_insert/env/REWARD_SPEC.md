# Reward / termination — Stage 1A

Dense shaping toward: approach → align → insert depth → dual latch close.
Force safety from Plan D (`scene/safety_config.json`) as soft penalty and hard terminate.

## Reward terms (sum → scalar `reward`)

| Key | Sign | Intent | Formula (2026-10-24 fix#1) |
|-----|------|--------|------------------------------|
| `approach` | −dist | Reduce DIMM–socket distance | `−‖rel_pos‖` · `w_approach` |
| `align` | − | XY + tilt, **cm-scale dense** | `clip( −xy/align_xy_scale − tilt/align_tilt_scale , [align_clip_min, align_clip_max] )` · `w_align` |
| `insert_depth` | + | Progress along socket −Z when near-aligned | `insert_depth_raw * near_align` · `w_insert_depth` |
| `dual_latch` | + | Hinges in closed band when near-aligned | `count(|angle|<0.05) * near_align` · `w_dual_latch` |
| `force_penalty` | − | Soft cost on contact magnitudes | `−Σ|F|` · `w_force_penalty` |

### Dense vs success scales (do not conflate)

| Use | XY | Tilt |
|-----|----|------|
| **Dense align** (`align_*_scale`) | `align_xy_scale_m=0.05` (5 cm) | `align_tilt_scale_rad=0.5` |
| **Clip** | `align_clip_min=-5`, `align_clip_max=0` | same |
| **Near-align gate** (depth/latch bonus) | `near_align_xy_m=0.03` | `near_align_tilt_rad=0.25` |
| **Success / terminate** (`xy_tol` / `tilt_tol`) | `xy_tol_m=0.0025` | `tilt_tol_rad=0.035` (~2°) |

Defaults weights unchanged: `w_approach=1`, `w_align=1`, `w_insert_depth=2`, `w_dual_latch=5`, `w_force_penalty=0.1`.

TB / `extras["log"]` also records raw means: `xy_err`, `tilt`, `align_raw`, `near_align_frac`.

## Termination

| Condition | `terminated` | Notes |
|-----------|--------------|-------|
| Success: dual latch closed + seated | True | Still uses **success** tols (`xy_tol*2`, `success_depth_m`, hinge band) |
| Over-force / `latch_overforce` | True | Hard fail; mirror teleop red / overforce alarm |
| Forbidden contact / penetration over limit | True | Optional Stage 1A |
| Horizon / timeout | Truncated | `max_episode_steps` (default 500) |

## Safety reference (demo thresholds, not material limits)

From `scene/safety_config.json` → `plan_d` / `rules`:

- `latch_overforce_N`: 3.0
- `grip_force_limit_N`: 15.0
- `axial_contact_N`: 5
- `lateral_contact_N`: 2

Env loads these from the scene JSON when present (see `load_safety_limits`).

## Curriculum prep (fix#2 — not default-on)

Cfg flags (default **False**, no behavior change until a train opts in):

- `dimm_gravity_off`
- `dimm_init_in_gripper`

See `REWARD_DIAG.md`.
