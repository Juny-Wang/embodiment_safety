# Reward diagnosis — Stage 1A train_20261024_130656

**When:** 2026-10-24 13:24 Asia/Shanghai (offline; no Kit / no GPU train)  
**Run:** `outputs/runs/task1_dimm_insert/rsl_rl_ppo/stage1a/train_20261024_130656`  
**Log:** `/tmp/embodied_kit_smoke/kit_train_stage1a_20261024_130650.log`

## Symptom

Mean episode reward stayed ≈ **−84…−90** for 2000 iters. Logged term means (already × weights):

| Term | Early (0–50) | Mid (900–1000) | Late (1900–2000) | Last (1999) |
|------|-------------:|---------------:|-----------------:|------------:|
| approach | −0.21 | −0.22 | −0.22 | −0.22 |
| **align** | **−87.0** | **−89.1** | **−88.5** | **−85.0** |
| insert_depth | 0.34 | 0.34 | 0.35 | 0.36 |
| dual_latch | 0.10 | 0.11 | 0.10 | 0.17 |
| force_penalty | ~0 | ~0 | ~0 | ~0 |
| reward | −86.8 | −88.9 | −88.3 | −84.7 |

`|align| / |reward| ≈ 1.00` for essentially the whole run. No meaningful learning signal outside align.

## Implementation (current)

```text
approach     = −‖rel_pos‖                         # w=1
align        = −(xy_err / xy_tol) − (tilt / tilt_tol)   # xy_tol=2.5mm, tilt_tol=2°=0.035rad; w=1
insert_depth = (−rel_z).clamp(≥0)                 # w=2
dual_latch   = count(|hinge|<0.05 rad)            # w=5
force_penalty= −Σ|contact|                        # w=0.1
```

`rel_*` = DIMM pose in **socket root** frame (`subtract_frame_transforms`). Quat is Lab **wxyz**; tilt = `2 acos(|w|)`.

## Root-cause hypothesis (ranked)

### 1) Align is scaled by *success* tolerances → saturates on approach-scale errors (PRIMARY)

Dividing by **2.5 mm / 2°** turns ordinary start geometry into huge negatives:

- Lab spawn: Socket `(0.55,0,0.20)`, DIMM `(0.45,0,0.28)` → Δxy≈**0.10 m** → `xy/tol = 40` → align ≈ **−40** with tilt≈0.
- Log **iter 0** align = **−45.8** matches that (partial episodes / small tilt).
- After free motion, align settles near **−π / 0.035 ≈ −89.8** when the DIMM tumbles (~180° relative tilt), or −80…−100 with XY+tilt mix.

So align is not a gentle shaping term; it is a **near-constant large offset** set by mm-scale gates meant for *final* clearance (`geometry_calib.alignment` / `insertion_clearance`), not for dense RL from 10 cm away.

### 2) DIMM is free-falling / not in-hand (PRIMARY task setup)

- DIMM: `disable_gravity=False`, dynamic RigidObject; reset only adds ±1 cm XY noise.
- Franka starts at home joints; **no grasp constraint**. Policy joint actions barely couple to DIMM pose until accidental contact.
- Episode length mean ≈ **462 / 500** → almost always timeout; success path never exercised.
- Falling below socket makes `insert_depth = (−rel_z)+` **spuriously large** (weighted log ~0.34 ⇒ raw depth ~0.17 m ≫ `seat_depth_m=0.01` / `success_depth_m=0.012`) without being seated or aligned.

### 3) Lab spawn orientation ≠ teleop calibrated world pose (SECONDARY)

`geometry_calib.json` live snapshot:

- Socket orient_wxyz ≈ `[0.707, 0, 0, −0.707]` (−90° about Z)
- DIMM orient_wxyz ≈ same rotation (sign-equivalent)

Lab cfg sets **both** `rot=(1,0,0,0)`. Relative tilt at t=0 is fine (identity), but **world** socket facing vs Franka reach may be wrong vs teleop; insert axis is still socket-local +Z per calib. Not the main −85 driver (spawn align already explained by XY/tol), but matters for later approach curriculum.

### 4) Other terms cannot compete

- `dual_latch`: hinges init at ±15°; closed band 0.05 rad (~3°) with no insertion → logged ~0.1 after ×5 → raw ≈0.02 (noise).
- `force_penalty` ~0 (little structured contact).
- Absolute potentials (`−dist`, absolute align) give weak Δ-improvement vs the align floor.

**Not** the main issue: PhysX GPU contact-filter warnings; OBS dim 30; latch Articulation registration (hinges present). Proxy cube geometry is coarse but secondary to scaling + free-fall.

## Evidence summary

- `π / tilt_tol = 89.76` ≈ late align magnitude when tumbling.
- Iter0 align −45.8 ≈ `−(0.10 / 0.0025)` spawn XY.
- approach stable ~−0.22 m (separation while falling).
- insert_depth stays ~0.34 without dual_latch success → depth gate missing.
- GPU idle / no Kit during this diagnosis.

## Ranked fixes (do before next long train)

1. **Rescale / reshape align (+ gate depth)** — highest leverage, code-only  
   - Stop dividing by success tols for dense reward. Prefer e.g.  
     `align = −(xy_err / 0.05) − (tilt / 0.5)` or `−tanh(xy/σ_xy) − tanh(tilt/σ_tilt)`, and **clip** to `[−5, 0]`.  
   - Log **raw** `xy_err_m`, `tilt_rad` in `extras["log"]` (not only weighted).  
   - Gate: `insert_depth_r = depth * (xy_ok & tilt_ok)` so falling below socket does not pay.

2. **Init / grasp curriculum** — required for a real insertion MDP  
   - Spawn DIMM near mouth (cm-scale XY/tilt noise) **or** fixed in gripper (disable DIMM gravity / hold joint until release).  
   - Optionally match teleop world Z rot (−90°) for both socket and DIMM so reach matches scene.  
   - Phase curriculum: approach → align → depth → latch (or weight schedule).

3. **Reweight + Δ-shaping** — after (1)(2)  
   - Lower `w_align` or raise approach/depth once scales are cm-level; use progress rewards (`−Δdist`, `−Δxy`) instead of only absolute potentials.  
   - Keep force limits as soft penalty; dual_latch bonus only when seated.

## Out of scope this turn

No code train relaunch, no thresh-4, no v008 SDF swap. Optional follow-up: implement fix (1) in `direct_env.py` / cfg + short kit_step smoke only.

## Fix#1 applied (2026-10-24 Asia/Shanghai) — code only, no Kit/train

- Dense align: `-(xy/0.05)-(tilt/0.5)` then **clip [−5, 0]** (`align_xy_scale_m`, `align_tilt_scale_rad`, `align_clip_*`).
- Success tols `xy_tol_m` / `tilt_tol_rad` unchanged for dones.
- `insert_depth` and `dual_latch` multiplied by near-align mask (`near_align_xy_m=0.03`, `near_align_tilt_rad=0.25`).
- `extras["log"]`: `xy_err`, `tilt`, `align_raw`, `near_align_frac`.
- Fix#2 prep only: cfg `dimm_gravity_off=False`, `dimm_init_in_gripper=False` (TODO — not wired in `_reset_idx` so smokes stay free-DIMM).
