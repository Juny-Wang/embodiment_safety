# Task2 `task2_riser_io` — Authoritative task procedure

**Status (2026-10-24 Asia/Shanghai):** rewritten per **user correction (Juny Wang)**.  
Wire connection is **Phase 1**. Four-screw fasten is **Phase 2**, **after** wires are done — **not** a replacement for the wire task.

**Not** train-ready. Inventory from USDA/JSON only (no Kit this pass).

| Source | Path |
|--------|------|
| Mother | `scene/riser_io_preview.usda` |
| Slim assets | `scene/assets/task2_wire/usd/` |
| Upstream library | `/mnt/data2/wjy_SERL/franka_sim/assets/task2_wire/usd/` |
| Sibling pack | `scene/assets/riser_io_card/` |
| Original T2 | `/mnt/data2/wjy_SERL/franka_sim/outputs/T2/` |

---

## 0. Intent (user-corrected — authoritative)

**Combined task = Phase 1 (接电线 / ring-on-terminal) → Phase 2 (四孔拧螺丝 ×4).**

User correction (paraphrase): the four-screw fasten is **not** replacing the wire work; it comes **after** the wire-connection phase (“那个接着接电线那个后面的”).

Chinese summary for operators:

> **先接线：** 四根软线末端环端子，分别套到端子柱/座上并确认。  
> **再拧螺丝：** PCB 四个等大通孔 + 板下螺纹板 + 置物架四颗螺丝 → 取螺丝 → 对准 → 拧紧；重复四次。

| Wrong | Right |
|-------|-------|
| **Pure “Task S only”** (screw-only FSM that drops wires) | Two-phase FSM: wires first, screws second |
| Old draft that **only** did ring seat and never screws | Phase 1 alone is incomplete for the full scene goal |
| Hard-wiring IO-card insert into the main line | Still deferred sibling (not mother) |

---

## 1. SUPERSEDED drafts (do not use as primary)

| Draft | Why wrong |
|-------|-----------|
| **Task S only** (`TASK_FLOW.md` evening rewrite that marked ring/wire “out of scope”) | User: screws come **after** wires, not instead of them |
| Early “Task A only” that never planned screws | Incomplete vs user intent for the full task |
| Serializing **IO-card insert** into the main line | `riser_io_card` not in mother; branch only |

Keep P1–P6 geometry hygiene (Task1PCB, kinematic holder, TBY21 scale, `PCB_4_Terminal` add, ring scale). Those edits stay on disk; procedure ownership is **§2–§3**.

---

## 2. Phase 1 — Wire connection (接电线)

Mother-scene wires / rings / TE as before (refined). Index `w = 1..4` (wire/terminal pair). Suggested map: `wire1→Terminal_1` … `wire4→Terminal_4` on `/PCB_4_Terminal` (fallback legacy: TBY21 studs if 4-terminal path blocked).

| # | Step ID | Intent | Expected prims / frames | Gate / accept (TBD) | Wrong if… |
|---|---------|--------|-------------------------|---------------------|-----------|
| 0 | `home` | Arm clear; gripper open | `/World/Robot` | No contact with wires/board | Start holding a ring or screw |
| 1 | `select_wire` | Choose wire `w` | `/wire{w}` | One free ring tip | Grab bundle / wrong wire |
| 2 | `grasp_ring` | Grasp **ring terminal** only | `/wire{w}/…ring…` | Grip on ring, not mid-cable | Crush cable segment |
| 3 | `lift_clear` | Lift clear of tray/cage lip | Ring of wire `w` | No scrape/drag | Drag other wires |
| 4 | `route_to_stud` | Carry ring near target stud/terminal `w` | Terminal_w or TBY21 | Soft path; respect wire joint limits | Hard yank (joint force >> baseline) |
| 5 | `align_ring_stud` | Align ring hole to stud/terminal axis | Terminal_w | XY/tilt within TBD | Start seating while tilted |
| 6 | `seat_ring` | Seat ring onto stud/terminal | Terminal_w M4 (or legacy stud) | Axial seat force soft-limited | Jam / overforce seat |
| 7 | `hold_verify` | Brief hold; confirm still seated | Ring + Terminal_w | Still on post | **Not** “tighten nut” — no nut prim |
| 8 | `release_retract` | Open gripper; retract | Ring stays on terminal | Ring remains seated | Retract while still gripping |
| 9 | `repeat_wire_or_phase2` | Next wire, or **enter Phase 2** when all 4 seated | wires 1–4 | All four rings seated | Jump to screws with wires undone |

**Phase 1 notes**

- TE connector end is anchored; cable is articulated / low joint force — do not treat as rigid pick-and-place.
- `/World/ResetTray` / empty trays are **not** Phase 1 pick sources (rings live on wire tips).
- `PCB_4_Terminal` M4 studs are the **current** seat targets after P1–P5; TBY21 kept as legacy fallback.

---

## 3. Phase 2 — Screw fasten ×4 (拧螺丝，接线之后)

Only after Phase 1 success (all four rings seated, or policy/episode configured for “screws-only eval” — default training episode is **Phase1 then Phase2**).

Ideal layout (target; S2/S3/S4 still missing — see §5):

| Element | Role |
|---------|------|
| PCB with 4 equal clearance holes | Pass screws |
| Backing / nut plate under PCB with 4 coaxial threaded holes | Receive screws |
| Parts tray / shelf with 4 screws | Pick source |
| Franka gripper (optional driver later) | Pick & turn |
| Fixture keeping PCB+plate registered | Repeatability |

| # | Step ID | Intent | Expected prims / frames | Gate / accept (TBD) | Wrong if… |
|---|---------|--------|-------------------------|---------------------|-----------|
| 10 | `select_screw` | Choose screw `i` on tray | `/World/ResetTray` or future `/PartsTray/Screw_i` | Screw free | Grab ring / wire / clutter |
| 11 | `grasp_screw` | Grasp screw head (or grasp strip) | `Screw_i` | Grip ok | Crush threads |
| 12 | `lift_from_tray` | Clear tray lip | Screw_i | No tray drag | Sweep other screws |
| 13 | `transport_to_hole` | Move above hole `i` | Hole_i on PCB; Thread_i below | Approach along hole axis | Sideways into PCB / wires |
| 14 | `align_screw_hole` | Coaxial align tip to hole `i` | Hole_i + Thread_i | XY/tilt ≤ TBD | Turn while tilted |
| 15 | `insert_start` | Lower tip through clearance into thread lead-in | PCB + nut plate | Soft axial force | Cross-thread / hard stop |
| 16 | `fasten_turn` | Drive to depth | Screw_i ↔ Thread_i | Torque/turns/depth TBD | Over-torque / strip |
| 17 | `release_verify` | Open gripper; confirm seated | Screw_i | Head flush / depth ok | Retract while gripping |
| 18 | `repeat_screw_or_done` | Next `i`, or **success** if all 4 done | Screws 1–4 | All four fastened | Claim done with empty holes |
| R | `reset` | Return unused screws; clear gripper; optional wire reset | Parts tray + wires | — | Empty tray still missing nests |

### Hole / screw indexing (convention)

2×2 grid (matches `pcb_4_terminal` MountHole spirit):

| Index | Corner | Notes |
|-------|--------|-------|
| 1 | −X −Y | `MountHole_1` / `Screw_1` |
| 2 | +X −Y | |
| 3 | −X +Y | |
| 4 | +X +Y | |

`pcb_4_terminal` MountHole positions (m, asset frame) — size/pitch **reference**: Hole1 (−0.06,−0.03) … Hole4 (+0.06,+0.03), r=**0.002**. Asset still embeds studs+nuts used in **Phase 1**; Phase 2 needs clearance + separate nut plate (S1 verify / S2).

---

## 4. Combined FSM step list (authoritative)

```
Phase 1: home → select_wire → grasp_ring → lift_clear → route_to_stud
       → align_ring_stud → seat_ring → hold_verify → release_retract
       → repeat_wire_or_phase2   (×4 wires)
Phase 2: select_screw → grasp_screw → lift_from_tray → transport_to_hole
       → align_screw_hole → insert_start → fasten_turn → release_verify
       → repeat_screw_or_done    (×4 screws)
```

Episode success = Phase 1 complete **and** Phase 2 complete (unless eval split).

---

## 5. Mother vs combined task — part inventory

### Present (Phase 1 usable / Phase 2 partial)

| Prim / asset | Phase | Comment |
|--------------|-------|---------|
| `/World/Robot`, `/World/Table` | 1+2 | Franka + bench |
| `/wire1..4` + ring terminals + TE | **1** | Primary Phase 1 |
| `/PCB_4_Terminal` M4 studs | **1** | Seat targets (also has MountHoles as size ref) |
| `/Task1PCB`, `/PCB_holder` | 1+2 | Board + nest; holes for Phase 2 **unverified** without Kit |
| `/World/ResetTray` | **2 shell** | **Empty** — no screw nests yet |
| TBY21 studs | 1 legacy | Scale fixed P3–P4; prefer PCB_4_Terminal |

### Missing for Phase 2 (required — **no meshes invented this pass**)

| ID | Part | Have? | Action |
|----|------|-------|--------|
| **S1** | PCB 4× equal clearance holes | Partial | Verify Task1PCB or author clearance-only board |
| **S2** | Backing / nut plate, 4 threaded holes coaxial | **Proxy present** (`/NutPlate_M4`) | Refine / print real |
| **S3** | Four identical graspable screws | **Proxy present** (`/Screw_1..4`) | Refine / print or scan |
| **S4** | Parts tray with 4 screw nests | **Proxy present** (`/ScrewTray`; ResetTray alias) | Refine print |
| **S5** | Registration fixture | Partial | Keep `PCB_holder` kinematic |
| **S6** | Driver bit (optional) | Missing | Optional; finger-turn OK for v0 |

Decorative `riser_fixture` Bolt_* / ThumbScrew are **not** Phase 2 tray screws.

### Out of main line (deferred)

| Branch | Status |
|--------|--------|
| Riser IO-card insert (`riser_io_card/`) | Sibling pack; not mother |
| P8–P12 (TE coupler extras, FLEXA, guide, ati) | Deferred |

---

## 6. Open path (user Isaac inspect)

```text
/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/riser_io_preview.usda
```

Check: wires/rings/TE for Phase 1; empty `ResetTray`; `PCB_4_Terminal` studs vs MountHoles; no tray screws yet for Phase 2.

---

## 7. Explicit non-claims

- Not train-ready; Phase 2 screw meshes **not** authored this pass.
- Hole sizes on `Task1PCB` not Kit-measured.
- Screw metric (M3 vs M4) undecided — prefer match S1/S2; MountHole r≈0.002 suggests ~M4 clearance.
- No Kit / no Franka control migration this doc pass.
- **Pure Task-S-only procedure is obsolete.**

---

## 8. Historical P1–P6

Geometry hygiene already applied. See `scene/P1P5_REPLACE_REPORT.json`, `scene/P6_RING_SCALE_REPORT.json`. Procedure ownership is **Phase 1 + Phase 2** above.
