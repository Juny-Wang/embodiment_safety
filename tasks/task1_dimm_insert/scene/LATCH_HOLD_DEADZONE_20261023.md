# Latch soft-hold dead-zone (2026-10-23 Asia/Shanghai) — 20261023_212025

## Bug
After soft-closed fix (k=0.08 always on), empty latches stayed closed (good) but
**intentionally opened latches were spring-pulled back closed** (bad).

## Fix
Per-frame `update_latch_soft_hold` in `plan_d_runtime.py`, called from `keyboard_panel`:

| condition | stiffness | effect |
|-----------|-----------|--------|
| \|angle\| ≤ `latch_hold_band_deg` (5°) AND press < 0.15 N | 0.08 | soft closed hold |
| \|angle\| > 5° OR finger contact ≥ 0.15 N | **0** | damping only — stays open |

Session reset (zero vel + closed orient) unchanged. USDA overlay zeroing unchanged.

## Reload
```
exec(compile(open("/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_manual_safety_v002/keyboard_panel.py", encoding="utf-8").read(), "keyboard_panel.py", "exec"))
```
Play → ENABLE. Empty stay closed; finger-open stays open; no need to Save USDA for this.

## Backup
`.bak_latch_hold_20261023_212025` on package dir.
