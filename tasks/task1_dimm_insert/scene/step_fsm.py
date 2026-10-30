"""Plan D StepFSM + TrafficLight + advance gates. No Isaac imports."""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple

STEPS = [
    "OPEN_LATCH",
    "GRASP_DIMM",
    "EXTRACT",
    "ALIGN",
    "INSERT",
    "LATCH_LOCK",
    "RELEASE",
]

STEP_LABELS = [
    "1 OPEN_LATCH",
    "2 GRASP_DIMM",
    "3 EXTRACT",
    "4 ALIGN",
    "5 INSERT",
    "6 LATCH_LOCK",
    "7 RELEASE",
]


class TrafficLight:
    """Gray/Green/Yellow/Red with yellow/green hysteresis; RED is immediate."""

    def __init__(self, warning_ratio=0.8, clear_ratio=0.7, yellow_hold_s=0.05, green_hold_s=0.05):
        self.warning_ratio = float(warning_ratio)
        self.clear_ratio = float(clear_ratio)
        self.yellow_hold_s = float(yellow_hold_s)
        self.green_hold_s = float(green_hold_s)
        self.state: Dict[str, str] = {}
        self.hold: Dict[str, float] = {}
        self.latched_red: Dict[str, bool] = {}
        self.reasons: List[str] = []
        self.peaks: Dict[str, float] = {}

    def evaluate_rule(self, key: str, value: Optional[float], limit: Optional[float], active: bool, dt: float) -> str:
        if not active or limit is None or value is None or not math.isfinite(value) or not math.isfinite(limit) or limit <= 0:
            self.state[key] = "GRAY"
            self.hold[key] = 0.0
            return "GRAY"
        ratio = abs(value) / limit
        self.peaks[key] = max(self.peaks.get(key, 0.0), abs(value))
        prev = self.state.get(key, "GRAY")
        if ratio >= 1.0 or self.latched_red.get(key):
            self.state[key] = "RED"
            self.latched_red[key] = True
            self.hold[key] = 0.0
            if key not in self.reasons:
                self.reasons.append(key)
            return "RED"
        # Candidate color
        if ratio >= self.warning_ratio or (prev == "YELLOW" and ratio >= self.clear_ratio):
            cand = "YELLOW"
        else:
            cand = "GREEN"
        # Hysteresis hold for yellow/green only
        if cand == prev:
            self.hold[key] = self.hold.get(key, 0.0) + dt
            return prev
        need = self.yellow_hold_s if cand == "YELLOW" else self.green_hold_s
        self.hold[key] = self.hold.get(key, 0.0) + dt
        if self.hold[key] < need and prev in ("YELLOW", "GREEN"):
            return prev
        self.state[key] = cand
        self.hold[key] = 0.0
        return cand

    def clear_red(self, key: Optional[str] = None):
        if key is None:
            self.latched_red.clear()
            self.reasons.clear()
            return
        self.latched_red.pop(key, None)
        self.reasons = [r for r in self.reasons if r != key]


class StepFSM:
    def __init__(self, calib: Dict[str, Any], safety: Dict[str, Any]):
        self.calib = calib
        self.safety = safety
        self.step_index = 0
        tl = safety.get("traffic_light", {})
        self.light = TrafficLight(
            warning_ratio=tl.get("warning_ratio", safety.get("warning_ratio", 0.8)),
            clear_ratio=tl.get("warning_clear_ratio", safety.get("warning_clear_ratio", 0.7)),
            yellow_hold_s=tl.get("yellow_min_duration_s", 0.05),
            green_hold_s=tl.get("green_min_duration_s", 0.05),
        )
        self.halt = False
        self.halt_reasons: List[str] = []
        self.last_gate: Dict[str, Any] = {}

    @property
    def step_name(self) -> str:
        return STEPS[self.step_index] if 0 <= self.step_index < len(STEPS) else "HALT"

    def set_step(self, index: int):
        self.step_index = int(index)
        # Changing step does not auto-clear RED latch; operator must acknowledge.

    def acknowledge(self):
        self.halt = False
        self.halt_reasons.clear()
        self.light.clear_red()

    def _point_in_boxes(self, pt, boxes) -> bool:
        x, y, z = pt
        for b in boxes or []:
            mn, mx = b["min"], b["max"]
            if mn[0] <= x <= mx[0] and mn[1] <= y <= mx[1] and mn[2] <= z <= mx[2]:
                return True
        return False

    def alignment_ok(self, metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """Prefer unified insertion_clearance (Item3); fall back to legacy xy+tilt gates."""
        if metrics.get("insertion_clearance_ok") is not None:
            if metrics.get("insertion_clearance_ok") is True:
                return True, "ok"
            why = metrics.get("insertion_clearance_reason") or "clearance_out"
            return False, str(why)
        # Legacy path (pre-Item3 metrics only)
        align = self.calib.get("alignment", {})
        xy = metrics.get("align_xy_m")
        tilt = metrics.get("tilt_deg")
        if xy is None or tilt is None:
            return False, "align_uncalibrated_or_unavailable"
        if abs(xy) > float(align.get("xy_tol_m", 0.0025)):
            return False, "align_xy_out"
        if abs(tilt) > float(align.get("tilt_tol_deg", 2.0)):
            return False, "align_tilt_out"
        return True, "ok"

    def dual_latched(self, metrics: Dict[str, Any]) -> Tuple[bool, str]:
        lock = self.calib.get("latch_lock", {})
        la = metrics.get("left_angle_abs_deg")
        ra = metrics.get("right_angle_abs_deg")
        seated = metrics.get("seated_ok")
        if la is None or ra is None:
            return False, "latch_angle_unavailable"
        lim = float(lock.get("closed_angle_abs_deg_max", 3.0))
        if abs(la) > lim or abs(ra) > lim:
            return False, "latches_not_closed"
        if seated is not True:
            return False, "not_seated_or_unconfirmed"
        return True, "ok"

    def evaluate_gates(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Return block flags. RED hard-stop reasons also block advance."""
        block_down = False
        block_up = False
        block_release = False
        block_close = False
        reasons: List[str] = []
        name = self.step_name

        aligned, a_why = self.alignment_ok(metrics)
        dual, d_why = self.dual_latched(metrics)

        if name in ("ALIGN", "INSERT") and not aligned:
            block_down = True
            reasons.append("gate_not_aligned:" + a_why)

        if name in ("LATCH_LOCK", "RELEASE") and not dual:
            block_release = True
            reasons.append("gate_not_dual_latched:" + d_why)
        # Always harden RELEASE: never open fingers without dual latch evidence.
        if name == "RELEASE" and not dual:
            block_release = True

        # Contact-limited grip (D-B2): freeze FURTHER close only for a real dual-finger
        # over-limit grasp. Single-finger scrape while still wide open must not block CLOSE
        # (needed for GRASP_DIMM / tray pickup). latch_overforce_N is separate (latch press).
        grip_lim = float(self.safety.get("plan_d", {}).get("grip_force_limit_N", 15.0))
        fl = float(metrics.get("finger_left_N") or 0.0)
        fr = float(metrics.get("finger_right_N") or 0.0)
        width = metrics.get("gripper_width_m")
        dual_touch = fl >= 0.05 and fr >= 0.05
        over = max(fl, fr) >= grip_lim
        if over and dual_touch:
            block_close = True
            reasons.append("grip_force_limit")
        elif over and width is not None and float(width) < 0.02:
            # Nearly closed and already crushing — hold.
            block_close = True
            reasons.append("grip_force_limit")

        # Forbidden grasp contact → block close / treat as red-ish gate
        if metrics.get("forbidden_contact", 0) and float(metrics.get("forbidden_contact", 0)) >= 0.5:
            block_close = True
            reasons.append("forbidden_grasp_contact")

        # Early steps sit far from the slot: never let clearance/align RED lock -Z.
        early = name in ("OPEN_LATCH", "GRASP_DIMM", "EXTRACT")
        if early:
            self.light.clear_red("insertion_clearance")
            self.light.clear_red("align_xy_m")

        # RED latched metrics → halt advance (directional)
        # Force/overforce may still halt; clearance/align must not block_down outside ALIGN/INSERT.
        clearance_keys = {"insertion_clearance", "align_xy_m"}
        light_reasons = list(self.light.reasons)
        force_or_other = [r for r in light_reasons if r not in clearance_keys]
        if force_or_other or metrics.get("latch_overforce"):
            self.halt = True
            for r in force_or_other:
                if r not in self.halt_reasons:
                    self.halt_reasons.append(r)
            if metrics.get("latch_overforce") and "latch_overforce" not in self.halt_reasons:
                self.halt_reasons.append("latch_overforce")
            # block_down from RED only during ALIGN/INSERT (insertion phase).
            if name in ("ALIGN", "INSERT"):
                block_down = True
                block_release = True
            reasons.extend(self.halt_reasons)
        elif light_reasons and early:
            # Stale clearance/align reasons alone: drop halt so PgDn works.
            self.halt = False
            self.halt_reasons = [r for r in self.halt_reasons if r not in clearance_keys]

        # Hard guarantee: OPEN_LATCH / GRASP / EXTRACT never block world -Z.
        if early:
            block_down = False

        self.last_gate = {
            "step": name,
            "aligned": aligned,
            "dual_latched": dual,
            "block_down": block_down,
            "block_up": block_up,
            "block_release": block_release,
            "block_close": block_close,
            "reasons": list(dict.fromkeys(reasons)),
        }
        return self.last_gate

    def apply_motion_gate(self, tv, gsign, metrics: Dict[str, Any]):
        """Filter translation vector (world xyz) and gripper sign. tv is length-3."""
        gate = self.evaluate_gates(metrics)
        tv = list(tv)
        # World -Z is down for this scene (socket insertion axis aligns with +Z local = world Z).
        if gate["block_down"] and tv[2] < 0:
            tv[2] = 0.0
        if gate["block_up"] and tv[2] > 0:
            tv[2] = 0.0
        if gate["block_release"] and gsign > 0:
            gsign = 0
        if gate["block_close"] and gsign < 0:
            gsign = 0
        return tv, gsign, gate
