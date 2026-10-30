"""Task2 StepFSM + TrafficLight. Phase1 wire then Phase2 screw. No Isaac imports."""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple

# Authoritative ids from TASK_FLOW.md Phase1 + Phase2
STEPS = [
    # Phase 1 — wire connection
    "home",
    "select_wire",
    "grasp_ring",
    "lift_clear",
    "route_to_stud",
    "align_ring_stud",
    "seat_ring",
    "hold_verify",
    "release_retract",
    "repeat_wire_or_phase2",
    # Phase 2 — screw fasten
    "select_screw",
    "grasp_screw",
    "lift_from_tray",
    "transport_to_hole",
    "align_screw_hole",
    "insert_start",
    "fasten_turn",
    "release_verify",
    "repeat_screw_or_done",
]

STEP_LABELS = [
    "P1.0 home",
    "P1.1 select_wire",
    "P1.2 grasp_ring",
    "P1.3 lift_clear",
    "P1.4 route_to_stud",
    "P1.5 align_ring_stud",
    "P1.6 seat_ring",
    "P1.7 hold_verify",
    "P1.8 release_retract",
    "P1.9 repeat_wire_or_phase2",
    "P2.0 select_screw",
    "P2.1 grasp_screw",
    "P2.2 lift_from_tray",
    "P2.3 transport_to_hole",
    "P2.4 align_screw_hole",
    "P2.5 insert_start",
    "P2.6 fasten_turn",
    "P2.7 release_verify",
    "P2.8 repeat_screw_or_done",
]

PHASE1 = set(STEPS[:10])
PHASE2 = set(STEPS[10:])


class TrafficLight:
    def __init__(self, warning_ratio=0.7, clear_ratio=0.7, yellow_hold_s=0.15, green_hold_s=0.15):
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
        if ratio >= self.warning_ratio or (prev == "YELLOW" and ratio >= self.clear_ratio):
            cand = "YELLOW"
        else:
            cand = "GREEN"
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
        self.calib = calib or {}
        self.safety = safety or {}
        self.step_index = 0
        tl = safety.get("traffic_light", {})
        self.light = TrafficLight(
            warning_ratio=tl.get("warning_ratio", safety.get("warning_ratio", 0.7)),
            clear_ratio=tl.get("warning_clear_ratio", safety.get("warning_clear_ratio", 0.7)),
            yellow_hold_s=tl.get("yellow_min_duration_s", 0.15),
            green_hold_s=tl.get("green_min_duration_s", 0.15),
        )
        self.halt = False
        self.halt_reasons: List[str] = []
        self.last_gate: Dict[str, Any] = {}

    @property
    def step_name(self) -> str:
        return STEPS[self.step_index] if 0 <= self.step_index < len(STEPS) else "HALT"

    @property
    def phase(self) -> int:
        return 1 if self.step_name in PHASE1 else (2 if self.step_name in PHASE2 else 0)

    def set_step(self, index: int):
        self.step_index = int(index)

    def acknowledge(self):
        self.halt = False
        self.halt_reasons.clear()
        self.light.clear_red()

    def alignment_ok(self, metrics: Dict[str, Any]) -> Tuple[bool, str]:
        """Generic XY/tilt gate; Task2 calib may leave unset -> soft pass with reason."""
        align = self.calib.get("alignment", {})
        xy = metrics.get("align_xy_m")
        tilt = metrics.get("tilt_deg")
        if xy is None or tilt is None:
            return False, "alignment_measurement_unavailable"
        if abs(xy) > float(align.get("xy_tol_m", 0.003)):
            return False, "align_xy_out"
        if abs(tilt) > float(align.get("tilt_tol_deg", 5.0)):
            return False, "align_tilt_out"
        return True, "ok"

    def evaluate_gates(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        block_down = False
        block_up = False
        block_release = False
        block_close = False
        reasons: List[str] = []
        name = self.step_name

        aligned, a_why = self.alignment_ok(metrics)

        # Phase1 seat / Phase2 insert: block further -Z if not aligned (when calibrated)
        if name in ("align_ring_stud", "seat_ring", "align_screw_hole", "insert_start", "fasten_turn"):
            if a_why != "align_uncalibrated_soft_pass" and not aligned:
                block_down = True
                reasons.append("gate_not_aligned:" + a_why)

        # Grip force freeze further close
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
            block_close = True
            reasons.append("grip_force_limit")

        if metrics.get("forbidden_contact", 0) and float(metrics.get("forbidden_contact", 0)) >= 0.5:
            block_close = True
            reasons.append("forbidden_grasp_contact")

        # Traffic light on configured rule keys
        for rule in self.safety.get("rules", []):
            key = rule.get("key")
            limit = rule.get("limit")
            step_ids = rule.get("steps") or []
            # steps may be names (Task2) or ints (legacy)
            active = False
            if not step_ids:
                active = True
            else:
                for s in step_ids:
                    if isinstance(s, str) and s == name:
                        active = True
                    elif isinstance(s, int) and s == self.step_index:
                        active = True
            val = metrics.get(key)
            if val is None:
                # also accept common aliases
                aliases = {
                    "wire_joint_force": "wire_joint_force_N",
                    "ring_axial_N": "ring_axial_N",
                    "ring_lateral_N": "ring_lateral_N",
                    "grip_force_N": "grip_force_N",
                    "screw_axial_N": "screw_axial_N",
                    "screw_torque_Nm": "screw_torque_Nm",
                }
                val = metrics.get(aliases.get(key, key))
            color = self.light.evaluate_rule(key, float(val) if val is not None else None, limit, active, float(metrics.get("dt") or 0.001))
            if color == "RED":
                self.halt = True
                if key not in self.halt_reasons:
                    self.halt_reasons.append(key)

        if self.light.reasons:
            self.halt = True
            for r in self.light.reasons:
                if r not in self.halt_reasons:
                    self.halt_reasons.append(r)

        self.last_gate = {
            "step": name,
            "phase": self.phase,
            "aligned": aligned,
            "align_why": a_why,
            "block_down": block_down,
            "block_up": block_up,
            "block_release": block_release,
            "block_close": block_close,
            "halt": self.halt,
            "reasons": list(reasons) + list(self.halt_reasons),
            "lights": dict(self.light.state),
        }
        return self.last_gate


# Aliases expected by keyboard_panel-style loaders
PLAN_D_STEPS = STEPS
