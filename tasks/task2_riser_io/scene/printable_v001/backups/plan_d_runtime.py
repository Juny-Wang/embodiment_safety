"""Task2 session helpers (Plan-D pattern adapted). No DIMM latches.

Provides the attribute names keyboard-style panels may probe, plus Task2
force→displayColor alarm hooks for NutPlate / ScrewTray / rings.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional

PACKAGE = Path(__file__).resolve().parent


def _load_json(name: str) -> Dict[str, Any]:
    return json.loads((PACKAGE / name).read_text())


def merge_calib_into_safety(safety: Dict[str, Any], calib: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(safety)
    if calib:
        out.setdefault("geometry_calib", calib)
        if "alignment" in calib:
            out.setdefault("alignment", calib["alignment"])
    return out


def apply_session_plan_d(stage, calib=None, safety=None) -> Dict[str, Any]:
    """Session-layer soft tweaks. Task2: no latch hinges; mark proxy prims."""
    report = {"ok": True, "task": "task2_riser_io", "notes": [], "errors": []}
    try:
        stage.SetEditTarget(stage.GetSessionLayer())
    except Exception as e:
        report["errors"].append(repr(e))
        report["ok"] = False
        return report
    safety = safety or {}
    paths = safety.get("paths") or {}
    for key in ("nut_plate", "screw_tray", "board", "pcb_4_terminal"):
        p = paths.get(key)
        if not p:
            continue
        prim = stage.GetPrimAtPath(p)
        if not prim or not prim.IsValid():
            report["notes"].append("missing " + str(p))
        else:
            report["notes"].append("seen " + str(p))
    report["notes"].append("Phase1 wires left untouched; Phase2 proxies kinematic/dynamic as authored")
    return report


def evaluate_insertion_clearance(stage, calib, safety, metrics=None) -> Dict[str, Any]:
    """No PCB-hole clearance solver yet — soft pass until geometry_calib is measured."""
    return {
        "insertion_clearance_ok": True,
        "insertion_clearance_reason": "task2_uncalibrated_soft_pass",
        "align_xy_m": (metrics or {}).get("align_xy_m"),
        "tilt_deg": (metrics or {}).get("tilt_deg"),
    }


def compute_alignment_metrics(stage, calib, safety) -> Dict[str, Any]:
    """Placeholder alignment until EE vs hole frames are calibrated."""
    return {
        "align_xy_m": None,
        "tilt_deg": None,
        "dimm_in_socket_local": None,
        "note": "task2_alignment_uncalibrated",
    }


def set_latch_overforce_color(stage, left_N, right_N, limit) -> Dict[str, Any]:
    """No latches in Task2; API kept for panel compatibility."""
    return {"latch_overforce": False, "left_N": left_N, "right_N": right_N, "limit": limit}


def update_latch_soft_hold(stage, *args, **kwargs) -> Dict[str, Any]:
    return {"sides": {"left": {"mode": "n/a"}, "right": {"mode": "n/a"}}}


def set_force_alarm_color(stage, prim_path: str, ratio: float, warning_ratio: float = 0.7) -> str:
    """Map force ratio to displayColor on a prim (GRAY/GREEN/YELLOW/RED)."""
    from pxr import UsdGeom, Gf
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return "MISSING"
    if ratio is None:
        color = (0.55, 0.55, 0.55)
        state = "GRAY"
    elif ratio >= 1.0:
        color = (1.0, 0.12, 0.08)
        state = "RED"
    elif ratio >= warning_ratio:
        color = (1.0, 0.75, 0.05)
        state = "YELLOW"
    else:
        color = (0.15, 0.85, 0.30)
        state = "GREEN"
    try:
        # Prefer imageable displayColor on first gprim child or self
        targets = [prim]
        for child in prim.GetChildren():
            targets.append(child)
        for t in targets:
            img = UsdGeom.Gprim(t) if t.IsA(UsdGeom.Gprim) else None
            if img:
                img.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
                break
            else:
                attr = t.GetAttribute("primvars:displayColor")
                if attr:
                    attr.Set([(color[0], color[1], color[2])])
                    break
    except Exception:
        pass
    return state


def apply_task2_alarms(stage, metrics: Dict[str, Any], safety: Dict[str, Any]) -> Dict[str, str]:
    """Color ScrewTray / NutPlate / optional ring from metrics vs limits."""
    paths = safety.get("paths") or {}
    tl = safety.get("traffic_light") or {}
    warn = float(tl.get("warning_ratio", 0.7))
    out = {}
    mapping = [
        ("screw_axial_N", paths.get("nut_plate", "/NutPlate_M4"), "screw_axial_N"),
        ("grip_force_N", paths.get("screw_tray", "/ScrewTray"), "grip_force_N"),
        ("ring_axial_N", paths.get("ring_terminal"), "ring_axial_N"),
        ("wire_joint_force", paths.get("wire_root"), "wire_joint_force"),
    ]
    limits = {r["key"]: r.get("limit") for r in safety.get("rules", []) if "key" in r}
    for metric_key, prim_path, limit_key in mapping:
        if not prim_path:
            continue
        val = metrics.get(metric_key)
        lim = limits.get(limit_key)
        ratio = None
        if val is not None and lim and float(lim) > 0:
            ratio = abs(float(val)) / float(lim)
        out[metric_key] = set_force_alarm_color(stage, prim_path, ratio, warn)
    return out
