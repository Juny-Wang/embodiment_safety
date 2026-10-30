"""Task2 session helpers (Plan-D pattern adapted). No DIMM latches.

Provides the attribute names keyboard-style panels may probe, plus Task2
force→displayColor alarm hooks for NutPlate / ScrewTray / rings.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PACKAGE = Path(__file__).resolve().parent
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

def _prim(stage, path_str):
    """Isaac 5 GetPrimAtPath needs Sdf.Path, not bare str."""
    from pxr import Sdf
    if path_str is None or stage is None:
        return None
    if isinstance(path_str, str):
        return stage.GetPrimAtPath(Sdf.Path(path_str))
    return stage.GetPrimAtPath(path_str)




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
        prim = _prim(stage, p)
        if not prim or not prim.IsValid():
            report["notes"].append("missing " + str(p))
        else:
            report["notes"].append("seen " + str(p))
    report["notes"].append("Phase1 wires left untouched; Phase2 proxies kinematic/dynamic as authored")

    # Breakable neck + screw reseat (Task2 T17 / missing Screw_4)
    try:
        if str(PACKAGE) not in sys.path:
            sys.path.insert(0, str(PACKAGE))
        import importlib
        import breakable_neck
        breakable_neck = importlib.reload(breakable_neck)
        report["screw_reseat"] = breakable_neck.reseat_screws(stage, safety)
        report["breakable_neck"] = breakable_neck.ensure_necks(stage, safety)
        report["breakable_neck_joints"] = breakable_neck.configure_break_forces(stage, safety)
        report["notes"].append("BREAKABLE_NECK_20261025 + SCREW_RESEAT_20261025")
    except ModuleNotFoundError:
        # Absolute fallback when Kit sys.path lacks scene/
        import importlib.util
        _bn_path = PACKAGE / "breakable_neck.py"
        if not _bn_path.is_file():
            _bn_path = Path("/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/breakable_neck.py")
        _spec = importlib.util.spec_from_file_location("breakable_neck", str(_bn_path))
        breakable_neck = importlib.util.module_from_spec(_spec)
        sys.modules["breakable_neck"] = breakable_neck
        _spec.loader.exec_module(breakable_neck)
        report["screw_reseat"] = breakable_neck.reseat_screws(stage, safety)
        report["breakable_neck"] = breakable_neck.ensure_necks(stage, safety)
        report["breakable_neck_joints"] = breakable_neck.configure_break_forces(stage, safety)
        report["notes"].append("BREAKABLE_NECK_20261025 + SCREW_RESEAT_20261025 (file_load)")
    except Exception as _bn:
        report["errors"].append("breakable_neck:" + repr(_bn))
        report["ok"] = False
    return report


def evaluate_insertion_clearance(stage, calib, safety, metrics=None):
    m=metrics or {}
    xy=m.get("align_xy_m");tilt=m.get("tilt_deg")
    ok=xy is not None and tilt is not None and xy<=calib["alignment"]["xy_tol_m"] and tilt<=calib["alignment"]["tilt_tol_deg"]
    return {"insertion_clearance_ok":ok,
            "insertion_clearance_reason":"cad_alignment_only_no_collision_proof" if ok else "unaligned_or_measurement_unavailable"}

def compute_alignment_metrics(stage, calib, safety):
    import math
    from pxr import UsdGeom,Gf
    calib = calib or {}
    i=int(safety.get("selected_index",0))
    phase=int(safety.get("selected_phase",1))
    seats=calib.get("seat_paths") or (safety.get("paths") or {}).get("seat_targets") or []
    rings=calib.get("ring_paths") or (safety.get("paths") or {}).get("ring_paths") or []
    screws=calib.get("screw_paths") or (safety.get("paths") or {}).get("screws") or []
    if i>=len(seats) or (phase==1 and i>=len(rings)) or (phase!=1 and i>=len(screws)):
        return {"align_xy_m":None,"tilt_deg":None,"alignment_source":"index_or_paths_missing"}
    target=_prim(stage, seats[i])
    source=_prim(stage, rings[i] if phase==1 else screws[i])
    if not source or not source.IsValid() or not target or not target.IsValid():
        return {"align_xy_m":None,"tilt_deg":None,"alignment_source":"missing_prim"}
    xc=UsdGeom.XformCache()
    sm=xc.GetLocalToWorldTransform(source);tm=xc.GetLocalToWorldTransform(target)
    relative=tm.GetInverse().Transform(sm.ExtractTranslation())
    axis=sm.TransformDir(Gf.Vec3d(0,0,1)).GetNormalized()
    target_axis=tm.TransformDir(Gf.Vec3d(0,0,1)).GetNormalized()
    return {"align_xy_m":math.hypot(relative[0],relative[1]),
            "tilt_deg":math.degrees(math.acos(max(-1.,min(1.,Gf.Dot(axis,target_axis))))),
            "axial_offset_m":float(relative[2]),"alignment_source":"composed_USD_live_refresh_required"}


def set_latch_overforce_color(stage, left_N, right_N, limit) -> Dict[str, Any]:
    """No latches in Task2; API kept for panel compatibility."""
    return {"latch_overforce": False, "left_N": left_N, "right_N": right_N, "limit": limit}


def update_latch_soft_hold(stage, *args, **kwargs) -> Dict[str, Any]:
    return {"sides": {"left": {"mode": "n/a"}, "right": {"mode": "n/a"}}}


def set_force_alarm_color(stage, prim_path: str, ratio: float, warning_ratio: float = 0.7) -> str:
    """Map force ratio to displayColor on a prim (GRAY/GREEN/YELLOW/RED)."""
    from pxr import UsdGeom, Gf
    prim = _prim(stage, prim_path)
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
        from pxr import Usd
        targets = list(Usd.PrimRange(prim))
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
        ("screw_axial_N", paths.get("selected_screw", "/Screw_1"), "screw_axial_N"),
        
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
