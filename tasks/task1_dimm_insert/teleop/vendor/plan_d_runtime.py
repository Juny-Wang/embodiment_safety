"""Plan D session-layer runtime: weak latch drive, thin contact, visual-only pins, overforce color."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PACKAGE = Path("/mnt/data2/wjy_SERL/Embodied_Safety/outputs/T1_rl/franka_manual_safety_v002")


def _load_json(name: str) -> Dict[str, Any]:
    return json.loads((PACKAGE / name).read_text())


def apply_session_plan_d(stage, calib: Optional[Dict[str, Any]] = None, safety: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Author Plan D physics tweaks on the session layer only. Soft / reversible."""
    from pxr import UsdPhysics, UsdGeom, Gf, Sdf, PhysxSchema

    if calib is None:
        calib = _load_json("geometry_calib.json")
    if safety is None:
        safety = _load_json("safety_config.json")
    plan = safety.get("plan_d", {})

    stage.SetEditTarget(stage.GetSessionLayer())
    report = {"latch_drives": [], "thin_contact": [], "pins_visual_only": [], "errors": []}

    # D-A1/A2: soft CLOSED hold that stays finger-openable.
    # Root cause of empty self-open: asset+Plan D used stiffness=0 viscous-only drive
    # (explicitly not a closed-position hold) + residual angularVelocity baked into
    # manual_franka_scene.usda latch overlays. Gravity COM torque is tiny (~1e-5 Nm)
    # but with k=0 any kick coasts open and never returns.
    # Fix: soft stiffness toward closed at session start + velocity/pose reset.
    # Per-frame update_latch_soft_hold() drops stiffness once opened past hold band
    # (or while finger contacts) so latches stay open after intentional open.
    damp = float(plan.get("latch_damping", 0.02))
    stiff = float(plan.get("latch_stiffness", 0.08))
    max_f = float(plan.get("latch_max_force", 0.06))
    reset_closed = bool(plan.get("latch_reset_closed_on_session", True))
    socket = safety["paths"]["socket"]
    report["latch_resets"] = []
    for side in ("Left", "Right"):
        path = f"{socket}/Joints/{side}Hinge"
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            report["errors"].append("missing " + path)
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        for name, val in (
            ("drive:angular:physics:stiffness", stiff),
            ("drive:angular:physics:damping", damp),
            ("drive:angular:physics:maxForce", max_f),
            ("drive:angular:physics:targetVelocity", 0.0),
            ("drive:angular:physics:targetPosition", 0.0),
        ):
            attr = prim.GetAttribute(name)
            if not attr:
                attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Float)
            attr.Set(float(val))
        type_attr = prim.GetAttribute("drive:angular:physics:type")
        if not type_attr:
            type_attr = prim.CreateAttribute("drive:angular:physics:type", Sdf.ValueTypeNames.Token)
        type_attr.Set("force")
        report["latch_drives"].append({
            "path": path, "damping": damp, "stiffness": stiff, "maxForce": max_f,
            "targetPosition_deg": 0.0, "hold": "soft_closed_startup; gated by update_latch_soft_hold",
        })

        if reset_closed:
            latch_path = f"{socket}/{side}Latch"
            latch = stage.GetPrimAtPath(latch_path)
            if not latch or not latch.IsValid():
                report["errors"].append("missing " + latch_path)
                continue
            for aname, zero in (
                ("physics:velocity", Gf.Vec3f(0.0, 0.0, 0.0)),
                ("physics:angularVelocity", Gf.Vec3f(0.0, 0.0, 0.0)),
            ):
                attr = latch.GetAttribute(aname)
                if not attr:
                    attr = latch.CreateAttribute(aname, Sdf.ValueTypeNames.Vector3f)
                attr.Set(zero)
            orient = latch.GetAttribute("xformOp:orient")
            if orient:
                orient.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            else:
                try:
                    xformable = UsdGeom.Xformable(latch)
                    op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
                    op.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                except Exception as _xe:
                    report["errors"].append("orient_reset %s: %r" % (latch_path, _xe))
            for jname in (
                "state:angular:physics:position",
                "state:angular:physics:velocity",
            ):
                jattr = prim.GetAttribute(jname)
                if jattr:
                    jattr.Set(0.0)
            report["latch_resets"].append({
                "path": latch_path, "orient": "identity", "velocity": "zeroed",
            })

    # D-B1: thin contactOffset on structural DIMM colliders (NOT thickening grasp ROI geometry)
    dimm = safety["paths"]["dimm"]
    thin = float(plan.get("thin_contact_offset_m", 5.0e-5))
    keep_prefixes = ("PCBMain", "TopPress", "Left_", "Right_")
    pin_prefix = calib.get("dimm_local", {}).get("dense_pins_visual_only", {}).get("prim_name_prefix", "Gold_")
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not path.startswith(dimm + "/"):
            continue
        name = prim.GetName()
        # Visual-only dense pins
        if name.startswith(pin_prefix):
            col = prim.GetAttribute("physics:collisionEnabled")
            if not col:
                col = prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool)
            col.Set(False)
            report["pins_visual_only"].append(path)
            continue
        if any(name.startswith(p) or name == p.rstrip("_") for p in keep_prefixes) or name in ("PCBMain", "TopPress"):
            attr = prim.GetAttribute("physxCollision:contactOffset")
            if not attr:
                # Apply schema if needed
                try:
                    PhysxSchema.PhysxCollisionAPI.Apply(prim)
                except Exception:
                    pass
                attr = prim.GetAttribute("physxCollision:contactOffset")
                if not attr:
                    attr = prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float)
            if attr:
                attr.Set(thin)
                rest = prim.GetAttribute("physxCollision:restOffset")
                if rest:
                    rest.Set(0.0)
                report["thin_contact"].append({"path": path, "contactOffset": thin})

    # Also keep latch geometry thin (already ~5e-5)
    for side in ("Left", "Right"):
        gpath = f"{socket}/{side}Latch/Geometry"
        gprim = stage.GetPrimAtPath(gpath)
        if gprim and gprim.IsValid():
            attr = gprim.GetAttribute("physxCollision:contactOffset")
            if attr:
                attr.Set(max(thin, float(attr.Get() or 0.0)))
                report["thin_contact"].append({"path": gpath, "contactOffset": attr.Get()})

    report["ok"] = not report["errors"]
    return report


def update_latch_soft_hold(
    stage,
    safety: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    calib: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Per-frame soft closed hold with an open dead-zone.

    Near closed (|angle| <= latch_hold_band_deg) and no finger contact:
      apply soft stiffness toward targetPosition=0 so empty latches stay shut.
    Past the band, OR while finger presses the latch:
      stiffness=0 (light damping only) so an opened latch is NOT dragged closed.

    Session apply still resets pose/velocity once; this function only retunes drive.
    """
    from pxr import Sdf

    if safety is None:
        safety = _load_json("safety_config.json")
    if calib is None:
        try:
            calib = _load_json("geometry_calib.json")
        except Exception:
            calib = {}
    plan = safety.get("plan_d", {})
    metrics = metrics or {}
    damp = float(plan.get("latch_damping", 0.02))
    stiff_closed = float(plan.get("latch_stiffness", 0.08))
    max_f = float(plan.get("latch_max_force", 0.06))
    band = float(plan.get(
        "latch_hold_band_deg",
        (calib or {}).get("latch_lock", {}).get("closed_angle_abs_deg_max", 3.0) * 1.5 or 5.0,
    ))
    if band <= 0:
        band = 5.0
    contact_N = float(plan.get("latch_release_on_contact_N", 0.15))
    socket = safety["paths"]["socket"]

    stage.SetEditTarget(stage.GetSessionLayer())
    out: Dict[str, Any] = {"band_deg": band, "contact_N": contact_N, "sides": {}}
    for side in ("Left", "Right"):
        path = f"{socket}/Joints/{side}Hinge"
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            out["sides"][side.lower()] = {"ok": False, "error": "missing"}
            continue
        key = side.lower()
        ang = metrics.get(f"{key}_angle_abs_deg")
        if ang is None:
            jattr = prim.GetAttribute("state:angular:physics:position")
            try:
                if jattr and jattr.Get() is not None:
                    ang = abs(float(jattr.Get()))
            except Exception:
                ang = None
        press = metrics.get(f"{key}_press_N")
        try:
            press_f = float(press) if press is not None else 0.0
        except Exception:
            press_f = 0.0
        near_closed = (ang is not None) and (float(ang) <= band)
        contacted = press_f >= contact_N
        # Unknown angle: keep soft hold (startup / measure lag) unless contacted.
        if ang is None:
            stiff = 0.0 if contacted else stiff_closed
            mode = "hold_unknown_angle" if not contacted else "release_contact"
        elif near_closed and not contacted:
            stiff = stiff_closed
            mode = "hold_closed"
        else:
            stiff = 0.0
            mode = "open_deadzone" if not contacted else "release_contact"

        for name, val in (
            ("drive:angular:physics:stiffness", stiff),
            ("drive:angular:physics:damping", damp),
            ("drive:angular:physics:maxForce", max_f),
            ("drive:angular:physics:targetVelocity", 0.0),
            ("drive:angular:physics:targetPosition", 0.0),
        ):
            attr = prim.GetAttribute(name)
            if not attr:
                attr = prim.CreateAttribute(name, Sdf.ValueTypeNames.Float)
            attr.Set(float(val))
        type_attr = prim.GetAttribute("drive:angular:physics:type")
        if not type_attr:
            type_attr = prim.CreateAttribute("drive:angular:physics:type", Sdf.ValueTypeNames.Token)
        type_attr.Set("force")
        out["sides"][key] = {
            "ok": True,
            "angle_abs_deg": None if ang is None else float(ang),
            "press_N": press_f,
            "stiffness": stiff,
            "mode": mode,
        }
    out["ok"] = all(v.get("ok") for v in out["sides"].values()) if out["sides"] else False
    return out



def set_latch_overforce_color(stage, left_force: float, right_force: float, limit_N: float) -> Dict[str, Any]:
    """D-A3: over-force latch → red displayColor (breakage alarm). Session layer only."""
    from pxr import UsdGeom, Gf, Sdf

    stage.SetEditTarget(stage.GetSessionLayer())
    out = {"left": False, "right": False}
    for side, force in (("Left", left_force), ("Right", right_force)):
        over = force is not None and limit_N is not None and force >= limit_N
        out[side.lower()] = bool(over)
        for suffix in (f"/World/MatchedSocket/Asset/{side}Latch/Geometry",
                       f"/World/MatchedSocket/Asset/{side}Latch"):
            prim = stage.GetPrimAtPath(suffix)
            if not prim or not prim.IsValid():
                continue
            # Prefer displayColor on gprim
            attr = prim.GetAttribute("primvars:displayColor")
            if not attr:
                # try child
                for child in prim.GetChildren():
                    a = child.GetAttribute("primvars:displayColor")
                    if a:
                        prim = child
                        attr = a
                        break
            if not attr:
                attr = prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray)
            color = [(1.0, 0.05, 0.05)] if over else [(0.75, 0.55, 0.15)]
            try:
                attr.Set(color)
            except Exception:
                try:
                    attr.Set([Gf.Vec3f(*color[0])])
                except Exception:
                    pass
    out["latch_overforce"] = out["left"] or out["right"]
    return out


def _mat44(m):
    import numpy as np
    return np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=float)


def compute_tilt_primary_deg(sock_xf, dimm_xf, dimm_length_m: float = 0.13335) -> Dict[str, Any]:
    """Primary tilt + end_difference matching Feedback.measure (Item2 §2.4 / §2.6).

    t1 = acos(|board_normal · socket_Y|)  [deg]
    t2 = asin(|long_axis · insert_axis|)  [deg]
    tilt = max(t1, t2)
    end_difference_mm = |long · insert| * length_m * 1000
    """
    import numpy as np

    Sw = _mat44(sock_xf)
    Dw = _mat44(dimm_xf)
    sock_y = Sw[:3, 1]
    insert = Sw[:3, 2]  # socket local +Z in world
    board_n = Dw[:3, 1]  # DIMM local Y
    long_ax = Dw[:3, 0]  # DIMM local X
    sock_y = sock_y / (np.linalg.norm(sock_y) + 1e-12)
    insert = insert / (np.linalg.norm(insert) + 1e-12)
    board_n = board_n / (np.linalg.norm(board_n) + 1e-12)
    long_ax = long_ax / (np.linalg.norm(long_ax) + 1e-12)
    t1 = float(np.degrees(np.arccos(float(np.clip(abs(np.dot(board_n, sock_y)), -1.0, 1.0)))))
    t2 = float(np.degrees(np.arcsin(float(np.clip(abs(np.dot(long_ax, insert)), -1.0, 1.0)))))
    end_mm = float(abs(np.dot(long_ax, insert)) * float(dimm_length_m) * 1000.0)
    return {
        "tilt_t1_deg": t1,
        "tilt_t2_deg": t2,
        "tilt_deg_primary": float(max(t1, t2)),
        "end_difference_mm": end_mm,
    }


def compute_tilt_fallback_board_z_deg(sock_xf, dimm_xf) -> float:
    """Legacy Plan D formula |90 - acos(|n_board · z_socket|)| — debug only, not for gates."""
    import numpy as np

    Sw = _mat44(sock_xf)
    Dw = _mat44(dimm_xf)
    sz = Sw[:3, 2]
    dy = Dw[:3, 1]
    cosang = float(np.clip(abs(np.dot(dy / (np.linalg.norm(dy) + 1e-12), sz / (np.linalg.norm(sz) + 1e-12))), -1, 1))
    return abs(90.0 - float(np.degrees(np.arccos(cosang))))


def compute_alignment_metrics(stage, calib: Dict[str, Any], safety: Dict[str, Any]) -> Dict[str, Any]:
    """DIMM pose vs socket entry in SOCKET ASSET local frame (same as force/tilt base class).

    Frame fix (Item3): use frames.socket_asset_prim (default /World/MatchedSocket/Asset),
    NOT the parent /World/MatchedSocket. Physics unchanged — pose query only.
    """
    from pxr import UsdGeom, Usd

    frames = calib.get("frames", {})
    socket_path = frames.get("socket_asset_prim") or safety.get("paths", {}).get("socket") or "/World/MatchedSocket/Asset"
    dimm_path = frames.get("dimm_prim") or safety.get("paths", {}).get("dimm") or "/World/MatchedDIMM"
    sock_prim = stage.GetPrimAtPath(socket_path)
    dimm_prim = stage.GetPrimAtPath(dimm_path)
    if not sock_prim or not dimm_prim:
        return {
            "align_xy_m": None,
            "tilt_deg": None,
            "tilt_deg_primary": None,
            "seated_ok": None,
            "seated_depth_m": None,
            "end_difference_mm": None,
            "socket_frame_used": socket_path,
        }

    tc = Usd.TimeCode.Default()
    sock_xf = UsdGeom.Xformable(sock_prim).ComputeLocalToWorldTransform(tc)
    dimm_xf = UsdGeom.Xformable(dimm_prim).ComputeLocalToWorldTransform(tc)
    inv = sock_xf.GetInverse()
    dimm_t = dimm_xf.ExtractTranslation()
    local = inv.Transform(dimm_t)
    entry = calib["socket_local"]["entry_point"]
    seated = calib["socket_local"]["seated_reference"]
    dxy = ((local[0] - entry[0]) ** 2 + (local[1] - entry[1]) ** 2) ** 0.5
    depth = float(local[2] - seated[2])  # ~0 when seated

    dimm_len = float(calib.get("dimm_local", {}).get("length_m", 0.13335))
    primary = compute_tilt_primary_deg(sock_xf, dimm_xf, dimm_len)
    fallback = compute_tilt_fallback_board_z_deg(sock_xf, dimm_xf)

    seated_tol = float(calib.get("alignment", {}).get("seated_depth_tol_m", 0.0015))
    xy_tol = float(calib.get("alignment", {}).get("xy_tol_m", 0.0025))
    # Keep historical 2x xy band for seated_ok (lock gate); Item4 may retune.
    seated_ok = abs(depth) <= seated_tol and dxy <= xy_tol * 2

    return {
        "align_xy_m": float(dxy),
        "align_z_m": float(local[2] - entry[2]),
        "seated_depth_m": float(depth),
        "seated_ok": bool(seated_ok),
        "tilt_deg_primary": float(primary["tilt_deg_primary"]),
        "tilt_t1_deg": float(primary["tilt_t1_deg"]),
        "tilt_t2_deg": float(primary["tilt_t2_deg"]),
        "end_difference_mm": float(primary["end_difference_mm"]),
        "tilt_board_to_socket_z_deg": float(fallback),  # debug / legacy only
        "socket_frame_used": socket_path,
        "dimm_in_socket_local": [float(local[0]), float(local[1]), float(local[2])],
    }


def evaluate_insertion_clearance(
    metrics: Dict[str, Any],
    calib: Dict[str, Any],
    safety: Dict[str, Any],
) -> Dict[str, Any]:
    """ONE coherent insertion-clearance gate from xy / tilt / end (pose-based).

    Does NOT invent new mm numbers: reuses geometry_calib.alignment + existing
    end_difference rule limit (1.0 mm demo). Seated depth is reported but not
    part of the ALIGN/INSERT block_down clearance (it feeds dual_latched).

    Returns metrics to merge, including compat mirrors of old keys.
    """
    align_cfg = calib.get("alignment", {})
    ic = safety.get("insertion_clearance", {})
    xy_tol = float(ic.get("xy_tol_m", align_cfg.get("xy_tol_m", 0.0025)))
    tilt_tol = float(ic.get("tilt_tol_deg", align_cfg.get("tilt_tol_deg", 2.0)))
    end_tol = float(ic.get("end_difference_tol_mm", 1.0))
    warn_r = float(ic.get("warning_ratio", safety.get("warning_ratio", 0.8)))
    include_end = bool(ic.get("include_end_when_available", True))

    xy = metrics.get("align_xy_m")
    tilt = metrics.get("tilt_deg")
    if tilt is None:
        tilt = metrics.get("tilt_deg_primary")
    end = metrics.get("end_difference_mm")
    seated_depth = metrics.get("seated_depth_m")

    comps: Dict[str, Optional[float]] = {
        "xy": (None if xy is None else abs(float(xy)) / xy_tol) if xy_tol > 0 else None,
        "tilt": (None if tilt is None else abs(float(tilt)) / tilt_tol) if tilt_tol > 0 else None,
        "end": (None if end is None else abs(float(end)) / end_tol) if end_tol > 0 else None,
    }

    active: List[Tuple[str, float]] = []
    missing: List[str] = []
    for k in ("xy", "tilt"):
        if comps[k] is None:
            missing.append(k)
        else:
            active.append((k, float(comps[k])))
    if include_end and comps["end"] is not None:
        active.append(("end", float(comps["end"])))

    out: Dict[str, Any] = {
        "insertion_clearance_xy_ratio": comps["xy"],
        "insertion_clearance_tilt_ratio": comps["tilt"],
        "insertion_clearance_end_ratio": comps["end"],
        "insertion_clearance_seated_depth_m": seated_depth,  # informative only
        "insertion_clearance_tol": {
            "xy_tol_m": xy_tol,
            "tilt_tol_deg": tilt_tol,
            "end_difference_tol_mm": end_tol,
            "warning_ratio": warn_r,
        },
    }

    if missing:
        out.update({
            "insertion_clearance_ok": False,
            "insertion_clearance_ratio": None,
            "insertion_clearance_worst": None,
            "insertion_clearance_color": "GRAY",
            "insertion_clearance_reason": "missing_" + "+".join(missing),
        })
        return out

    worst_k, worst_r = max(active, key=lambda x: x[1])
    if worst_r >= 1.0:
        color = "RED"
        ok = False
        reason = f"clearance_{worst_k}_out"
    elif worst_r >= warn_r:
        color = "YELLOW"
        ok = True
        reason = f"clearance_{worst_k}_warn"
    else:
        color = "GREEN"
        ok = True
        reason = "ok"

    out.update({
        "insertion_clearance_ok": bool(ok and color != "RED"),
        "insertion_clearance_ratio": float(worst_r),
        "insertion_clearance_worst": worst_k,
        "insertion_clearance_color": color,
        "insertion_clearance_reason": reason,
        # Compat: keep traffic-light able to use a single 0..∞ score with limit 1.0
        "insertion_clearance": float(worst_r),
    })
    return out


def merge_calib_into_safety(safety: Dict[str, Any], calib: Dict[str, Any]) -> Dict[str, Any]:
    """Populate grasp_regions from local-frame calib; mark calibrated."""
    safety = json.loads(json.dumps(safety))  # deep copy
    dimm = calib.get("dimm_local", {})
    safety["grasp_regions"] = {
        "calibrated": True,
        "frame": "dimm_local",
        "allowed_boxes": [
            {"min": b["min"], "max": b["max"], "name": b.get("name")}
            for b in dimm.get("grasp_allowed_boxes", [])
        ],
        "forbidden_boxes": [
            {"min": b["min"], "max": b["max"], "name": b.get("name")}
            for b in dimm.get("grasp_forbidden_boxes", [])
        ],
    }
    geom = safety.setdefault("geometry", {})
    geom["seated_reference_calibrated"] = True
    geom["seated_reference_socket_local"] = calib["socket_local"]["seated_reference"]
    geom["entry_socket_local"] = calib["socket_local"]["entry_point"]
    geom["socket_insertion_axis"] = calib["socket_local"]["insertion_axis"]
    geom["dimm_length_m"] = dimm.get("length_m", geom.get("dimm_length_m"))
    geom["dimm_long_axis"] = dimm.get("long_axis", geom.get("dimm_long_axis"))
    geom["dimm_normal_axis"] = dimm.get("normal_axis", geom.get("dimm_normal_axis"))
    return safety
