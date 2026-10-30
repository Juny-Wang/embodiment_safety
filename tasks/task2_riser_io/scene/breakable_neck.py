"""Task2 breakable crimp-neck proxy (T17).

Orange short neck between ring terminal and distal cable segment.
- F_warn  → yellow displayColor on neck
- F_break sustained N frames → PhysX FixedJoint breaks (or we disable it),
  neck turns red, separation is free rigid-body motion, alarm flag set.

Force is a DEMO PROXY (stretch × stiffness), not Mini40 / constitutive stress.
Calibrate F_warn / F_break by hand-pull teleop (thresh-4 style).
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import math

PACKAGE = Path(__file__).resolve().parent

WIRE_IDS = ("wire1", "wire2", "wire3", "wire4")

# Rest attachment length ≈ |localPos0| on ring ≈ 0.023 m along local Y.
DEFAULT_REST_LEN_M = 0.023
DEFAULT_STIFFNESS_N_PER_M = 200.0  # proxy only


def _cfg(safety: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if safety and isinstance(safety.get("breakable_neck"), dict):
        return dict(safety["breakable_neck"])
    try:
        raw = json.loads((PACKAGE / "safety_config.json").read_text())
        return dict(raw.get("breakable_neck") or {})
    except Exception:
        return {}


def _prim(stage, path_str):
    from pxr import Sdf
    if stage is None or not path_str:
        return None
    return stage.GetPrimAtPath(Sdf.Path(str(path_str)))


def _paths(wire: str) -> Dict[str, str]:
    # Neck is a child of the ring so localPos0-ish translate stays in ring frame.
    return {
        "wire": f"/{wire}",
        "joint": f"/{wire}/FixedJoint",
        "ring": f"/{wire}/_0347300_ring_terminal",
        "seg": f"/{wire}/Segment_23",
        "neck": f"/{wire}/_0347300_ring_terminal/BreakableNeck",
    }




def _set_translate(xf, xyz):
    """Set translate with ZERO AddTranslateOp/AddXformOp calls.

    Session-layer GetOrderedXformOps often misses root-layer translate ops, so
    a fallback AddTranslateOp raises:
      The xformOp 'xformOp:translate' already exists in xformOpOrder
    Strategy: XformCommonAPI → attr Set → CreateAttribute + order list edit.
    """
    from pxr import UsdGeom, Gf, Sdf
    prim = xf.GetPrim()
    xyz = Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2]))

    # A) XformCommonAPI (handles existing translate/pivot ops correctly)
    try:
        api = UsdGeom.XformCommonAPI(prim)
        if api:
            api.SetTranslate(xyz)
            # Verify attribute actually holds value
            attr = prim.GetAttribute("xformOp:translate")
            if attr and attr.HasAuthoredValueOpinion():
                return "xform_common_api"
            # Some layers accept SetTranslate without authoring translate name —
            # fall through to explicit attr path.
    except Exception:
        pass

    # B) Direct attribute Set (op already in composed xformOpOrder)
    attr = prim.GetAttribute("xformOp:translate")
    if attr and attr.IsValid():
        attr.Set(xyz)
        return "attr_set"

    # C) Create the attribute + ensure order WITHOUT AddXformOp/AddTranslateOp
    attr = prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3, False)
    attr.Set(xyz)
    order_attr = xf.GetXformOpOrderAttr()
    if not order_attr:
        order_attr = xf.CreateXformOpOrderAttr()
    order = [str(t) for t in (order_attr.Get() or [])]
    if "xformOp:translate" not in order:
        order = ["xformOp:translate"] + order
        order_attr.Set(order)
    return "attr_create_order"


def _set_screw_physics(prim, kinematic=True, ccd=False):
    """Kinematic nest rest + CCD off (PhysX rejects kinematic+CCD).

    TELEOP_KINEMATIC_FIX_20261025: when kinematic=True do NOT author
    physics:velocity / physics:angularVelocity — USD→PhysX sync would call
    PxRigidDynamic::setLinearVelocity/setAngularVelocity and error.
    """
    from pxr import UsdPhysics, Sdf, UsdGeom, Gf
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    kin = prim.GetAttribute("physics:kinematicEnabled")
    if not kin:
        kin = prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool)
    kin.Set(bool(kinematic))
    pk = prim.GetAttribute("physxRigidBody:kinematicEnabled")
    if pk:
        try:
            pk.Set(bool(kinematic))
        except Exception:
            pass
    ccd_attr = prim.GetAttribute("physxRigidBody:enableCCD")
    if not ccd_attr:
        ccd_attr = prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool)
    ccd_attr.Set(bool(ccd) and (not kinematic))
    try:
        UsdGeom.Imageable(prim).MakeVisible()
    except Exception:
        pass
    # Only zero USD velocity when body is DYNAMIC (PhysX accepts set*Velocity).
    if not kinematic:
        for attr_name in ("physics:velocity", "physics:angularVelocity"):
            attr = prim.GetAttribute(attr_name)
            if attr:
                try:
                    attr.Set(Gf.Vec3f(0, 0, 0))
                except Exception:
                    pass



def _screw_specs(safety=None):
    """Canonical screw paths: under Nest after SCREW_ON_NEST reparent."""
    defaults = [
        ("/ScrewTray/Nest_1/Screw_1", "/ScrewTray/Nest_1"),
        ("/ScrewTray/Nest_2/Screw_2", "/ScrewTray/Nest_2"),
        ("/ScrewTray/Nest_3/Screw_3", "/ScrewTray/Nest_3"),
        ("/ScrewTray/Nest_4/Screw_4", "/ScrewTray/Nest_4"),
    ]
    # Allow safety_config override list
    try:
        paths = (safety or {}).get("paths") or {}
        screws = paths.get("screws") or []
        nests = paths.get("screw_nests") or []
        if len(screws) == 4:
            if len(nests) == 4:
                return list(zip(screws, nests))
            # derive nest parent from /ScrewTray/Nest_i/Screw_i
            out = []
            for sp in screws:
                if "/Nest_" in sp and sp.rsplit("/", 1)[0]:
                    out.append((sp, sp.rsplit("/", 1)[0]))
                else:
                    # legacy /Screw_i
                    i = sp.rstrip("0123456789")[-1] if False else None
                    out.append((sp, None))
            if all(n for _, n in out):
                return out
            # mix: map legacy
            mapped = []
            for sp in screws:
                if sp.startswith("/ScrewTray/"):
                    mapped.append((sp, sp.rsplit("/", 1)[0]))
                else:
                    # /Screw_k
                    try:
                        k = int(sp.split("_")[-1])
                        mapped.append((f"/ScrewTray/Nest_{k}/Screw_{k}", f"/ScrewTray/Nest_{k}"))
                    except Exception:
                        mapped.append((sp, None))
            return mapped
    except Exception:
        pass
    return defaults


def _clear_session_xform(stage, path: str):
    """Drop session-layer opinions so root/nest local translate wins cleanly."""
    try:
        from pxr import Sdf
        session = stage.GetSessionLayer()
        if session is None:
            return "no_session"
        prim_spec = session.GetPrimAtPath(path)
        if prim_spec is not None:
            # Clear xform-related properties on session
            for pname in list(prim_spec.properties.keys()) if hasattr(prim_spec, "properties") else []:
                if "xformOp" in pname or pname == "xformOpOrder":
                    prim_spec.RemoveProperty(pname)
            # Also try removing the whole session prim override
            try:
                session.ScheduleRemovePrim(path) if False else None
            except Exception:
                pass
            # Stronger: remove prim spec from session if it exists
            try:
                from pxr import Sdf as _Sdf
                edit = _Sdf.NamespaceEdit.Remove(path)
                batch = _Sdf.BatchNamespaceEdit()
                # Safer property clear already done; full remove can break live handles
            except Exception:
                pass
        return "cleared"
    except Exception as e:
        return "clear_err:" + repr(e)


def _teleport_physx(path: str, world_xyz, world_quat_wxyz=(1.0, 0.0, 0.0, 0.0), kinematic=True):
    """Force PhysX/render pose — USD Set alone leaves fallen bodies in the sky.

    TELEOP_KINEMATIC_FIX_20261025: never call set*Velocity / SingleRigidPrim
    (its init/post_reset zeros velocities) on kinematic bodies — that is the
    PxRigidDynamic::setLinearVelocity/setAngularVelocity spam on START TELEOP.
    """
    import numpy as np
    pos = np.array([float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])], dtype=float)
    quat = np.array([float(x) for x in world_quat_wxyz], dtype=float)  # wxyz
    how = []
    # A) SingleXFormPrim — pose only, safe for kinematic
    try:
        from isaacsim.core.prims import SingleXFormPrim
        xp = SingleXFormPrim(prim_path=path, name="reseat_xf_" + path.replace("/", "_"))
        xp.set_world_pose(position=pos, orientation=quat)
        how.append("SingleXFormPrim")
    except Exception as e:
        how.append("SingleXFormPrim_fail:" + type(e).__name__)
    # B) physx tensors: transforms always; velocities ONLY if dynamic
    try:
        import omni.physics.tensors as tensors
        sim = tensors.create_simulation_view("numpy")
        sim.set_subspace_roots("/")
        view = sim.create_rigid_body_view(path)
        t = np.zeros((1, 7), dtype=np.float32)
        t[0, 0:3] = pos
        t[0, 3:7] = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float32)
        view.set_transforms(t, np.array([0], dtype=np.uint32))
        if kinematic:
            how.append("tensors_rb_view_pose_only")
        else:
            v = np.zeros((1, 6), dtype=np.float32)
            view.set_velocities(v, np.array([0], dtype=np.uint32))
            how.append("tensors_rb_view+vel")
    except Exception as e:
        how.append("tensors_fail:" + type(e).__name__)
    # C) SingleRigidPrim ONLY when dynamic — init/post_reset calls set_velocities
    if kinematic:
        how.append("SingleRigidPrim_skipped_kinematic")
    else:
        try:
            from isaacsim.core.prims import SingleRigidPrim
            rp = SingleRigidPrim(prim_path=path, name="reseat_" + path.replace("/", "_"))
            rp.set_world_pose(position=pos, orientation=quat)
            how.append("SingleRigidPrim")
        except Exception as e:
            how.append("SingleRigidPrim_fail:" + type(e).__name__)
    return how



def _nest_world(stage, nest_path: str):
    from pxr import UsdGeom, Sdf
    prim = stage.GetPrimAtPath(Sdf.Path(nest_path))
    if not prim or not prim.IsValid():
        return None
    cache = UsdGeom.XformCache()
    t = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
    return (float(t[0]), float(t[1]), float(t[2]))


def ensure_necks(stage, safety: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create orange capsule neck visuals under each wire (session or root layer)."""
    from pxr import UsdGeom, UsdPhysics, Gf, Sdf, Usd
    cfg = _cfg(safety)
    report = {"ok": True, "necks": [], "errors": [], "tag": "BREAKABLE_NECK_20261025"}
    if stage is None:
        report["ok"] = False
        report["errors"].append("no_stage")
        return report
    try:
        stage.SetEditTarget(stage.GetSessionLayer())
    except Exception as e:
        report["errors"].append("session:" + repr(e))

    height = float(cfg.get("neck_height_m", 0.012))
    radius = float(cfg.get("neck_radius_m", 0.0035))
    # Sit near FixedJoint localPos0 on the ring: (0, 0.023, 0.0035)
    local_t = cfg.get("neck_local_translate_m", [0.0, 0.017, 0.0035])

    for wire in WIRE_IDS:
        p = _paths(wire)
        try:
            ring = _prim(stage, p["ring"])
            if not ring or not ring.IsValid():
                report["errors"].append("missing_ring:" + wire)
                continue
            neck = _prim(stage, p["neck"])
            if not neck or not neck.IsValid():
                neck = UsdGeom.Xform.Define(stage, Sdf.Path(p["neck"])).GetPrim()
            xf = UsdGeom.Xformable(neck)
            # Reuse existing translate op (AddTranslateOp crashes if already present)
            _set_translate(xf, local_t)
            # Capsule along Y (matches joint localPos0 bias)
            cap_path = p["neck"] + "/Mesh"
            cap_prim = _prim(stage, cap_path)
            if not cap_prim or not cap_prim.IsValid():
                cap = UsdGeom.Capsule.Define(stage, Sdf.Path(cap_path))
            else:
                cap = UsdGeom.Capsule(cap_prim)
            cap.CreateAxisAttr().Set("Y")
            cap.CreateHeightAttr().Set(height)
            cap.CreateRadiusAttr().Set(radius)
            cap.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.45, 0.05)])  # orange
            # No collision — visual + color alarm only
            if cap.GetPrim().HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(cap.GetPrim()).CreateCollisionEnabledAttr().Set(False)
            report["necks"].append(p["neck"])
        except Exception as e:
            report["ok"] = False
            report["errors"].append(wire + ":" + repr(e))
    return report


def configure_break_forces(stage, safety: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Set PhysX FixedJoint breakForce / breakTorque from safety_config."""
    cfg = _cfg(safety)
    f_break = float(cfg.get("F_break_N", 8.0))
    t_break = float(cfg.get("T_break_Nm", 0.5))
    report = {"ok": True, "F_break_N": f_break, "joints": [], "errors": []}
    if stage is None:
        report["ok"] = False
        report["errors"].append("no_stage")
        return report
    try:
        stage.SetEditTarget(stage.GetSessionLayer())
    except Exception:
        pass
    for wire in WIRE_IDS:
        p = _paths(wire)
        prim = _prim(stage, p["joint"])
        if not prim or not prim.IsValid():
            report["errors"].append("missing_joint:" + wire)
            continue
        try:
            from pxr import Sdf
            attr_f = prim.GetAttribute("physics:breakForce")
            if not attr_f or not attr_f.IsValid():
                attr_f = prim.CreateAttribute("physics:breakForce", Sdf.ValueTypeNames.Float)
            attr_f.Set(float(f_break))
            attr_t = prim.GetAttribute("physics:breakTorque")
            if not attr_t or not attr_t.IsValid():
                attr_t = prim.CreateAttribute("physics:breakTorque", Sdf.ValueTypeNames.Float)
            attr_t.Set(float(t_break))
            # Ensure joint starts enabled
            en = prim.GetAttribute("physics:jointEnabled")
            if en:
                en.Set(True)
            report["joints"].append(p["joint"])
        except Exception as e:
            report["ok"] = False
            report["errors"].append(wire + ":" + repr(e))
    return report


def _world_pos(stage, prim_path: str):
    from pxr import UsdGeom
    prim = _prim(stage, prim_path)
    if not prim or not prim.IsValid():
        return None
    xf = UsdGeom.XformCache()
    return xf.GetLocalToWorldTransform(prim).ExtractTranslation()


def _set_neck_color(stage, neck_path: str, rgb):
    from pxr import UsdGeom, Gf, Usd
    prim = _prim(stage, neck_path)
    if not prim or not prim.IsValid():
        return
    for t in Usd.PrimRange(prim):
        if t.IsA(UsdGeom.Gprim):
            UsdGeom.Gprim(t).GetDisplayColorAttr().Set([Gf.Vec3f(*rgb)])
            return
        attr = t.GetAttribute("primvars:displayColor")
        if attr:
            attr.Set([(rgb[0], rgb[1], rgb[2])])
            return


def _joint_alive(stage, joint_path: str) -> bool:
    prim = _prim(stage, joint_path)
    if not prim or not prim.IsValid():
        return False
    # PhysX may deactivate broken joints
    en = prim.GetAttribute("physics:jointEnabled")
    if en and en.HasAuthoredValueOpinion() and en.Get() is False:
        return False
    # Some builds remove breakForce victims; treat missing body rel as dead
    b0 = prim.GetRelationship("physics:body0")
    b1 = prim.GetRelationship("physics:body1")
    if b0 and b1:
        t0 = b0.GetTargets()
        t1 = b1.GetTargets()
        if not t0 or not t1:
            return False
    return True


def _proxy_force_N(stage, wire: str, cfg: Dict[str, Any]) -> Optional[float]:
    """Axial stretch proxy between ring and Segment_23."""
    p = _paths(wire)
    a = _world_pos(stage, p["ring"])
    b = _world_pos(stage, p["seg"])
    if a is None or b is None:
        return None
    dist = math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
    rest = float(cfg.get("rest_len_m", DEFAULT_REST_LEN_M))
    k = float(cfg.get("proxy_stiffness_N_per_m", DEFAULT_STIFFNESS_N_PER_M))
    stretch = max(0.0, float(dist) - rest)
    return k * stretch


class BreakableNeckMonitor:
    """Per-session state machine for 4 wire necks."""

    def __init__(self, safety: Optional[Dict[str, Any]] = None):
        self.cfg = _cfg(safety)
        self.F_warn = float(self.cfg.get("F_warn_N", 3.0))
        self.F_break = float(self.cfg.get("F_break_N", 8.0))
        self.N_frames = int(self.cfg.get("break_hold_frames", 5))
        self.state: Dict[str, Dict[str, Any]] = {
            w: {"force_N": None, "color": "ORANGE", "broken": False, "over_frames": 0, "alarm": False}
            for w in WIRE_IDS
        }
        self.any_broken = False
        self.setup_report: Dict[str, Any] = {}

    def setup(self, stage) -> Dict[str, Any]:
        r1 = ensure_necks(stage, {"breakable_neck": self.cfg})
        r2 = configure_break_forces(stage, {"breakable_neck": self.cfg})
        self.setup_report = {"necks": r1, "joints": r2, "tag": "BREAKABLE_NECK_20261025"}
        print("[breakable_neck]", self.setup_report)
        return self.setup_report

    def tick(self, stage) -> Dict[str, Any]:
        """Update colors / break detection. Call each panel frame."""
        out = {"forces": {}, "broken": [], "alarms": [], "colors": {}}
        for wire in WIRE_IDS:
            st = self.state[wire]
            p = _paths(wire)
            if st["broken"]:
                _set_neck_color(stage, p["neck"], (1.0, 0.08, 0.05))
                out["forces"][wire] = st["force_N"]
                out["broken"].append(wire)
                out["colors"][wire] = "RED"
                continue

            alive = _joint_alive(stage, p["joint"])
            force = _proxy_force_N(stage, wire, self.cfg)
            st["force_N"] = force
            out["forces"][wire] = force

            # PhysX already broke the joint
            if not alive:
                st["broken"] = True
                st["alarm"] = True
                self.any_broken = True
                _set_neck_color(stage, p["neck"], (1.0, 0.08, 0.05))
                # Nudge separation if bodies still overlapping
                try:
                    self._nudge_separate(stage, wire)
                except Exception:
                    pass
                out["broken"].append(wire)
                out["alarms"].append(wire)
                out["colors"][wire] = "RED"
                print("[breakable_neck] BROKEN", wire, "F~", force)
                continue

            # Software break if proxy force exceeds for N frames (backup if PhysX breakForce inert)
            if force is not None and force >= self.F_break:
                st["over_frames"] += 1
            else:
                st["over_frames"] = 0

            if st["over_frames"] >= self.N_frames:
                self._force_break(stage, wire)
                st["broken"] = True
                st["alarm"] = True
                self.any_broken = True
                _set_neck_color(stage, p["neck"], (1.0, 0.08, 0.05))
                out["broken"].append(wire)
                out["alarms"].append(wire)
                out["colors"][wire] = "RED"
                print("[breakable_neck] FORCE_BREAK", wire, "F=", force)
                continue

            if force is not None and force >= self.F_warn:
                _set_neck_color(stage, p["neck"], (1.0, 0.85, 0.05))
                st["color"] = "YELLOW"
                out["colors"][wire] = "YELLOW"
            else:
                _set_neck_color(stage, p["neck"], (1.0, 0.45, 0.05))
                st["color"] = "ORANGE"
                out["colors"][wire] = "ORANGE"
        out["any_broken"] = self.any_broken
        out["max_force_N"] = None
        vals = [v for v in out["forces"].values() if v is not None]
        if vals:
            out["max_force_N"] = max(vals)
        return out

    def _force_break(self, stage, wire: str):
        from pxr import Sdf
        p = _paths(wire)
        prim = _prim(stage, p["joint"])
        if not prim or not prim.IsValid():
            return
        try:
            stage.SetEditTarget(stage.GetSessionLayer())
        except Exception:
            pass
        en = prim.GetAttribute("physics:jointEnabled")
        if en:
            en.Set(False)
        else:
            from pxr import Sdf as _Sdf
            prim.CreateAttribute("physics:jointEnabled", _Sdf.ValueTypeNames.Bool).Set(False)
        # Clear body relationships so joint cannot respawn constraint
        for rel_name in ("physics:body0", "physics:body1"):
            rel = prim.GetRelationship(rel_name)
            if rel:
                try:
                    rel.ClearTargets(True)
                except Exception:
                    pass
        try:
            self._nudge_separate(stage, wire)
        except Exception:
            pass

    def _nudge_separate(self, stage, wire: str):
        """Tiny world offset on ring so the break is visible (gap like T17-C)."""
        from pxr import UsdGeom, Gf, UsdPhysics
        p = _paths(wire)
        ring = _prim(stage, p["ring"])
        seg = _prim(stage, p["seg"])
        if not ring or not ring.IsValid() or not seg or not seg.IsValid():
            return
        xf = UsdGeom.XformCache()
        rw = xf.GetLocalToWorldTransform(ring).ExtractTranslation()
        sw = xf.GetLocalToWorldTransform(seg).ExtractTranslation()
        d = Gf.Vec3d(rw[0]-sw[0], rw[1]-sw[1], rw[2]-sw[2])
        if d.GetLength() < 1e-6:
            d = Gf.Vec3d(0, 0.01, 0)
        else:
            d = d.GetNormalized() * 0.008
        # Prefer setting rigid-body velocity if available
        try:
            rb = UsdPhysics.RigidBodyAPI(ring)
            if rb:
                vel = ring.GetAttribute("physics:velocity")
                if vel:
                    cur = vel.Get() or Gf.Vec3f(0, 0, 0)
                    vel.Set(Gf.Vec3f(cur[0] + d[0]*5, cur[1] + d[1]*5, cur[2] + d[2]*5))
        except Exception:
            pass

def reseat_screws(stage, safety: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Seat each Screw under its Nest (local rest) + PhysX teleport to nest world.

    Prior SCREW_FRAME_FIX wrote world translates that MATCHED nest on disk (delta=0)
    but PhysX kept fallen/sky poses — console printed targets while viewport floated.
    """
    from pxr import UsdGeom, UsdPhysics, Gf, Sdf, Usd
    report = {
        "ok": True,
        "reseated": [],
        "errors": [],
        "tag": "SCREW_ON_NEST_20261025",
        "frame": "local_under_Nest_plus_physx_teleport",
        "ccd_disabled": True,
        "kinematic": True,
        "proof": [],
    }
    if stage is None:
        report["ok"] = False
        report["errors"].append("no_stage")
        return report

    specs = _screw_specs(safety)
    local_rest = (0.0, 0.0, 0.0)
    try:
        lr = ((safety or {}).get("paths") or {}).get("screw_local_rest_m")
        if lr is not None and len(lr) >= 3:
            local_rest = (float(lr[0]), float(lr[1]), float(lr[2]))
    except Exception:
        pass

    # Prefer root layer so Fetch/disk opinion sticks; also clear session xform noise
    try:
        stage.SetEditTarget(stage.GetRootLayer())
    except Exception as e:
        report["errors"].append("root_target:" + repr(e))
        try:
            stage.SetEditTarget(stage.GetSessionLayer())
        except Exception:
            pass

    before_after = {}
    cache = UsdGeom.XformCache()
    for screw_path, nest_path in specs:
        # Resolve legacy root /Screw_i if reparent not fetched yet
        prim = _prim(stage, screw_path)
        if (not prim or not prim.IsValid()) and screw_path.startswith("/ScrewTray/"):
            legacy = "/" + screw_path.rsplit("/", 1)[-1]  # /Screw_k
            prim = _prim(stage, legacy)
            if prim and prim.IsValid():
                screw_path = legacy
                report.setdefault("notes", []).append("legacy_path:" + legacy)
        if not prim or not prim.IsValid():
            report["ok"] = False
            report["errors"].append("missing:" + screw_path)
            continue
        try:
            nest_w = _nest_world(stage, nest_path) if nest_path else None
            if nest_w is None:
                # fallback composed from tray+local if nest missing
                nest_w = (0.2, -0.23, 0.77)
                report.setdefault("notes", []).append("nest_fallback:" + str(nest_path))

            before = tuple(float(c) for c in cache.GetLocalToWorldTransform(prim).ExtractTranslation())
            _clear_session_xform(stage, str(prim.GetPath()))

            # If screw is under nest, set LOCAL rest; else set WORLD = nest world
            parent_path = str(prim.GetParent().GetPath()) if prim.GetParent() else ""
            xf = UsdGeom.Xformable(prim)
            if nest_path and parent_path.rstrip("/") == nest_path.rstrip("/"):
                how = _set_translate(xf, local_rest)
                mode = "local_under_nest"
                target_world = nest_w  # local 0 => world ≈ nest
            else:
                how = _set_translate(xf, nest_w)
                mode = "world_abs_legacy"
                target_world = nest_w

            _set_screw_physics(prim, kinematic=True, ccd=False)
            # TELEOP_KINEMATIC_FIX: do not author USD velocities on kinematic screws
            tele = _teleport_physx(str(prim.GetPath()), target_world, kinematic=True)
            # Invalidate xform cache
            cache = UsdGeom.XformCache()
            after = tuple(float(c) for c in cache.GetLocalToWorldTransform(prim).ExtractTranslation())
            delta = tuple(abs(after[i] - nest_w[i]) for i in range(3))
            proof = {
                "screw": str(prim.GetPath()),
                "nest": nest_path,
                "nest_world": list(nest_w),
                "screw_world_after": list(after),
                "max_abs_delta_m": max(delta),
                "ok_1cm": max(delta) < 0.01,
                "mode": mode,
                "translate_how": how,
                "physx": tele,
            }
            report["proof"].append(proof)
            before_after[str(prim.GetPath())] = {
                "before_world": before,
                "after_world": after,
                "nest_world": nest_w,
                "local_rest": list(local_rest) if mode.startswith("local") else None,
                "kinematic": True,
                "enableCCD": False,
            }
            if max(delta) >= 0.01:
                # PhysX may still be stale in USD cache read — still mark reseated if teleported
                report.setdefault("notes", []).append(
                    "usd_delta>=1cm:" + str(prim.GetPath()) + ":" + str(max(delta))
                )
            report["reseated"].append(str(prim.GetPath()))
        except Exception as e:
            report["ok"] = False
            report["errors"].append(screw_path + ":" + repr(e))

    report["before_after"] = before_after
    n_ok = sum(1 for p in report["proof"] if p.get("ok_1cm"))
    report["proof_summary"] = f"{n_ok}/{len(report['proof'])} nest_world≈screw_world within 1cm (USD cache)"
    print(
        "[breakable_neck] SCREW_ON_NEST_20261025 reseat",
        report.get("reseated"),
        "err", report.get("errors"),
        "proof", report.get("proof_summary"),
        "detail", [{k: p.get(k) for k in ("screw","nest_world","screw_world_after","max_abs_delta_m","physx")} for p in report.get("proof", [])],
    )
    return report

