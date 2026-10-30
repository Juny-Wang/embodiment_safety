"""Task2 keyboard teleop + StepFSM + safety color skeleton.

Run from Isaac Sim Script Editor AFTER opening riser_io_preview.usda.
Does NOT depend on missing Task1 live_safety_feedback base.
"""
from __future__ import annotations
from pathlib import Path
import sys, json, time, math, copy, traceback

PACKAGE = Path(__file__).resolve().parent
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

import carb
import omni.usd
import omni.timeline
import omni.physx
import omni.ui as ui
from pxr import UsdPhysics, PhysicsSchemaTools
import numpy as np

import keyboard_input
from franka_teleop import FrankaTeleop
from safety_core import SafetyEngine, COLORS
import step_fsm
import plan_d_runtime

# Robust load: Kit Script Editor often lacks scene/ on sys.path after exec().
import importlib
import importlib.util
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))
try:
    import breakable_neck
    breakable_neck = importlib.reload(breakable_neck)
except ModuleNotFoundError:
    _bn_path = PACKAGE / "breakable_neck.py"
    if not _bn_path.is_file():
        _bn_path = Path("/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/breakable_neck.py")
    if not _bn_path.is_file():
        raise
    _spec = importlib.util.spec_from_file_location("breakable_neck", str(_bn_path))
    breakable_neck = importlib.util.module_from_spec(_spec)
    sys.modules["breakable_neck"] = breakable_neck
    _spec.loader.exec_module(breakable_neck)
    print("[keyboard_panel] breakable_neck loaded via file:", _bn_path)
else:
    print("[keyboard_panel] breakable_neck OK from", getattr(breakable_neck, "__file__", "?"))
import cameras_setup
import camera_viewports

CONFIG = json.loads((PACKAGE / "safety_config.json").read_text())
CALIB = json.loads((PACKAGE / "geometry_calib.json").read_text())
CONFIG = plan_d_runtime.merge_calib_into_safety(CONFIG, CALIB)
STEPS = step_fsm.STEPS
STEP_LABELS = step_fsm.STEP_LABELS
print("[keyboard_panel] FIX3_TELEOP_20261025 + FIX4_INPUT + SCREW_ON_NEST_20261025")


def rgba(c):
    r, g, b, a = c
    return int(a * 255) << 24 | int(b * 255) << 16 | int(g * 255) << 8 | int(r * 255)


class Task2KeyboardPanel:
    def __init__(self):
        self.config = CONFIG
        self.engine = SafetyEngine(self._engine_config())
        self.controller = FrankaTeleop(self.config)
        self.fsm = step_fsm.StepFSM(CALIB, self.config)
        self.step_index = 0
        self.metrics = {}
        self._physics_dt = None
        self._contact_peaks = {}
        self._physics_serial = 0
        self._last_consumed_serial = -1
        self._contact_points = []
        self.safety = {"rows": [], "stop": False, "latched_reasons": []}
        self.ack = False
        self.enabled = False
        self.held = set()
        self.user_message = "Pause → run script → Play → Initialize → ack → ENABLE"
        self.dt = 0.001
        self.t = 0.0
        self.contact_force_N = 0.0
        self.finger_left_N = 0.0
        self.finger_right_N = 0.0
        self.commands = 0
        self.last_key = ""
        self.accept_reject = ""
        self.keyboard_events = 0
        self._sub_update = None
        self._sub_input = None
        self._sub_contact = None
        self.timeline = omni.timeline.get_timeline_interface()
        self.stage = omni.usd.get_context().get_stage()
        self.logdir = PACKAGE / "runs" / time.strftime("%Y%m%d_%H%M%S")
        self.logdir.mkdir(parents=True, exist_ok=True)
        (self.logdir / "config_snapshot.json").write_text(json.dumps(self.config, indent=2))
        self.event_log = (self.logdir / "keyboard_events.jsonl").open("w", buffering=1)
        try:
            self._plan_d_report = plan_d_runtime.apply_session_plan_d(self.stage, CALIB, self.config)
        except Exception as e:
            self._plan_d_report = {"ok": False, "errors": [repr(e)]}

        # Breakable neck monitor (orange crimp proxy; screws reseated in plan_d)
        try:
            self._neck = breakable_neck.BreakableNeckMonitor(self.config)
            self._neck_report = getattr(self, "_plan_d_report", {}) or {}
            # Re-run setup so session necks exist even if plan_d partial-failed
            self._neck_report = self._neck.setup(self.stage)
            print("[keyboard_panel] BREAKABLE_NECK_20261025", self._neck_report)
        except Exception as _ne:
            self._neck = None
            self._neck_report = {"ok": False, "errors": [repr(_ne)]}
            print("[keyboard_panel] breakable_neck FAILED", self._neck_report)

        # Session cameras + right-side mini viewports (ported from Task1).
        try:
            self._cameras_report = cameras_setup.apply_session_cameras(
                self.stage, self.config, CALIB
            )
            print(
                "[keyboard_panel] cameras",
                self._cameras_report.get("paths"),
                "ok",
                self._cameras_report.get("ok"),
                "err",
                self._cameras_report.get("errors"),
            )
        except Exception as _ce:
            self._cameras_report = {"ok": False, "errors": [repr(_ce)]}
            print("[keyboard_panel] cameras FAILED", self._cameras_report)
        try:
            self._cam_split_report = camera_viewports.open_camera_split(
                self.stage, self.config, getattr(self, "_cameras_report", None)
            )
            print("[keyboard_panel] cam_split", self._cam_split_report)
        except Exception as _se:
            self._cam_split_report = {"ok": False, "errors": [repr(_se)]}
            print("[keyboard_panel] cam_split FAILED", self._cam_split_report)
        self._build_ui()
        for prim in self.stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and (str(prim.GetPath()).startswith("/World/Robot") or "ring_terminal" in str(prim.GetPath()) or str(prim.GetPath()).startswith("/Screw_")):
                prim.AddAppliedSchema("PhysxContactReportAPI")
                from pxr import Sdf
                prim.CreateAttribute("physxContactReport:threshold",Sdf.ValueTypeNames.Float).Set(0.)
        self._physics_subscription=omni.physx.get_physx_interface().subscribe_physics_step_events(self._on_physics_step)
        self._hook_contacts()
        self._hook_update()

    def _engine_config(self):
        # SafetyEngine expects rules with int steps + top-level warning ratios
        cfg = copy.deepcopy(CONFIG)
        cfg.setdefault("warning_ratio", CONFIG.get("traffic_light", {}).get("warning_ratio", 0.7))
        cfg.setdefault("warning_clear_ratio", CONFIG.get("traffic_light", {}).get("warning_clear_ratio", 0.7))
        cfg.setdefault("contact_gate_N", CONFIG.get("contact_gate_N", 0.05))
        cfg.setdefault("status", CONFIG.get("status", "demo"))
        # drop rules with empty steps (deferred)
        cfg["rules"] = [r for r in cfg.get("rules", []) if r.get("steps")]
        return cfg


    def show_cam_split(self):
        """Recreate the three mini camera viewports (safe to click anytime)."""
        try:
            stage = self.stage or omni.usd.get_context().get_stage()
            if not getattr(self, "_cameras_report", None) or not (self._cameras_report or {}).get("ok"):
                self._cameras_report = cameras_setup.apply_session_cameras(
                    stage, self.config, CALIB
                )
                print("[keyboard_panel] cameras(re)", self._cameras_report)
            self._cam_split_report = camera_viewports.open_camera_split(
                stage, self.config, getattr(self, "_cameras_report", None)
            )
            print("[keyboard_panel] cam_split(re)", self._cam_split_report)
            n = len((self._cam_split_report or {}).get("windows") or [])
            self.user_message = "Camera split windows: %d" % n
        except Exception as e:
            self.user_message = "Camera split failed: " + repr(e)
            print("[keyboard_panel] cam_split FAILED", repr(e))


    def _build_ui(self):
        self.ui = ui.Window("Task2 riser_io keyboard + safety", width=720, height=780)
        with self.ui.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6):
                    ui.Label("TASK2 PROTOTYPE — Phase1 wire then Phase2 screw. Demo thresholds.", word_wrap=True)
                    ui.Label("Scene: riser_io_preview.usda | Esc=E-stop | arrows/PgUp/Dn move | Shift=rot | [/]=grip", word_wrap=True)
                    self.status_label = ui.Label(self.user_message, word_wrap=True)
                    self.keys_label = ui.Label("Keys: —", word_wrap=True)
                    self.gate_label = ui.Label("Gates: —", word_wrap=True)
                    ui.Label("Object index (wire / ring / screw share the same index)")
                    self.object_combo=ui.ComboBox(0,"1","2","3","4")
                    ui.Label("WARNING ONLY | Print prototype | Torque and wire tension uncalibrated")
                    ui.Label("Step")
                    self.step_combo = ui.ComboBox(0, *STEP_LABELS)
                    self.step_combo.model.get_item_value_model().add_value_changed_fn(lambda *_: self._on_step())
                    with ui.HStack(spacing=8):
                        ui.Button("▶ START TELEOP", clicked_fn=self.start_teleop, height=36)
                        ui.Button("STOP", clicked_fn=self.user_stop)
                        ui.Button("SHOW CAM SPLIT", clicked_fn=self.show_cam_split)
                    with ui.HStack(spacing=8):
                        ui.Button("INITIALIZE", clicked_fn=self.initialize)
                        ui.Button("ACK thresholds", clicked_fn=self.acknowledge)
                        ui.Button("ENABLE", clicked_fn=self.enable)
                    self.metric_labels = {}
                    for key, title in [
                        ("grip_force_N", "Grip N"),
                        ("wire_joint_force", "Wire joint N"),
                        ("align_xy_m", "Axis offset m"),
                        ("tilt_deg", "Axis tilt deg"),
                        ("ring_lateral_N", "Ring lateral N"),
                        ("ring_axial_N", "Ring axial N"),
                        ("screw_axial_N", "Screw axial N"),
                        ("finger_left_N", "Finger L"),
                        ("finger_right_N", "Finger R"),
                        ("neck_force_N", "Neck proxy N"),
                        ("neck_state", "Neck state"),
                    ]:
                        with ui.HStack():
                            ui.Label(title, width=140)
                            self.metric_labels[key] = ui.Label("—")
                    ui.Label("Traffic / reasons")
                    self.reason_label = ui.Label("(none)", word_wrap=True)

    def _on_step(self):
        idx = int(self.step_combo.model.get_item_value_model().as_int)
        self.step_index = idx
        self.fsm.set_step(idx)
        self.user_message = "Step -> " + STEPS[idx]
        # changing step drops enable (Task1 pattern)
        if self.controller.armed:
            self.controller.stop("STEP_CHANGED")
        self.enabled = False
        self._refresh_status()


    def start_teleop(self):
        """One-click: Play → Initialize → arm → ENABLE. Short retries only (do not block Kit long)."""
        try:
            self.ack = True
            self.enabled = False
            # Re-apply cameras so wrist/external prims exist even if USDA session was stale.
            try:
                self._cameras_report = cameras_setup.apply_session_cameras(
                    self.stage or omni.usd.get_context().get_stage(), self.config, CALIB
                )
                self._cam_split_report = camera_viewports.open_camera_split(
                    self.stage or omni.usd.get_context().get_stage(),
                    self.config,
                    self._cameras_report,
                )
                print("[keyboard_panel] FIX3 cams", self._cameras_report.get("paths"),
                      "split", (self._cam_split_report or {}).get("ok"))
            except Exception as _ce:
                print("[keyboard_panel] FIX3 cams warn", repr(_ce))
            # Reseat screws + refresh necks each START (Screw_4 often falls after Play)
            try:
                if str(PACKAGE) not in sys.path:
                    sys.path.insert(0, str(PACKAGE))
                import breakable_neck as _bn
                _sr = _bn.reseat_screws(self.stage or omni.usd.get_context().get_stage(), self.config)
                if getattr(self, "_neck", None) is None:
                    self._neck = _bn.BreakableNeckMonitor(self.config)
                self._neck.setup(self.stage or omni.usd.get_context().get_stage())
                print("[keyboard_panel] SCREW_ON_NEST_20261025 START reseat", _sr.get("tag"), _sr.get("reseated"), "err", _sr.get("errors"), "proof", _sr.get("proof_summary"))
            except Exception as _sr_e:
                print("[keyboard_panel] START reseat/neck warn", repr(_sr_e))
            if not self.timeline.is_playing():
                self.timeline.play()
            self.user_message = "START: waiting for physics backend…"
            self._refresh_status()
            last_err = None
            names = None
            # Prefer Play already running; only brief retries so Kit can still tick.
            for i in range(12):
                if not self.timeline.is_playing():
                    self.timeline.play()
                try:
                    names = self.controller.initialize(force=True)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    self.user_message = "START retry %d/12: %s — keep timeline Playing" % (i + 1, type(e).__name__)
                    if i % 2 == 0:
                        self._refresh_status()
                    time.sleep(0.15)
            if names is None:
                raise RuntimeError(
                    "Initialize failed (click Play first, wait 1s, START again): " + repr(last_err)
                )
            if not self.timeline.is_playing():
                self.timeline.play()
            self.controller.arm()
            # Wake drives: write current measured pose as targets.
            try:
                q = self.controller.get_joint_positions()
                self.controller.apply_targets(q)
                self.controller.last_action = q.copy()
            except Exception as _we:
                print("[keyboard_panel] wake drives warn", repr(_we))
            self.enabled = True
            self._register_input()
            self.user_message = (
                "READY (%d DOF) | click MAIN viewport | arrows=XY PgUp/Dn|Home/End=Z | [/]=grip"
                % len(names)
            )
        except Exception as e:
            self.enabled = False
            self.user_message = "START FAIL: " + repr(e)
            traceback.print_exc()
        self._refresh_status()


    def initialize(self):
        try:
            if not self.timeline.is_playing():
                self.timeline.play()
                self.user_message = "Auto Play started — wait ~1s, click INITIALIZE again"
                self.enabled = False
                self._refresh_status()
                return
            names = self.controller.initialize()
            self.user_message = "Initialized DOFs: %d. ACK then ENABLE." % len(names)
            self.enabled = False
        except Exception as e:
            self.user_message = "INIT FAIL: " + repr(e)
            traceback.print_exc()
        self._refresh_status()

    def acknowledge(self):
        self.ack = True
        self.engine.acknowledge(self.metrics, self.step_index, self.dt or 0.001)
        self.fsm.acknowledge()
        self.user_message = "Thresholds acknowledged"
        self._refresh_status()

    def enable(self):
        try:
            if not self.ack:
                raise RuntimeError("ACK thresholds first (tick the checkbox)")
            if not self.timeline.is_playing():
                self.timeline.play()
                self.user_message = "Auto Play started — wait ~1s, click ENABLE again"
                self.enabled = False
                self._refresh_status()
                return
            if not getattr(self.controller, "ready", False):
                names = self.controller.initialize()
                self.user_message = "Auto-initialized %d DOFs — click ENABLE again" % len(names)
                self.enabled = False
                self._refresh_status()
                return
            self.controller.arm()
            self.enabled = True
            self._register_input()
            self.user_message = "ENABLED — click MAIN viewport once, then arrows/PgUp/PgDn"
        except Exception as e:
            self.user_message = "ENABLE FAIL: " + repr(e)
            self.enabled = False
            traceback.print_exc()
        self._refresh_status()

    def user_stop(self):
        self.enabled = False
        try:
            if self.controller.ready:
                self.controller.stop("USER STOP")
        except Exception:
            pass
        self.timeline.pause()
        self.user_message = "STOPPED / paused"
        self._refresh_status()

    def _register_input(self):
        """Subscribe like Task1: InputEvent stream + optional keyboard-specific stream.

        FIX4: subscribe_to_input_events delivers InputEvent wrappers
        (deviceType + .event), NOT bare KeyboardEvent. Old Task2 handler read
        event.type/input on the wrapper → silent miss → events=0 forever.
        """
        import omni.appwindow
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        try:
            self._keyboard = self._appwindow.get_keyboard()
        except Exception:
            self._keyboard = None
        # Drop previous subs (remote sessions may replace keyboard device).
        if self._sub_input is not None:
            try:
                self._input.unsubscribe_to_input_events(self._sub_input)
            except Exception:
                pass
            self._sub_input = None
        kb_sub = getattr(self, "_sub_input_kb", None)
        if kb_sub is not None:
            try:
                if self._keyboard is not None:
                    self._input.unsubscribe_to_keyboard_events(self._keyboard, kb_sub)
                else:
                    self._input.unsubscribe_to_keyboard_events(kb_sub)
            except Exception:
                pass
            self._sub_input_kb = None
        self._sub_input = self._input.subscribe_to_input_events(self._on_input, order=-100000)
        self._sub_input_kb = None
        if self._keyboard is not None and hasattr(self._input, "subscribe_to_keyboard_events"):
            try:
                self._sub_input_kb = self._input.subscribe_to_keyboard_events(
                    self._keyboard, self._on_key
                )
            except Exception as _ke:
                print("[keyboard_panel] FIX4 keyboard_events subscribe skipped:", repr(_ke))
        print(
            "[keyboard_panel] FIX4_INPUT_20261025 sub=",
            self._sub_input,
            "kb_sub=",
            self._sub_input_kb,
            "kb=",
            self._keyboard,
        )

    def _on_input(self, event, *args):
        # InputEvent wrapper from subscribe_to_input_events (Task1 pattern).
        try:
            if getattr(event, "deviceType", None) != carb.input.DeviceType.KEYBOARD:
                return True
            ev = getattr(event, "event", None)
            if ev is None:
                return True
            try:
                if getattr(ev, "keyboard", None) is not None:
                    self._keyboard = ev.keyboard
            except Exception:
                pass
            return self._on_key(ev)
        except Exception:
            return True

    def _on_key(self, event, *unused):
        # KeyboardEvent (or keyboard-specific callback). Handle str/enum input.
        try:
            _inp = getattr(event, "input", None)
            if isinstance(_inp, str):
                key = _inp
            else:
                key = str(getattr(_inp, "name", _inp) or "")
            kind = getattr(event, "type", None)
            interesting = keyboard_input.KEYS | {"LEFT_SHIFT", "RIGHT_SHIFT", "ESCAPE"}
            if key not in interesting:
                return True
            self.keyboard_events += 1
            self.last_key = key
            if kind == carb.input.KeyboardEventType.KEY_RELEASE:
                self.held.discard(key)
                return not (self.enabled or key in self.held)
            if kind in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT):
                if key == "ESCAPE":
                    self.user_stop()
                    return False
                if self.enabled:
                    self.held.add(key)
                    self.accept_reject = "key_ok"
                else:
                    self.accept_reject = "key_ignored_disabled"
                return False if self.enabled else True
        except Exception as ex:
            self.accept_reject = "key_err:" + repr(ex)
            return True
        return True

    def _hook_contacts(self):
        try:
            self._sub_contact = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(
                lambda h, d: self._on_contacts(h, d)
            )
        except Exception as e:
            self.user_message = "contact sub failed: " + repr(e)

    def _on_physics_step(self,dt):
        if dt>0:
            self._physics_dt=float(dt)
            self._physics_serial+=1

    def _on_contacts(self,headers,data):
        if self._physics_dt is None:return
        dt=self._physics_dt
        idx=int(self.object_combo.model.get_item_value_model().as_int)
        ring=self.config["paths"]["ring_paths"][idx]
        screw=self.config["paths"]["screws"][idx]
        result={"finger_left_N":0.,"finger_right_N":0.,"ring_axial_N":0.,"ring_lateral_N":0.,"screw_axial_N":0.}
        points=[]
        try:
            for h in headers:
                a=str(PhysicsSchemaTools.intToSdfPath(h.collider0))
                b=str(PhysicsSchemaTools.intToSdfPath(h.collider1))
                pair=a+"|"+b
                finger="panda_leftfinger" in pair or "panda_rightfinger" in pair
                for k in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
                    c=data[k]
                    imp=np.asarray(c.impulse,dtype=float)
                    normal=np.asarray(c.normal,dtype=float)
                    force=normal*float(imp)/dt if imp.ndim==0 else imp/dt
                    fn=abs(float(np.dot(force,normal)))
                    if "panda_leftfinger" in pair:result["finger_left_N"]+=fn
                    if "panda_rightfinger" in pair:result["finger_right_N"]+=fn
                    # Target axes are world Z in this fixed assembly; exclude finger squeeze.
                    if ring in pair and not finger:
                        result["ring_axial_N"]+=abs(float(force[2]))
                        result["ring_lateral_N"]+=float(np.linalg.norm(force[:2]))
                    if screw in pair and not finger:result["screw_axial_N"]+=abs(float(force[2]))
                    if finger or ring in pair or screw in pair:
                        points.append({"point":list(c.position),"force_N":force.tolist(),"a":a,"b":b,"physics_dt":dt})
            for key,val in result.items():self._contact_peaks[key]=max(self._contact_peaks.get(key,0.),val)
            self._contact_points=points[:128]
        except Exception as ex:self.user_message="Contact measurement unavailable: "+repr(ex)

    def _hook_update(self):
        import omni.kit.app
        self._sub_update = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_update)

    def _on_update(self, e):
        try:
            dt = float(e.payload.get("dt", 0.0) or 0.0) if hasattr(e, "payload") else 0.0
            if dt <= 0:
                dt = 1.0 / 60.0
            self.dt = dt
            self.t += dt
            self.metrics["dt"] = dt
            idx=int(self.object_combo.model.get_item_value_model().as_int)
            self.config["selected_index"]=idx
            self.config["selected_phase"]=self.fsm.phase
            self.config["paths"]["wire_root"]="/wire"+str(idx+1)
            self.config["paths"]["ring_terminal"]=self.config["paths"]["ring_paths"][idx]
            self.config["paths"]["selected_screw"]=self.config["paths"]["screws"][idx]
            if self._physics_dt is not None and self._physics_serial!=self._last_consumed_serial:
                for key in ["finger_left_N","finger_right_N","ring_axial_N","ring_lateral_N","screw_axial_N"]:
                    self.metrics[key]=self._contact_peaks.get(key,0.)
                self.metrics["grip_force_N"]=max(self.metrics["finger_left_N"],self.metrics["finger_right_N"])
                self.contact_force_N=max(self.metrics["grip_force_N"],self.metrics["ring_axial_N"],self.metrics["screw_axial_N"])
                self._contact_peaks={}
                self._last_consumed_serial=self._physics_serial
            self.metrics["wire_joint_force"]=None
            self.metrics["screw_torque_Nm"]=None
            try:
                self.metrics.update(plan_d_runtime.compute_alignment_metrics(self.stage,CALIB,self.config))
            except Exception as _align_ex:
                # GetPrimAtPath(str) or missing prims must not kill the update loop.
                self.metrics["align_xy_m"] = None
                self.metrics["tilt_deg"] = None
                self.metrics["alignment_source"] = "error:" + repr(_align_ex)
            # Slow speeds on contact
            ctrl = self.config["control"]
            if self.contact_force_N > float(ctrl.get("contact_force_gate_N", 0.05)):
                # temporarily poke speeds via cfg used by teleop
                self.config["control"]["translation_speed_m_s"] = ctrl.get("contact_slow_translation_m_s", 0.001)
                self.config["control"]["rotation_speed_rad_s"] = ctrl.get("contact_slow_rotation_rad_s", 0.0349)
            else:
                self.config["control"]["translation_speed_m_s"] = 0.05
                self.config["control"]["rotation_speed_rad_s"] = 0.35

            if self.enabled and self.controller.armed:
                # Task1-style: set EE goal from held-key velocity * dt (do NOT accumulate tiny jogs).
                # Old per-frame jog() + 3mm tracking_error_stop immediately E-stopped motion.
                lin, ang = keyboard_input.vectors(self.held)
                try:
                    gp, gR = self.controller.goal
                    ctrl = self.config["control"]
                    v_lin = float(ctrl.get("translation_speed_m_s", 0.05))
                    v_ang = float(ctrl.get("rotation_speed_rad_s", 0.35))
                    if any(abs(x) > 0 for x in lin):
                        import numpy as _np
                        from scipy.spatial.transform import Rotation as _Rot
                        delta = _np.asarray(lin, dtype=float) * v_lin * float(dt)
                        self.controller.goal = (gp + delta, gR)
                        self.commands += 1
                        self.accept_reject = "lin"
                    if any(abs(x) > 0 for x in ang):
                        import numpy as _np
                        from scipy.spatial.transform import Rotation as _Rot
                        rv = _np.asarray(ang, dtype=float) * v_ang * float(dt)
                        self.controller.goal = (self.controller.goal[0], _Rot.from_rotvec(rv) * self.controller.goal[1])
                        self.commands += 1
                        self.accept_reject = "ang"
                except Exception as ex:
                    self.accept_reject = "reject_goal:" + repr(ex)
                    self.controller.stop(str(ex))
                    self.enabled = False
                g = keyboard_input.gripper_sign(self.held)
                if g:
                    try:
                        self.controller.gripper(open_gripper=(g > 0))
                    except Exception as ex:
                        self.accept_reject = "reject_grip:" + repr(ex)
                try:
                    self.controller.tick(dt)
                except Exception as ex:
                    self.user_message = "tick fail: " + repr(ex)
                    self.enabled = False

            # FSM gates + safety engine
            self.fsm.set_step(self.step_index)
            gate = self.fsm.evaluate_gates(self.metrics)
            self.safety = self.engine.evaluate(self.metrics, self.step_index, dt)
            if self.config.get("warning_only",False):
                self.safety["would_stop"]=self.safety.get("stop",False)
                self.safety["stop"]=False
                gate["would_halt"]=gate.get("halt",False)
                gate["halt"]=False
            if self.safety.get("stop") and self.controller.armed:
                self.controller.stop("SAFETY: " + ",".join(self.safety.get("latched_reasons") or []))
                self.enabled = False
                self.timeline.pause()
                self.user_message = "SAFETY STOP"

            # color alarms
            try:
                plan_d_runtime.apply_task2_alarms(self.stage, self.metrics, self.config)
                # Breakable neck color / break / alarm
                try:
                    if getattr(self, "_neck", None) is not None:
                        _nr = self._neck.tick(self.stage)
                        self.metrics["neck_force_N"] = _nr.get("max_force_N")
                        if _nr.get("any_broken"):
                            self.metrics["neck_state"] = "BROKEN:" + ",".join(_nr.get("broken") or [])
                            self.safety.setdefault("latched_reasons", [])
                            reason = "NECK_BREAK:" + ",".join(_nr.get("broken") or [])
                            if reason not in self.safety["latched_reasons"]:
                                self.safety["latched_reasons"].append(reason)
                            self.user_message = "NECK BREAK " + ",".join(_nr.get("broken") or [])
                        else:
                            cols = _nr.get("colors") or {}
                            self.metrics["neck_state"] = ",".join("%s=%s" % (k, cols[k]) for k in cols)
                except Exception as _nt:
                    self.metrics["neck_state"] = "err:" + type(_nt).__name__

            except Exception:
                pass

            if int(self.t * 10) % 3 == 0:
                self._refresh_status(gate)
                self._log_event(gate)
        except Exception:
            # Never spam stderr every frame (old GetPrim bugs flooded Console).
            self._update_err_n = getattr(self, "_update_err_n", 0) + 1
            if self._update_err_n <= 3 or (self._update_err_n % 180) == 0:
                traceback.print_exc()

    def _refresh_status(self, gate=None):
        gate = gate or self.fsm.last_gate
        stop = self.safety.get("stop")
        self.status_label.text = (
            ("STOP: " + ",".join(self.safety.get("latched_reasons") or []) if stop else "WARNING ONLY")
            + " | " + self.user_message
        )
        self.status_label.style = {"color": rgba(COLORS["RED"] if stop else COLORS["YELLOW"])}
        self.keys_label.text = (
            "events=%d last=%s enabled=%s cmds=%d held=%s | %s"
            % (self.keyboard_events, self.last_key, self.enabled, self.commands, sorted(self.held), self.accept_reject)
        )
        if gate:
            self.gate_label.text = (
                "phase=%s step=%s block_down=%s block_close=%s halt=%s lights=%s"
                % (gate.get("phase"), gate.get("step"), gate.get("block_down"), gate.get("block_close"),
                   gate.get("halt"), gate.get("lights"))
            )
        for k, lab in self.metric_labels.items():
            v = self.metrics.get(k)
            lab.text = ("%.4f" % v) if isinstance(v, (int, float)) else "—"
        self.reason_label.text = str(self.safety.get("latched_reasons") or self.fsm.halt_reasons or "(none)")

    def _log_event(self, gate):
        rec = {
            "t": self.t,
            "step": STEPS[self.step_index] if 0 <= self.step_index < len(STEPS) else None,
            "enabled": self.enabled,
            "held": sorted(self.held),
            "metrics": {k: self.metrics.get(k) for k in ("grip_force_N", "screw_axial_N", "ring_axial_N", "finger_left_N", "finger_right_N")},
            "gate": gate,
            "safety_stop": self.safety.get("stop"),
            "would_stop":self.safety.get("would_stop"),
            "selected_index":self.config.get("selected_index"),
            "contact_points":self._contact_points,
            "reasons": self.safety.get("latched_reasons"),
        }
        self.event_log.write(json.dumps(rec) + "\n")

    def close(self):
        self.enabled = False
        try:
            if getattr(self, "controller", None) is not None and getattr(self.controller, "ready", False):
                self.controller.stop("CLOSE")
        except Exception:
            pass
        # Drop subscriptions (Omni unsubscribes when Subscription is GC'd).
        self._sub_update = None
        self._sub_input = None
        self._sub_contact = None
        self._physics_subscription = None
        try:
            if getattr(self, "ui", None) is not None:
                self.ui.visible = False
                try:
                    self.ui.destroy()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if getattr(self, "event_log", None) is not None:
                self.event_log.close()
        except Exception:
            pass


# Persist panel across launch_panel re-exec (which uses a fresh globals dict).
import types as _types
_HOLDER = "_task2_panel_holder"
if _HOLDER not in sys.modules:
    sys.modules[_HOLDER] = _types.ModuleType(_HOLDER)
_holder = sys.modules[_HOLDER]
_old_panel = getattr(_holder, "panel", None)
if _old_panel is not None:
    try:
        _old_panel.close()
    except Exception:
        pass
    try:
        _holder.panel = None
    except Exception:
        pass
_task2_keyboard_panel = Task2KeyboardPanel()
_holder.panel = _task2_keyboard_panel
print("[task2] keyboard panel ready:", PACKAGE, "| GETPRIM_FIX", "20261024")
