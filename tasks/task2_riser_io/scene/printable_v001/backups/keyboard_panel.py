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

import keyboard_input
from franka_teleop import FrankaTeleop
from safety_core import SafetyEngine, COLORS
import step_fsm
import plan_d_runtime

CONFIG = json.loads((PACKAGE / "safety_config.json").read_text())
CALIB = json.loads((PACKAGE / "geometry_calib.json").read_text())
CONFIG = plan_d_runtime.merge_calib_into_safety(CONFIG, CALIB)
STEPS = step_fsm.STEPS
STEP_LABELS = step_fsm.STEP_LABELS


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
        self._build_ui()
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
                    ui.Label("Step")
                    self.step_combo = ui.ComboBox(0, *STEP_LABELS)
                    self.step_combo.model.get_item_value_model().add_value_changed_fn(lambda *_: self._on_step())
                    with ui.HStack(spacing=8):
                        ui.Button("INITIALIZE", clicked_fn=self.initialize)
                        ui.Button("ACK thresholds", clicked_fn=self.acknowledge)
                        ui.Button("ENABLE", clicked_fn=self.enable)
                        ui.Button("STOP", clicked_fn=self.user_stop)
                    self.metric_labels = {}
                    for key, title in [
                        ("grip_force_N", "Grip N"),
                        ("wire_joint_force", "Wire joint N"),
                        ("ring_axial_N", "Ring axial N"),
                        ("screw_axial_N", "Screw axial N"),
                        ("finger_left_N", "Finger L"),
                        ("finger_right_N", "Finger R"),
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

    def initialize(self):
        try:
            if not self.timeline.is_playing():
                raise RuntimeError("Press Play and wait for settle before Initialize")
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
                raise RuntimeError("ACK thresholds first")
            if not self.timeline.is_playing():
                raise RuntimeError("Play before ENABLE")
            self.controller.arm()
            self.enabled = True
            self._register_input()
            self.user_message = "ENABLED — click viewport; Esc stops"
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
        # Carb keyboard subscription
        try:
            import omni.appwindow
            app_window = omni.appwindow.get_default_app_window()
            self._input = carb.input.acquire_input_interface()
            if self._sub_input:
                self._sub_input = None
            # Match Task1: global input events; optional keyboard identity not required.
            self._sub_input = self._input.subscribe_to_input_events(self._on_input, order=-100000)
            _ = app_window  # keep get_default_app_window side effect / future use
        except Exception as e:
            # Fallback: try without app_window binding
            try:
                self._input = carb.input.acquire_input_interface()
                self._sub_input = self._input.subscribe_to_input_events(self._on_input, order=-100000)
            except Exception as e2:
                self.user_message = "Input subscribe failed: " + repr(e2)

    def _on_input(self, event, *args):
        try:
            et = event.type
            # carb.input.KeyboardEventType
            if et == carb.input.KeyboardEventType.KEY_PRESS or et == carb.input.KeyboardEventType.KEY_REPEAT:
                key = event.input.name if hasattr(event.input, "name") else str(event.input)
                self.held.add(key)
                self.last_key = key
                self.keyboard_events += 1
                if key == "ESCAPE":
                    self.user_stop()
                    return True
                return bool(self.enabled and key in (keyboard_input.KEYS | {"LEFT_SHIFT", "RIGHT_SHIFT", "ESCAPE"}))
            if et == carb.input.KeyboardEventType.KEY_RELEASE:
                key = event.input.name if hasattr(event.input, "name") else str(event.input)
                self.held.discard(key)
                return bool(self.enabled)
        except Exception:
            return False
        return False

    def _hook_contacts(self):
        try:
            self._sub_contact = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(
                lambda h, d: self._on_contacts(h, d)
            )
        except Exception as e:
            self.user_message = "contact sub failed: " + repr(e)

    def _on_contacts(self, headers, data):
        robot = self.config["paths"]["robot"]
        fl = fr = 0.0
        axial = 0.0
        try:
            for h in headers:
                a = str(PhysicsSchemaTools.intToSdfPath(h.collider0))
                b = str(PhysicsSchemaTools.intToSdfPath(h.collider1))
                # impulse estimate
                imp = float(getattr(h, "impulse", 0.0) or 0.0) if hasattr(h, "impulse") else 0.0
                # Some APIs expose contact data separately; keep conservative
                fn = abs(imp) / max(self.dt, 1e-4)
                pair = a + "|" + b
                if "panda_leftfinger" in pair or "finger_joint1" in pair:
                    fl = max(fl, fn)
                if "panda_rightfinger" in pair or "finger_joint2" in pair:
                    fr = max(fr, fn)
                if "Screw_" in pair or "NutPlate" in pair:
                    axial = max(axial, fn)
                if "ring" in pair.lower() or "10347300" in pair or "_0347300" in pair:
                    axial = max(axial, fn)
        except Exception:
            pass
        self.finger_left_N = fl
        self.finger_right_N = fr
        self.contact_force_N = max(fl, fr, axial)
        self.metrics["finger_left_N"] = fl
        self.metrics["finger_right_N"] = fr
        self.metrics["grip_force_N"] = max(fl, fr)
        self.metrics["screw_axial_N"] = axial
        self.metrics["ring_axial_N"] = axial  # coarse until dedicated sensors

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
            # Slow speeds on contact
            ctrl = self.config["control"]
            if self.contact_force_N > float(ctrl.get("contact_force_gate_N", 0.05)):
                # temporarily poke speeds via cfg used by teleop
                self.config["control"]["translation_speed_m_s"] = ctrl.get("contact_slow_translation_m_s", 0.001)
                self.config["control"]["rotation_speed_rad_s"] = ctrl.get("contact_slow_rotation_rad_s", 0.0349)
            else:
                self.config["control"]["translation_speed_m_s"] = 0.01
                self.config["control"]["rotation_speed_rad_s"] = 0.174532925

            if self.enabled and self.controller.armed:
                # motion from held keys
                lin, ang = keyboard_input.vectors(self.held)
                # integrate jog goals
                for axis, sign in enumerate(lin):
                    if abs(sign) > 0:
                        try:
                            self.controller.jog(axis, sign, rotate=False)
                            self.commands += 1
                        except Exception as ex:
                            self.accept_reject = "reject_lin:" + repr(ex)
                            self.controller.stop(str(ex))
                            self.enabled = False
                for axis, sign in enumerate(ang):
                    if abs(sign) > 0:
                        try:
                            self.controller.jog(axis, sign, rotate=True)
                            self.commands += 1
                        except Exception as ex:
                            self.accept_reject = "reject_ang:" + repr(ex)
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
            if self.safety.get("stop") and self.controller.armed:
                self.controller.stop("SAFETY: " + ",".join(self.safety.get("latched_reasons") or []))
                self.enabled = False
                self.timeline.pause()
                self.user_message = "SAFETY STOP"

            # color alarms
            try:
                plan_d_runtime.apply_task2_alarms(self.stage, self.metrics, self.config)
            except Exception:
                pass

            if int(self.t * 10) % 3 == 0:
                self._refresh_status(gate)
                self._log_event(gate)
        except Exception:
            traceback.print_exc()

    def _refresh_status(self, gate=None):
        gate = gate or self.fsm.last_gate
        stop = self.safety.get("stop")
        self.status_label.text = (
            ("STOP: " + ",".join(self.safety.get("latched_reasons") or []) if stop else "MONITORING")
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
            "reasons": self.safety.get("latched_reasons"),
        }
        self.event_log.write(json.dumps(rec) + "\n")

    def close(self):
        self.enabled = False
        try:
            if self.controller.ready:
                self.controller.stop("CLOSE")
        except Exception:
            pass
        self._sub_update = None
        self._sub_input = None
        self._sub_contact = None
        try:
            self.event_log.close()
        except Exception:
            pass


if "_task2_keyboard_panel" in globals():
    try:
        _task2_keyboard_panel.close()
    except Exception:
        pass
_task2_keyboard_panel = Task2KeyboardPanel()
print("[task2] keyboard panel ready:", PACKAGE)
