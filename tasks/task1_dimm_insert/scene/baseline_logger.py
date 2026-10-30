"""Item4 baseline CSV logger for franka_manual_safety_v002.

Enable (any one):
  - env BASELINE_LOG=1  (before launching keyboard_panel)
  - env BASELINE_LOG_PATH=/path/to/run.csv  (implies enable; custom path)
  - after panel up:  panel.enable_baseline_log()  or  baseline_logger.attach(panel, enable=True)

Disable: BASELINE_LOG=0 / unset, or panel.disable_baseline_log().
When disabled, every call is a no-op — teleop must not break.

Isaac Script Editor one-liner (optional, after panel exists):
  exec(open("/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_manual_safety_v002/baseline_logger.py").read())
  attach(_franka_keyboard, enable=True)   # if panel bound as _franka_keyboard
"""
from __future__ import annotations

import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

# Prefer scene package dir when present; fall back to this file's directory.
_HERE = Path(__file__).resolve().parent
_DEFAULT_PKG = Path(
    "/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_manual_safety_v002"
)
LOG_DIR_CANDIDATES = [
    _DEFAULT_PKG / "baseline_logs",
    _HERE / "baseline_logs",
]

CSV_FIELDS: List[str] = [
    "wall_time_iso_asia_shanghai",
    "unix_s",
    "sim_t",
    "dt",
    "run_id",
    "trial_tag",
    "step_index",
    "step_name",
    "socket_frame_used",
    "align_xy_m",
    "align_z_m",
    "tilt_deg",
    "tilt_t1_deg",
    "tilt_t2_deg",
    "tilt_deg_primary",
    "end_difference_mm",
    "seated_depth_m",
    "seated_ok",
    "insertion_clearance_ratio",
    "insertion_clearance_ok",
    "insertion_clearance_color",
    "insertion_clearance_worst",
    "insertion_clearance_reason",
    "insertion_clearance_xy_ratio",
    "insertion_clearance_tilt_ratio",
    "insertion_clearance_end_ratio",
    "left_press_N",
    "right_press_N",
    "axial_contact_N",
    "lateral_contact_N",
    "dimm_contact_N",
    "finger_left_N",
    "finger_right_N",
    "unexpected_robot_contact_N",
    "left_angle_abs_deg",
    "right_angle_abs_deg",
    "block_down",
    "block_release",
    "block_close",
    "aligned_gate",
    "dual_latched_gate",
    "gate_reasons",
    "fsm_halt",
    "traffic_color",
]


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _pick_log_dir() -> Path:
    custom = os.environ.get("BASELINE_LOG_DIR", "").strip()
    if custom:
        return Path(custom)
    for d in LOG_DIR_CANDIDATES:
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_probe"
            probe.write_text("ok")
            probe.unlink(missing_ok=True)
            return d
        except Exception:
            continue
    # last resort: cwd
    d = Path.cwd() / "baseline_logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


class BaselineLogger:
    """Append-only CSV logger. Safe no-op when disabled."""

    def __init__(
        self,
        enable: Optional[bool] = None,
        path: Optional[Path] = None,
        run_id: Optional[str] = None,
        trial_tag: str = "",
    ):
        if enable is None:
            enable = _env_truthy("BASELINE_LOG") or bool(
                os.environ.get("BASELINE_LOG_PATH", "").strip()
            )
        self.enabled = bool(enable)
        self.trial_tag = trial_tag or os.environ.get("BASELINE_TRIAL_TAG", "")
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._fh: Optional[TextIO] = None
        self._writer: Optional[csv.DictWriter] = None
        self.path: Optional[Path] = None
        self.rows_written = 0
        self._last_error: Optional[str] = None
        if not self.enabled:
            return
        try:
            env_path = os.environ.get("BASELINE_LOG_PATH", "").strip()
            if path is not None:
                self.path = Path(path)
            elif env_path:
                self.path = Path(env_path)
            else:
                self.path = _pick_log_dir() / f"baseline_{self.run_id}.csv"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not self.path.exists() or self.path.stat().st_size == 0
            self._fh = open(self.path, "a", newline="", buffering=1, encoding="utf-8")
            self._writer = csv.DictWriter(
                self._fh, fieldnames=CSV_FIELDS, extrasaction="ignore"
            )
            if new_file:
                self._writer.writeheader()
                self._fh.flush()
        except Exception as e:
            self.enabled = False
            self._last_error = repr(e)
            self._fh = None
            self._writer = None

    def set_trial_tag(self, tag: str) -> None:
        self.trial_tag = str(tag or "")

    def close(self) -> None:
        try:
            if self._fh:
                self._fh.flush()
                self._fh.close()
        except Exception:
            pass
        self._fh = None
        self._writer = None

    def log(self, panel: Any, dt: float = 0.0) -> None:
        if not self.enabled or self._writer is None:
            return
        try:
            metrics = getattr(panel, "metrics", None) or {}
            gate = getattr(panel, "_last_gate", None) or {}
            fsm = getattr(panel, "fsm", None)
            if not gate and fsm is not None:
                gate = getattr(fsm, "last_gate", None) or {}
            light = getattr(fsm, "light", None) if fsm is not None else None
            traffic = None
            if light is not None:
                traffic = getattr(light, "color", None) or getattr(light, "state", None)

            def g(key: str, default=None):
                return metrics.get(key, default)

            row: Dict[str, Any] = {
                "wall_time_iso_asia_shanghai": datetime.now().isoformat(timespec="milliseconds"),
                "unix_s": f"{time.time():.6f}",
                "sim_t": getattr(panel, "t", None),
                "dt": dt,
                "run_id": self.run_id,
                "trial_tag": self.trial_tag,
                "step_index": getattr(panel, "step_index", None),
                "step_name": gate.get("step")
                or (getattr(fsm, "step_name", None) if fsm else None),
                "socket_frame_used": g("socket_frame_used"),
                "align_xy_m": g("align_xy_m"),
                "align_z_m": g("align_z_m"),
                "tilt_deg": g("tilt_deg"),
                "tilt_t1_deg": g("tilt_t1_deg"),
                "tilt_t2_deg": g("tilt_t2_deg"),
                "tilt_deg_primary": g("tilt_deg_primary"),
                "end_difference_mm": g("end_difference_mm"),
                "seated_depth_m": g("seated_depth_m"),
                "seated_ok": g("seated_ok"),
                "insertion_clearance_ratio": g("insertion_clearance_ratio"),
                "insertion_clearance_ok": g("insertion_clearance_ok"),
                "insertion_clearance_color": g("insertion_clearance_color"),
                "insertion_clearance_worst": g("insertion_clearance_worst"),
                "insertion_clearance_reason": g("insertion_clearance_reason"),
                "insertion_clearance_xy_ratio": g("insertion_clearance_xy_ratio"),
                "insertion_clearance_tilt_ratio": g("insertion_clearance_tilt_ratio"),
                "insertion_clearance_end_ratio": g("insertion_clearance_end_ratio"),
                "left_press_N": g("left_press_N"),
                "right_press_N": g("right_press_N"),
                "axial_contact_N": g("axial_contact_N"),
                "lateral_contact_N": g("lateral_contact_N"),
                "dimm_contact_N": g("dimm_contact_N"),
                "finger_left_N": g("finger_left_N"),
                "finger_right_N": g("finger_right_N"),
                "unexpected_robot_contact_N": g("unexpected_robot_contact_N"),
                "left_angle_abs_deg": g("left_angle_abs_deg"),
                "right_angle_abs_deg": g("right_angle_abs_deg"),
                "block_down": gate.get("block_down"),
                "block_release": gate.get("block_release"),
                "block_close": gate.get("block_close"),
                "aligned_gate": gate.get("aligned"),
                "dual_latched_gate": gate.get("dual_latched"),
                "gate_reasons": "|".join(gate.get("reasons") or []),
                "fsm_halt": getattr(fsm, "halt", None) if fsm else None,
                "traffic_color": traffic,
            }
            self._writer.writerow(row)
            self.rows_written += 1
            if self.rows_written % 120 == 0 and self._fh:
                self._fh.flush()
        except Exception as e:
            # Never raise into physics_step
            self._last_error = repr(e)


_GLOBAL: Optional[BaselineLogger] = None


def get_logger(create_if_env: bool = True) -> Optional[BaselineLogger]:
    global _GLOBAL
    if _GLOBAL is not None:
        return _GLOBAL
    if create_if_env and (
        _env_truthy("BASELINE_LOG") or os.environ.get("BASELINE_LOG_PATH", "").strip()
    ):
        _GLOBAL = BaselineLogger(enable=True)
        return _GLOBAL
    return None


def attach(panel: Any, enable: bool = True, trial_tag: str = "") -> BaselineLogger:
    """Attach logger to a live KeyboardPanel instance."""
    global _GLOBAL
    lg = BaselineLogger(enable=enable, trial_tag=trial_tag)
    _GLOBAL = lg
    try:
        panel._baseline_logger = lg
        # convenience methods on panel
        panel.enable_baseline_log = lambda tag="": _enable_on_panel(panel, tag)
        panel.disable_baseline_log = lambda: _disable_on_panel(panel)
        panel.set_baseline_trial_tag = lambda tag: setattr(lg, "trial_tag", str(tag or ""))
    except Exception:
        pass
    return lg


def _enable_on_panel(panel: Any, trial_tag: str = "") -> str:
    lg = BaselineLogger(enable=True, trial_tag=trial_tag)
    panel._baseline_logger = lg
    global _GLOBAL
    _GLOBAL = lg
    return f"baseline_log ON path={lg.path} run_id={lg.run_id}"


def _disable_on_panel(panel: Any) -> str:
    lg = getattr(panel, "_baseline_logger", None)
    if lg is not None:
        try:
            lg.close()
        except Exception:
            pass
        lg.enabled = False
    panel._baseline_logger = None
    return "baseline_log OFF"


def maybe_log(panel: Any, dt: float = 0.0) -> None:
    """Called from physics_step; no-op if disabled."""
    lg = getattr(panel, "_baseline_logger", None)
    if lg is None:
        lg = get_logger(create_if_env=True)
        if lg is not None:
            try:
                panel._baseline_logger = lg
            except Exception:
                pass
    if lg is not None:
        lg.log(panel, dt)


# Script-Editor friendly aliases when exec'd
def enable(panel=None, trial_tag: str = ""):
    if panel is None:
        g = globals()
        panel = g.get("_franka_keyboard") or g.get("panel")
    if panel is None:
        return "pass panel=... or bind _franka_keyboard"
    return attach(panel, enable=True, trial_tag=trial_tag).path


if __name__ == "__main__":
    # Smoke: write one fake row when BASELINE_LOG=1
    os.environ.setdefault("BASELINE_LOG", "1")
    class _P:
        metrics = {
            "align_xy_m": 0.001,
            "tilt_deg": 0.5,
            "insertion_clearance_ok": True,
            "socket_frame_used": "/World/MatchedSocket/Asset",
        }
        t = 0.0
        step_index = 3
        _last_gate = {"step": "ALIGN", "block_down": False, "reasons": []}
        fsm = None
    lg = BaselineLogger(enable=True)
    lg.log(_P(), 1.0 / 240.0)
    lg.close()
    print("SMOKE_OK", lg.path, "rows", lg.rows_written, "err", lg._last_error)
