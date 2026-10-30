"""Task2 Script Editor entry — force-reload scene package modules then start panel."""
from pathlib import Path
import sys, gc, shutil, types

PACKAGE = Path("/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene")
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

_HOLDER = "_task2_panel_holder"

def _close_panel(panel):
    if panel is None:
        return
    try:
        panel.close()
    except Exception:
        pass

# 1) Close panel tracked by holder (survives prior reloads).
_holder = sys.modules.get(_HOLDER)
if _holder is not None:
    _close_panel(getattr(_holder, "panel", None))
    try:
        _holder.panel = None
    except Exception:
        pass

# 2) Sweep zombie Task2KeyboardPanel instances from BEFORE the holder existed
#    (their update subscriptions keep calling old GetPrimAtPath(str) code).
for obj in gc.get_objects():
    try:
        if obj.__class__.__name__ == "Task2KeyboardPanel":
            _close_panel(obj)
    except Exception:
        pass

# Drop cached copies so disk edits take effect. Keep holder module.
_drop_prefixes = (
    "keyboard_panel", "franka_teleop", "keyboard_input", "safety_core",
    "plan_d_runtime", "step_fsm", "cameras_setup", "camera_viewports",
    "baseline_logger", "grasp_hold_diag", "breakable_neck",
)
for name in list(sys.modules):
    if name == _HOLDER:
        continue
    if name in _drop_prefixes or name.startswith(_drop_prefixes):
        sys.modules.pop(name, None)

_pyc = PACKAGE / "__pycache__"
if _pyc.is_dir():
    try:
        shutil.rmtree(_pyc)
    except Exception:
        pass

_kp = PACKAGE / "keyboard_panel.py"
if not _kp.is_file():
    raise FileNotFoundError("missing keyboard_panel at %s" % _kp)

# Make sure scene/ is importable (breakable_neck lives here).
_bn = PACKAGE / "breakable_neck.py"
if not _bn.is_file():
    raise FileNotFoundError("missing breakable_neck at %s — FETCH USD/disk changes if Nucleus dialog shows" % _bn)
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))
# Drop stale/failed partial import so reload is clean
sys.modules.pop("breakable_neck", None)
print("[launch_panel] scene path OK;", PACKAGE, "| breakable_neck.py present")

exec(
    compile(_kp.read_text(encoding="utf-8"), str(_kp), "exec"),
    {"__name__": "__main__", "__file__": str(_kp)},
)

print('WIRE_JITTER_FIX_20261025 FixedJoint realign+damp on disk')
print('TELEOP_KINEMATIC_FIX_20261025 skip set*Velocity on kinematic screws')
