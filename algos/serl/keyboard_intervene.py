"""Carb keyboard → 7-D Task1 diff_ik action in [-1, 1].

Key map mirrors tasks/task1_dimm_insert/scene/keyboard_input.py:
  arrows / PgUp / PgDn  → translation (or rotation with Shift)
  [ / O open, ] / P close → gripper
  R → request episode reset (caller handles)
  Enter → mark attempt done early (optional; usually env success/timeout)

Import only after Kit / AppLauncher is up.
"""

from __future__ import annotations

from typing import Set

import numpy as np

# axis index, sign  (same as scene/keyboard_input.TRANSLATION)
_TRANSLATION = {
    "UP": (0, 1.0),
    "DOWN": (0, -1.0),
    "LEFT": (1, 1.0),
    "RIGHT": (1, -1.0),
    "PAGE_UP": (2, 1.0),
    "PAGE_DOWN": (2, -1.0),
}
_OPEN = {"LEFT_BRACKET", "O"}
_CLOSE = {"RIGHT_BRACKET", "P"}
_SHIFT = {"LEFT_SHIFT", "RIGHT_SHIFT"}
_RESET = {"R"}
_SKIP = {"ENTER", "NUMPAD_ENTER"}


def _norm_key(name: str) -> str:
    return str(name).upper().replace(" ", "_")


class KeyboardIntervene:
    """Hold-to-move keyboard; produces a fresh 7-D action each poll."""

    def __init__(self, move_scale: float = 1.0, rot_scale: float = 1.0):
        self.move_scale = float(move_scale)
        self.rot_scale = float(rot_scale)
        self.held: Set[str] = set()
        self.reset_requested = False
        self.skip_requested = False
        self._sub = None
        self._input = None
        self.enabled = False
        self.last_key = "none"
        self.key_count = 0

    def enable(self) -> None:
        import carb.input
        import omni.appwindow

        if self.enabled:
            return
        self._input = carb.input.acquire_input_interface()
        appwindow = omni.appwindow.get_default_app_window()
        keyboard = appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            keyboard, lambda event, *args: self._on_event(event)
        )
        self.enabled = True
        print("[keyboard_intervene] ENABLED arrows/PgUp/Dn move; Shift+rotate; [/] gripper; R reset", flush=True)

    def disable(self) -> None:
        if self._sub is not None and self._input is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(self._sub)
            except Exception:
                pass
        self._sub = None
        self.enabled = False
        self.held.clear()

    def _on_event(self, event) -> bool:
        import carb.input

        try:
            key = _norm_key(event.input.name)
        except Exception:
            return False
        self.last_key = key
        self.key_count += 1
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.held.add(key)
            if key in _RESET:
                self.reset_requested = True
            if key in _SKIP:
                self.skip_requested = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.held.discard(key)
        return True

    def action(self) -> np.ndarray:
        """Return action[7] in [-1,1]: dx,dy,dz,droll,dpitch,dyaw,gripper."""
        a = np.zeros(7, dtype=np.float32)
        rotate = bool(self.held & _SHIFT)
        for key in self.held:
            if key not in _TRANSLATION:
                continue
            axis, sign = _TRANSLATION[key]
            if rotate:
                a[3 + axis] += sign * self.rot_scale
            else:
                a[axis] += sign * self.move_scale
        # normalize translation / rotation blocks separately
        t = a[0:3]
        n = float(np.linalg.norm(t))
        if n > 1.0:
            a[0:3] = t / n
        r = a[3:6]
        n = float(np.linalg.norm(r))
        if n > 1.0:
            a[3:6] = r / n
        open_h = bool(self.held & _OPEN)
        close_h = bool(self.held & _CLOSE)
        if open_h and not close_h:
            # env: action=-1 → open fingers
            a[6] = -1.0
        elif close_h and not open_h:
            a[6] = 1.0
        else:
            a[6] = 0.0
        return np.clip(a, -1.0, 1.0)

    def consume_reset(self) -> bool:
        flag = self.reset_requested
        self.reset_requested = False
        return flag

    def consume_skip(self) -> bool:
        flag = self.skip_requested
        self.skip_requested = False
        return flag
