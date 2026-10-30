# Camera split viewports (2026-10-23) — 20261023_213916

Three small Kit viewport windows stacked on the right:

- Cam · Wrist L
- Cam · Wrist R
- Cam · External

Opened automatically when `keyboard_panel.py` loads (after session cameras are created).
Size/position: `safety_config.json` → `cameras.split_ui`.

Reload panel to refresh. Closing a mini-window is fine; reload recreates them.
