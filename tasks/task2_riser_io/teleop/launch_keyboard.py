"""Convenience launcher path under teleop/."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1] / "scene" / "keyboard_panel.py"), run_name="__main__")
