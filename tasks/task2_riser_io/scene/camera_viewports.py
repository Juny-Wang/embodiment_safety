"""Small side viewports for WristCam_L / WristCam_R / ExternalOverview."""
from __future__ import annotations


def _as_path(path):
    from pxr import Sdf
    if path is None:
        return None
    if isinstance(path, str):
        return Sdf.Path(path)
    return path


def _get_prim(stage, path):
    """Isaac 5 Stage.GetPrimAtPath requires Sdf.Path, not bare str."""
    p = _as_path(path)
    if p is None or stage is None:
        return None
    return stage.GetPrimAtPath(p)




from typing import Any, Dict, List, Optional, Tuple

_WINDOWS: Dict[str, Any] = {}


def _destroy_named(title: str) -> None:
    w = _WINDOWS.pop(title, None)
    if w is not None:
        try:
            w.destroy()
        except Exception:
            pass
    try:
        from omni.kit.viewport.utility import get_viewport_window_instances
        for inst in list(get_viewport_window_instances() or []):
            name = getattr(inst, "title", None) or getattr(inst, "name", None)
            if name == title:
                try:
                    inst.destroy()
                except Exception:
                    pass
    except Exception:
        pass
    try:
        import omni.ui as ui
        existing = ui.Workspace.get_window(title)
        if existing is not None:
            try:
                existing.visible = False
                existing.destroy()
            except Exception:
                pass
    except Exception:
        pass


def _set_camera(window, camera_path: str) -> None:
    api = getattr(window, "viewport_api", None) or getattr(window, "viewport", None)
    if api is None:
        return
    try:
        from pxr import Sdf
        path = Sdf.Path(str(camera_path))
    except Exception:
        path = camera_path
    for meth in ("set_active_camera", "set_camera_path"):
        fn = getattr(api, meth, None)
        if callable(fn):
            try:
                fn(path)
                return
            except Exception:
                try:
                    fn(str(camera_path))
                    return
                except Exception:
                    pass
    try:
        api.camera_path = path
    except Exception:
        pass


def _place(window, pos_x: int, pos_y: int, width: int, height: int) -> None:
    try:
        if hasattr(window, "set_position"):
            window.set_position(pos_x, pos_y)
        if hasattr(window, "set_size"):
            window.set_size(width, height)
    except Exception:
        pass
    for name, val in (("position_x", pos_x), ("position_y", pos_y), ("width", width), ("height", height)):
        try:
            if hasattr(window, name):
                setattr(window, name, val)
        except Exception:
            pass
    try:
        window.visible = True
        if hasattr(window, "focus"):
            window.focus()
    except Exception:
        pass



# NVIDIA forum bits for /persistent/app/viewport/displayOptions
_DISP_FPS = 1 << 0
_DISP_RESOLUTION = 1 << 3
_DISP_PROGRESS = 1 << 11
_DISP_DEV_MEM = 1 << 13
_DISP_HOST_MEM = 1 << 14
_HUD_BITS = _DISP_FPS | _DISP_RESOLUTION | _DISP_PROGRESS | _DISP_DEV_MEM | _DISP_HOST_MEM


def _clear_display_options_hud() -> Dict[str, Any]:
    """Clear FPS/res/mem HUD bits on the Kit displayOptions bitfield (the real control)."""
    info: Dict[str, Any] = {}
    try:
        import carb.settings
        s = carb.settings.get_settings()
    except Exception as e:
        return {"ok": False, "error": repr(e)}
    for key in (
        "/persistent/app/viewport/displayOptions",
        "/app/viewport/displayOptions",
    ):
        try:
            # get_as_int is required; s.get() often returns a wrapper / None
            getter = getattr(s, "get_as_int", None)
            if callable(getter):
                cur = int(getter(key))
            else:
                raw = s.get(key)
                cur = int(raw) if raw is not None else 0
            nxt = int(cur) & ~_HUD_BITS
            s.set(key, nxt)
            info[key] = {"before": cur, "after": nxt}
        except Exception as e:
            info[key] = {"error": repr(e)}
    # Extra bool flags some Kit builds still honor
    for key, val in (
        ("/app/viewport/show/fps", False),
        ("/app/viewport/show/gpuMemory", False),
        ("/app/viewport/show/hostMemory", False),
        ("/app/viewport/show/resolution", False),
        ("/app/viewport/show/progress", False),
        ("/persistent/app/viewport/show/fps", False),
        ("/persistent/app/viewport/show/gpuMemory", False),
        ("/persistent/app/viewport/show/hostMemory", False),
        ("/persistent/app/viewport/show/resolution", False),
        ("/exts/omni.kit.viewport.window/hud/visible", False),
        ("/exts/omni.kit.viewport.window/hud/fps/visible", False),
        ("/exts/omni.kit.viewport.window/hud/resolution/visible", False),
        ("/exts/omni.kit.viewport.window/hud/progress/visible", False),
    ):
        try:
            s.set(key, val)
        except Exception:
            pass
    info["ok"] = True
    return info


def _force_hud_off_once() -> Dict[str, Any]:
    """Disable Kit viewport performance HUD globally (mini cams inherit this)."""
    return _clear_display_options_hud()


def _declutter_viewport(window, width: int, height: int) -> None:
    """Hide FPS/mem/resolution HUD and match render size to the small window."""
    _clear_display_options_hud()
    api = getattr(window, "viewport_api", None) or getattr(window, "viewport", None)
    if api is not None:
        for name, val in (
            ("fill_frame", True),
            ("resolution", (int(width), int(height))),
        ):
            try:
                setattr(api, name, val)
            except Exception:
                pass
        for meth, args in (
            ("set_texture_resolution", ((int(width), int(height)),)),
            ("set_hud_visible", (False,)),
            ("set_fill_frame", (True,)),
        ):
            fn = getattr(api, meth, None)
            if callable(fn):
                try:
                    fn(*args)
                except Exception:
                    pass
        # Per-viewport display options when API exposes them
        for meth in ("set_display_options", "set_displayOptions"):
            fn = getattr(api, meth, None)
            if callable(fn):
                try:
                    cur = getattr(api, "display_options", None)
                    if cur is None:
                        cur = getattr(api, "displayOptions", None)
                    if isinstance(cur, int):
                        fn(int(cur) & ~_HUD_BITS)
                    else:
                        # try read via settings already cleared; skip
                        pass
                except Exception:
                    pass
        for attr in ("display_options", "displayOptions"):
            if hasattr(api, attr):
                try:
                    cur = getattr(api, attr)
                    if isinstance(cur, int):
                        setattr(api, attr, int(cur) & ~_HUD_BITS)
                except Exception:
                    pass
    try:
        for name in ("show_fps", "show_hud", "hud_visible"):
            if hasattr(window, name):
                try:
                    setattr(window, name, False)
                except Exception:
                    pass
    except Exception:
        pass


def _schedule_hud_rehide(frames: int = 5) -> None:
    """Re-clear HUD after Kit finishes creating the new viewport (it often resets flags)."""
    global _HUD_REHIDE_SUB
    try:
        import omni.kit.app
        app = omni.kit.app.get_app()
        state = {"left": int(max(1, frames))}
        sub_holder: Dict[str, Any] = {"sub": None}

        def _on_update(_e):
            try:
                _clear_display_options_hud()
            except Exception:
                pass
            state["left"] -= 1
            if state["left"] <= 0:
                sub = sub_holder.get("sub")
                sub_holder["sub"] = None
                if sub is not None:
                    try:
                        sub.unsubscribe()
                    except Exception:
                        try:
                            del sub
                        except Exception:
                            pass

        sub_holder["sub"] = app.get_update_event_stream().create_subscription_to_pop(_on_update)
        _HUD_REHIDE_SUB = sub_holder
    except Exception:
        pass


_HUD_REHIDE_SUB = None

def _create_one(title: str, camera_path: str, width: int, height: int, pos_x: int, pos_y: int):
    _destroy_named(title)
    errors = []
    try:
        from omni.kit.viewport.utility import create_viewport_window
        from pxr import Sdf
        win = create_viewport_window(
            name=title, width=int(width), height=int(height),
            position_x=int(pos_x), position_y=int(pos_y),
            camera_path=Sdf.Path(str(camera_path)),
        )
        if win is not None:
            _WINDOWS[title] = win
            _set_camera(win, camera_path)
            _place(win, pos_x, pos_y, width, height)
            _declutter_viewport(win, width, height)
            return win, "kit_utility"
    except Exception as e:
        errors.append(("kit_utility", repr(e)))
    try:
        from omni.kit.viewport.utility import create_viewport_window
        win = create_viewport_window(name=title, width=int(width), height=int(height))
        if win is not None:
            _WINDOWS[title] = win
            _set_camera(win, camera_path)
            _place(win, pos_x, pos_y, width, height)
            _declutter_viewport(win, width, height)
            return win, "kit_utility_setcam"
    except Exception as e:
        errors.append(("kit_utility_setcam", repr(e)))
    try:
        from omni.kit.viewport.utility import create_viewport_for_camera
        win = create_viewport_for_camera(camera_path)
        if win is not None:
            try:
                win.title = title
            except Exception:
                pass
            _WINDOWS[title] = win
            _place(win, pos_x, pos_y, width, height)
            _declutter_viewport(win, width, height)
            return win, "create_viewport_for_camera"
    except Exception as e:
        errors.append(("create_viewport_for_camera", repr(e)))
    try:
        from isaacsim.core.rendering_manager import ViewportManager
        win = ViewportManager.create_viewport_window(
            camera=str(camera_path), title=title, resolution=(int(width), int(height)),
        )
        if win is not None:
            _WINDOWS[title] = win
            _set_camera(win, camera_path)
            _place(win, pos_x, pos_y, width, height)
            _declutter_viewport(win, width, height)
            return win, "viewport_manager"
    except Exception as e:
        errors.append(("viewport_manager", repr(e)))
    raise RuntimeError("create viewport failed: " + "; ".join("%s=%s" % p for p in errors))


def resolve_camera_paths(stage, cameras_cfg: Dict[str, Any], cameras_report: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    if cameras_report and isinstance(cameras_report.get("paths"), dict):
        for k, v in cameras_report["paths"].items():
            if v:
                paths[k] = str(v)
    want = {
        "wrist_left": (cameras_cfg.get("wrist_left") or {}).get("name", "WristCam_L"),
        "wrist_right": (cameras_cfg.get("wrist_right") or {}).get("name", "WristCam_R"),
        "external": (cameras_cfg.get("external") or {}).get("name", "ExternalOverview"),
    }
    found_by_name: Dict[str, str] = {}
    if stage is not None:
        for prim in stage.Traverse():
            try:
                if prim.GetTypeName() != "Camera":
                    continue
            except Exception:
                continue
            n = prim.GetName()
            if n in want.values():
                found_by_name[n] = prim.GetPath().pathString
    role_to_name = {v: k for k, v in want.items()}
    for name, path in found_by_name.items():
        role = role_to_name.get(name)
        if role and role not in paths:
            paths[role] = path
    if "external" not in paths:
        paths["external"] = f"/World/Cameras/{want['external']}"
    return paths


def open_camera_split(
    stage,
    safety: Optional[Dict[str, Any]] = None,
    cameras_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    safety = safety or {}
    cfg = safety.get("cameras") or {}
    ui_cfg = cfg.get("split_ui") or {}
    if not bool(ui_cfg.get("enabled", True)):
        return {"ok": False, "skipped": True, "reason": "cameras.split_ui.enabled=false"}

    width = int(ui_cfg.get("width", 360))
    height = int(ui_cfg.get("height", 270))
    margin_x = int(ui_cfg.get("margin_x", 24))
    margin_y = int(ui_cfg.get("margin_y", 72))
    gap = int(ui_cfg.get("gap", 10))
    try:
        import omni.ui as ui
        getter = getattr(ui.Workspace, "get_main_window_width", None)
        app_w = int(getter() if callable(getter) else 1920) or 1920
    except Exception:
        app_w = 1920
    pos_x = int(ui_cfg.get("position_x", max(40, app_w - width - margin_x)))
    pos_y0 = int(ui_cfg.get("position_y", margin_y))

    hide_hud = bool(ui_cfg.get("hide_hud", True))
    hud_info = None
    if hide_hud:
        hud_info = _force_hud_off_once()

    paths = resolve_camera_paths(stage, cfg, cameras_report)
    order: List[Tuple[str, str, Optional[str]]] = [
        ("wrist_left", "Wrist L", paths.get("wrist_left")),
        ("wrist_right", "Wrist R", paths.get("wrist_right")),
        ("external", "External", paths.get("external")),
    ]
    report: Dict[str, Any] = {"windows": [], "errors": [], "paths": paths}
    for role, _, cam_path in order:
        if not cam_path:
            report["errors"].append("missing camera path for " + role)
            continue
        if stage is not None:
            prim = _get_prim(stage, cam_path)
            if not prim or not prim.IsValid():
                report["errors"].append("camera prim missing: " + cam_path)
    for i, (role, title, cam_path) in enumerate(order):
        if not cam_path:
            continue
        win_title = str((ui_cfg.get("titles") or {}).get(role, f"Cam · {title}"))
        pos_y = pos_y0 + i * (height + gap)
        try:
            win, how = _create_one(win_title, cam_path, width, height, pos_x, pos_y)
            # FIX3_REBIND: force camera path after window creation (USDA rel can race).
            try:
                _set_camera(win, cam_path)
            except Exception as _rb:
                report.setdefault("rebind_errors", []).append(repr(_rb))
            report["windows"].append({
                "title": win_title, "camera": cam_path, "via": how,
                "pos": [pos_x, pos_y], "size": [width, height],
            })
        except Exception as e:
            report["errors"].append(f"{win_title}: {e!r}")
    if hide_hud:
        report["hud"] = hud_info
        # Kit often re-enables HUD when new ViewportWindow spawns; re-clear for a few frames.
        _schedule_hud_rehide(5)
        _force_hud_off_once()
    report["ok"] = len(report["windows"]) == 3 and not report["errors"]
    report["partial_ok"] = len(report["windows"]) > 0
    return report


def close_camera_split() -> None:
    for title in list(_WINDOWS.keys()):
        _destroy_named(title)
