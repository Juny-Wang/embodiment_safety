"""Run in Isaac Script Editor. Reuse a live panel; pause before first load."""
import asyncio
import builtins
import importlib
importlib.invalidate_caches()
import sys
from pathlib import Path

ROOT = Path("/mnt/data2/wjy_SERL/Embodied_Safety")
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def launch_collection():
    import omni.usd
    import omni.timeline
    import omni.kit.app

    context = omni.usd.get_context()
    stage = context.get_stage()
    expected = ROOT / "tasks/task1_dimm_insert/scene/collection.usda"
    if stage is None or Path(stage.GetRootLayer().realPath).resolve() != expected:
        raise RuntimeError("Open " + str(expected) + " first")

    panel = getattr(builtins, "_task1_collection", None)
    if panel is not None and not getattr(panel, "closed", True) and panel.stage == stage:
        if getattr(panel, "ui", None):
            panel.ui.visible = True
        panel.show_previews()
        print("Existing collection panel restored; playback and recording unchanged.")
        return panel

    pending = getattr(builtins, "_task1_collection_launch_task", None)
    if pending is not None and not pending.done():
        print("Collection panel startup already pending.")
        return None

    # Capture legacy globals now: Script Editor may reuse its namespace.
    old_objects = []
    for name in ("_franka_keyboard", "_franka_manual_safety", "_task1_collection"):
        for obj in (globals().get(name), getattr(builtins, name, None)):
            if obj is not None and all(obj is not old for old in old_objects):
                old_objects.append(obj)

    async def start_paused():
        timeline = omni.timeline.get_timeline_interface()
        app = omni.kit.app.get_app()
        if timeline.is_playing():
            timeline.pause()
        # Timeline pause is applied on a Kit update, not necessarily immediately.
        for _ in range(30):
            await app.next_update_async()
            if not timeline.is_playing():
                break
        if timeline.is_playing():
            raise RuntimeError("Pause did not take effect; panel was not replaced.")
        if context.get_stage() != stage:
            raise RuntimeError("Stage changed during startup; please run launcher again.")
        for obj in old_objects:
            if hasattr(obj, "close"):
                obj.close()
        from tasks.task1_dimm_insert.env import actions as _actions_mod
        importlib.reload(_actions_mod)
        from tasks.task1_dimm_insert.teleop import runtime
        importlib.reload(runtime)
        new_panel = runtime.CollectionPanel()
        builtins._task1_collection = new_panel
        print("Collection panel ready, timeline paused. Play -> INITIALIZE -> RECORD (check growing sample count) -> ENABLE.")
        return new_panel

    def report_completion(task):
        try:
            task.result()
        except asyncio.CancelledError:
            print("Collection panel startup cancelled.")
        except Exception:
            import traceback
            traceback.print_exc()

    task = asyncio.ensure_future(start_paused())
    builtins._task1_collection_launch_task = task
    task.add_done_callback(report_completion)
    return None


launch_collection()
