"""Three live previews of existing capture RGB; no extra cameras/render products."""
def close(panel):
    panel.preview_update_sub=None
    for w in getattr(panel,"preview_windows",[]):
        try:w.destroy()
        except Exception:pass
    panel.preview_windows=[];panel.preview_providers={}

def hide_camera_visuals(stage):
    """Session-only eye visibility; leave active state and optics untouched."""
    from pxr import Usd,UsdGeom
    if stage is None:
        return {"hidden":[],"warning":"No stage open"}
    targets=[]
    for prim in stage.Traverse():
        path=str(prim.GetPath())
        if (path.startswith("/World/Robot/") or path.startswith("/Robot/")) and prim.GetParent().GetName()=="panda_hand" and prim.GetName() in ("WristCam_L","WristCam_R"):
            targets.append(prim)
    root=stage.GetPrimAtPath("/World/Cameras")
    if root:targets.append(root)
    hidden=[]
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        for prim in targets:
            img=UsdGeom.Imageable(prim)
            if img:
                img.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                hidden.append(str(prim.GetPath()))
    return {"hidden":hidden,"session_only":True,"cameras_deactivated":False}

def hide_camera_gizmos():
    """Hide camera drawings and selected camera visuals, not camera sensors."""
    import carb.settings
    settings=carb.settings.get_settings()
    settings.set_bool("/app/viewport/show/camera",False)
    # Installed viewport.actions maps camera drawings to displayOptions bit 5.
    key="/persistent/app/viewport/displayOptions"
    options=settings.get(key)
    if isinstance(options,int):settings.set_int(key,options & ~(1 << 5))
    import omni.usd
    report=hide_camera_visuals(omni.usd.get_context().get_stage())
    report["camera_gizmos_visible"]=settings.get("/app/viewport/show/camera")
    return report

def preview_frame(panel,key):
    """Display may use a t=0 sensor frame; never insert it into recording cache."""
    import numpy as np
    item=getattr(panel,"image_cache",{}).get(key)
    cam=getattr(panel,"cameras",{}).get(key)
    if item is not None:
        return item,""
    if cam is not None:
        frame=cam.get_current_frame()
        rgb=frame.get("rgb")
        if rgb is None:rgb=frame.get("rgba")
        if rgb is not None:
            a=np.asarray(rgb)
            if a.ndim==3 and a.shape[2]>=3 and a.size:
                return {"rgb":a[:,:,:3],"sim_time":float(frame.get("rendering_time",0.)),
                        "frame":"preview-only:"+str(frame.get("rendering_frame",0))},""
    if not panel.timeline.is_playing():
        return None,"PAUSED - click Play; RECORD is not required"
    errors=getattr(panel,"camera_errors",[])
    if errors:return None,"Camera error: "+str(errors[-1])[:130]
    if cam is not None and cam.is_paused():
        return None,"Camera acquisition paused - run restore_previews.py"
    return None,"PLAYING - no RGB received from render product yet"

def show(panel):
    import omni.ui as ui,omni.kit.app
    import numpy as np
    close(panel)
    hide_camera_gizmos()
    cfg=panel.safety.get("cameras",{}).get("split_ui",{})
    width=int(cfg.get("width",360));height=int(cfg.get("height",270))
    try:
        import omni.appwindow
        aw=omni.appwindow.get_default_app_window()
        screen_w=int(aw.get_width());screen_h=int(aw.get_height())
    except Exception:screen_w,screen_h=1920,1080
    if screen_w<100 or screen_h<100:screen_w,screen_h=1920,1080
    y=int(cfg.get("margin_y",72));gap=int(cfg.get("gap",10))
    height=min(height,max(100,(screen_h-y-3*gap-24)//3))
    x=max(0,screen_w-width-int(cfg.get("margin_x",24)))
    frames={};labels={};report={"windows":[],"errors":[],"mode":"existing_capture_rgb"}
    for i,(key,label) in enumerate([("wrist_left","Wrist L"),("wrist_right","Wrist R"),("external","External")]):
        path=panel.camera_setup["paths"][key]
        w=ui.Window("Task1 Cam - "+label,width=width,height=height,
                    position_x=x,position_y=y+i*(height+gap))
        provider=ui.ByteImageProvider()
        with w.frame:
            with ui.VStack():
                labels[key]=ui.Label("Waiting for captured RGB",height=20)
                ui.ImageWithProvider(provider)
        panel.preview_windows.append(w);panel.preview_providers[key]=provider
        report["windows"].append({"title":"Task1 Cam - "+label,"camera":path})
    pending_hide=[3]
    def update(_=None):
        if getattr(panel,"closed",False):close(panel);return
        if pending_hide[0]>0:
            hide_camera_gizmos();pending_hide[0]-=1
        for key,provider in panel.preview_providers.items():
            try:item,message=preview_frame(panel,key)
            except Exception as e:
                labels[key].text="Camera read error: "+repr(e)[:120]
                continue
            if item is None:
                labels[key].text=message
                continue
            if frames.get(key)==item["frame"]:
                labels[key].text=("PAUSED | " if not panel.timeline.is_playing() else "")+"Last RGB | t=%.3f s"%item["sim_time"]
                continue
            rgb=np.asarray(item["rgb"],dtype=np.uint8)
            rgba=np.empty((*rgb.shape[:2],4),dtype=np.uint8);rgba[:,:,:3]=rgb[:,:,:3];rgba[:,:,3]=255
            provider.set_bytes_data(rgba.ravel().tolist(),[rgba.shape[1],rgba.shape[0]])
            labels[key].text="Live preview | t=%.3f s | frame %s"%(item["sim_time"],item["frame"])
            frames[key]=item["frame"]
    panel.preview_update_sub=omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(update)
    update()
    report["ok"]=True;panel.preview_report=report
    return report
