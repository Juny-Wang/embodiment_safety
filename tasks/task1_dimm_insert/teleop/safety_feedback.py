"""Measured safety feedback. UI overlays only: never authors USD or camera RGB."""
import itertools
import time
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation

COLORS = {"GREEN": 0xff60cc60, "YELLOW": 0xff40ccff, "RED": 0xff4040ff,
          "GRAY": 0xffaaaaaa}
LABELS = {"left_press_N":"Left latch", "right_press_N":"Right latch",
          "dimm_environment_N":"DIMM/environment", "finger_left_N":"Left finger/DIMM",
          "finger_right_N":"Right finger/DIMM"}

def inside(point, box):
    return bool(np.all(np.asarray(point)>=box["min"]) and np.all(np.asarray(point)<=box["max"]))

def classify(point, geometry):
    # Explicit forbidden regions take precedence if boxes overlap.
    forbidden=[b["name"] for b in geometry.get("grasp_forbidden_boxes",[]) if inside(point,b)]
    if forbidden:return "FORBIDDEN",forbidden
    allowed=[b["name"] for b in geometry.get("grasp_allowed_boxes",[]) if inside(point,b)]
    return ("ALLOWED",allowed) if allowed else ("UNCLASSIFIED",[])

def severity(value, limit, warning=.8):
    if value is None:return "GRAY"
    return "RED" if value>=limit else ("YELLOW" if value>=warning*limit-1e-12 else "GREEN")

def world_to_local(point, pose, scale):
    return Rotation.from_quat(pose[3:]).inv().apply(np.asarray(point)-pose[:3])/scale

def local_to_world(points, pose, scale):
    return Rotation.from_quat(pose[3:]).apply(np.asarray(points)*scale)+pose[:3]

class Feedback:
    def __init__(self,panel,show_ui=True):
        self.panel=panel;self.window=None;self.scene_view=None;self.viewport=None
        self.history=deque();self.peak={};self.samples=[];self.last_ui=0.
        self.error="";self.episode=None;self.started=None;self.last_path=""
        self.rows={};self.last_metrics={};self.last_measure_t=None
        from pxr import UsdGeom,Gf
        self.scale=np.asarray(Gf.Transform(UsdGeom.XformCache().GetLocalToWorldTransform(
            panel.stage.GetPrimAtPath(panel.body_paths["dimm"]))).GetScale(),dtype=float)
        if np.any(abs(self.scale)<1e-12):raise RuntimeError("Invalid DIMM scale")
        if show_ui:self.build_ui()

    def pose(self):
        p=self.panel
        if p.controller.ready and "dimm" in p.bodies:
            return np.asarray(p.bodies["dimm"].get_transforms()[0],dtype=float)
        from pxr import UsdGeom
        m=UsdGeom.XformCache().GetLocalToWorldTransform(p.stage.GetPrimAtPath(p.body_paths["dimm"]))
        q=m.ExtractRotationQuat().GetNormalized()
        return np.array([*m.ExtractTranslation(),*q.GetImaginary(),q.GetReal()],dtype=float)

    def consume(self, contacts):
        p=self.panel;pose=self.pose()
        gate=float(p.safety.get("contact_gate_N",.05))
        forbidden=[];unknown=[];samples=[]
        for c in contacts:
            a,b=c["a"],c["b"]
            is_dimm=lambda s:s==p.body_paths["dimm"] or s.startswith(p.body_paths["dimm"]+"/")
            finger=lambda s: s.startswith("/World/Robot/") and any(x in s for x in ("panda_leftfinger","panda_rightfinger"))
            if (is_dimm(a) and finger(b)) or (is_dimm(b) and finger(a)):
                point=world_to_local(c["point"],pose,self.scale)
                region,names=classify(point,p.calib["dimm_local"])
                c["grasp_region"]=region;c["region_names"]=names;c["dimm_local_point_m"]=point.tolist()
                if c["normal_force_N"]>=gate:
                    if region=="FORBIDDEN":forbidden.append(c)
                    elif region=="UNCLASSIFIED":unknown.append(c)
            else:region=None
            if c["normal_force_N"]>=gate:
                keys=[]
                both=a+" "+b
                for side in ("left","right"):
                    if side.title()+"Latch" in both and "/World/Robot/" in both:keys.append(side+"_press_N")
                    if "panda_"+side+"finger" in both and (is_dimm(a) or is_dimm(b)):keys.append("finger_"+side+"_N")
                if (is_dimm(a) or is_dimm(b)) and not (finger(a) or finger(b)):keys.append("dimm_environment_N")
                samples.append(dict(point=c["point"],region=region,keys=keys,force=c["normal_force_N"]))
        p.contact_metrics["forbidden_contact"]=float(bool(forbidden))
        p.contact_metrics["unclassified_grasp_contact"]=float(bool(unknown))
        if forbidden:p.violations.add("forbidden_grasp_contact")
        if unknown:p.violations.add("unclassified_grasp_contact")
        for key in ("forbidden_contact","unclassified_grasp_contact"):
            p.interval_peaks[key]=max(p.interval_peaks.get(key,0.),p.contact_metrics[key])
        for key,limit in p.cfg["demonstration_limits"].items():
            if p.contact_metrics.get(key,0.)>=limit:p.violations.add(key)
        if p.active is not self.episode:
            if p.active:
                self.peak={};self.started=p.t;self.last_path=str(p.active.path)
            self.episode=p.active
        self.last_measure_t=p.t
        self.last_metrics=dict(p.contact_metrics)
        for key,value in self.last_metrics.items():self.peak[key]=max(self.peak.get(key,0.),value)
        self.history.append((p.t,dict(self.last_metrics),samples))
        while self.history and p.t-self.history[0][0]>.5:self.history.popleft()
        self.samples=samples

    def recent(self):
        keys=set(self.last_metrics)
        return {k:max((m.get(k,0.) for _,m,_ in self.history),default=0.) for k in keys}

    def build_ui(self):
        import omni.ui as ui
        self.window=ui.Window("Task1 SAFETY - live contact",width=560,height=340)
        with self.window.frame:
            with ui.VStack(spacing=5):
                ui.Label("DEMO LIMITS | warnings only | normal contact force, not wrist F/T",word_wrap=True)
                self.record_label=ui.Label("Not recording | no fixed time limit",word_wrap=True)
                for key,label in LABELS.items():self.rows[key]=ui.Label(label+" : waiting")
                self.grasp_label=ui.Label("Grasp regions: waiting",word_wrap=True)
                self.history_label=ui.Label("Episode alarms: none",word_wrap=True)
                self.scope_label=ui.Label("Gold band = forbidden; green = allowed. Other areas UNCLASSIFIED; chips not separately calibrated.",word_wrap=True)
                self.overlay_label=ui.Label("Viewport overlay: starting",word_wrap=True)
        self.attach_overlay()

    def attach_overlay(self):
        try:
            from omni.ui import scene as sc
            from omni.kit.viewport.utility import get_active_viewport_window
            vp=get_active_viewport_window()
            if vp is None:
                self.error="No active viewport";return
            with vp.get_frame("task1_collection_safety"):
                self.scene_view=sc.SceneView()
            self.viewport=vp.viewport_api
            self.viewport.add_scene_view(self.scene_view)
            self.error=""
        except Exception as e:self.error=repr(e)

    def draw(self,recent):
        if self.scene_view is None:return
        from omni.ui import scene as sc
        from omni.ui import color as cl
        self.scene_view.scene.clear()
        pose=self.pose();p=self.panel
        with self.scene_view.scene:
            geom=p.calib["dimm_local"]
            for key,base_color in (("grasp_allowed_boxes","GREEN"),("grasp_forbidden_boxes","YELLOW")):
                for box in geom.get(key,[]):
                    lower=list(box["min"]);upper=list(box["max"])
                    if "display_x_range_m" in box:
                        lower[0],upper[0]=box["display_x_range_m"]
                    points=list(itertools.product(*zip(lower,upper)))
                    world=local_to_world(points,pose,self.scale)
                    color="RED" if key=="grasp_forbidden_boxes" and recent.get("forbidden_contact",0) else base_color
                    rgba={"GREEN":cl("#60cc60"),"YELLOW":cl("#ffcc40"),"RED":cl("#ff4040")}[color]
                    for i in range(8):
                        for j in range(i+1,8):
                            if sum(points[i][k]!=points[j][k] for k in range(3))==1:
                                sc.Line(world[i].tolist(),world[j].tolist(),color=rgba,thickness=3)
            # Show recent real contact positions (0.5 s peak hold), capped for UI cost.
            samples=[s for _,_,group in self.history for s in group][-80:]
            for sample in samples:
                color="RED" if sample["region"]=="FORBIDDEN" else ("YELLOW" if sample["region"]=="UNCLASSIFIED" else "GREEN")
                for key in sample["keys"]:
                    level=severity(recent.get(key,0.),p.cfg["demonstration_limits"][key])
                    if level=="RED" or (level=="YELLOW" and color=="GREEN"):color=level
                rgba={"GREEN":cl("#60cc60"),"YELLOW":cl("#ffcc40"),"RED":cl("#ff4040")}[color]
                center=np.asarray(sample["point"],dtype=float)
                for axis in np.eye(3)*.002:
                    sc.Line((center-axis).tolist(),(center+axis).tolist(),color=rgba,thickness=5)

    def update(self):
        p=self.panel
        if self.window is None or time.monotonic()-self.last_ui<.1:return
        self.last_ui=time.monotonic();recent=self.recent()
        valid=self.last_measure_t is not None and p.timeline.is_playing()
        for key,label in LABELS.items():
            value=self.last_metrics.get(key);limit=p.cfg["demonstration_limits"][key]
            current=f"{value:.3f}" if value is not None else "--"
            self.rows[key].text=f"{label}: {current} N / {limit:g} N | 0.5s peak {recent.get(key,0):.3f} | episode peak {self.peak.get(key,0):.3f}"
            self.rows[key].style={"color":COLORS[severity(recent.get(key) if valid else None,limit)]}
        bad=recent.get("forbidden_contact",0)>0;unknown=recent.get("unclassified_grasp_contact",0)>0
        self.grasp_label.text=("FORBIDDEN GRASP CONTACT" if bad else ("UNCLASSIFIED grasp contact - verify region" if unknown else "No forbidden contact detected"))+(" | PAUSED / last readings" if not valid else "")
        self.grasp_label.style={"color":COLORS["GRAY" if not valid else ("RED" if bad else ("YELLOW" if unknown else "GREEN"))]}
        self.history_label.text="Episode alarms: "+(", ".join(sorted(p.violations)) or "none")+" (warnings only)"
        self.history_label.style={"color":COLORS["RED" if p.violations else "GRAY"]}
        elapsed=max(0.,p.t-self.started) if self.started is not None else 0.
        self.record_label.text=(f"RECORDING {elapsed:.1f} simulation seconds | {p.active.count} transitions | no time limit\n{p.active.path}" if p.active else "NOT RECORDING | no fixed time limit")
        try:self.draw(recent)
        except Exception as e:self.error=repr(e)
        self.overlay_label.text="Overlay: "+(self.error or "active; UI only, raw RGB unchanged")

    def close(self):
        if self.viewport is not None and self.scene_view is not None:
            self.viewport.remove_scene_view(self.scene_view)
            self.scene_view.scene.clear()
        self.scene_view=None;self.viewport=None
        if self.window:self.window.destroy();self.window=None

def install(panel, show_ui=True):
    previous=getattr(panel,"feedback",None)
    if previous:previous.close()
    panel.feedback=Feedback(panel,show_ui=show_ui)
    if panel.active:
        import hashlib
        from pathlib import Path
        event=dict(event="safety_feedback_installed",sim_time=panel.t,
                   feedback_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   regions_source="configs/geometry_source.json:dimm_local")
        panel.active.physics(event)
        panel.active.meta.setdefault("instrumentation_events",[]).append(event)
    # Existing GUI instances keep their old class until reset. Wrap only once.
    if not getattr(panel,"_feedback_native_hook",False) and not getattr(panel,"_feedback_wrapped",False):
        measure=panel.measure_contacts
        def measured(dt):
            contacts=measure(dt)
            panel.feedback.consume(contacts)
            return contacts
        panel.measure_contacts=measured
        update=panel.update
        # Existing Kit callback retains its original bound method: separate UI subscription.
        import omni.kit.app
        panel.feedback_ui_sub=omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            lambda event: panel.feedback.close() if panel.closed else panel.feedback.update())
        close=panel.close
        def closed():
            panel.feedback.close()
            panel.feedback_ui_sub=None
            close()
        panel.close=closed
        panel._feedback_wrapped=True
    return panel.feedback
