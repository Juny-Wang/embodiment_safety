"""Isaac Script Editor entry point; all project files remain on isaac-b."""
from pathlib import Path
import sys,importlib.util,json,time,math
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics,Gf,PhysicsSchemaTools
import omni.usd,omni.physx,omni.ui as ui,omni.timeline
PACKAGE=Path("/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_manual_safety_v002")
if str(PACKAGE) not in sys.path:sys.path.insert(0,str(PACKAGE))
from safety_core import SafetyEngine,COLORS,observation,dimm_contact_kind,unexpected_robot_contact
from franka_teleop import FrankaTeleop
spec=importlib.util.spec_from_file_location("_v002_contact_base",PACKAGE.parent/"dimm_safety_feedback_v001/live_safety_feedback.py")
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);base.OUT=PACKAGE
CONFIG=json.loads((PACKAGE/"safety_config.json").read_text())
def rgba(c):
 r,g,b,a=c;return int(a*255)<<24|int(b*255)<<16|int(g*255)<<8|int(r*255)
def rot(q):return Rotation.from_quat([*q.GetImaginary(),q.GetReal()])
class ManualSafetyPanel(base.Feedback):
 def __init__(self,show_ui=True):
  self.config=CONFIG;self.engine=SafetyEngine(CONFIG);self.controller=FrankaTeleop(CONFIG)
  self.safety={"rows":[],"stop":False};self.metrics={};self.metric_labels={}
  self.user_message="";self.ack=False;self.last_refresh=0.;self.paused_for_safety=False;self.slip_reference=None;self.draw_iface=None
  super().__init__(show_ui=False,stop_enabled=False)
  # Normal-only reports avoid requesting SDF friction-anchor material queries.
  self.contact_sub=None
  self.contact_sub=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(lambda h,d:self.on_contacts(h,d,[]))
  self.detail=(self.logdir/"task_frames.jsonl").open("w",buffering=1)
  (self.logdir/"config_snapshot.json").write_text(json.dumps(CONFIG,indent=2))
  self.events=self.timeline.get_timeline_event_stream().create_subscription_to_pop(self.timeline_event)
  self.pause_requested=False;self.internal_overlap_mm=0.
  try:
   from isaacsim.core.utils.extensions import enable_extension
   enable_extension("isaacsim.util.debug_draw")
   from isaacsim.util.debug_draw import _debug_draw
   self.draw_iface=_debug_draw.acquire_debug_draw_interface()
  except Exception as e:self.errors.append("debug draw: "+repr(e))
  if show_ui:self.build_ui()
 def build_ui(self):
  self.ui=ui.Window("Franka manual operation + measured safety",width=880,height=960)
  with self.ui.frame:
   with ui.ScrollingFrame():
    with ui.VStack(spacing=5):
     ui.Label("PROTOTYPE / DEMONSTRATION THRESHOLDS / mechanical task NOT ACCEPTED",word_wrap=True)
     ui.Label("Hand frame: panda_hand. XYZ and rotations are WORLD axes. No automatic motion on load.",word_wrap=True)
     self.status_label=ui.Label("Pause -> run script -> Play -> Initialize -> acknowledge -> Arm",word_wrap=True)
     with ui.HStack():
      ui.Button("Initialize from MEASURED pose",clicked_fn=lambda:self.command(self.initialize_robot))
      ui.Button("ARM / acknowledge cleared stop",clicked_fn=lambda:self.command(self.arm_robot))
      ui.Button("STOP + pause",clicked_fn=self.user_stop,style={"background_color":0xff3030aa})
     with ui.HStack():
      cb=ui.CheckBox();cb.model.add_value_changed_fn(lambda m:setattr(self,"ack",m.as_bool))
      ui.Label("I understand thresholds are uncalibrated; manual diagnostic operation only.",word_wrap=True)
     self.combo=ui.ComboBox(0,*base.STEPS)
     self.combo.model.add_item_changed_fn(lambda m,i:self.change_step(m.get_item_value_model().as_int))
     ui.Label("Each click: 0.25 mm translation / 0.5 deg rotation; speed <=1 mm/s and 2 deg/s.",word_wrap=True)
     for rotate in [False,True]:
      with ui.HStack():
       for axis,name in enumerate(["X","Y","Z"]):
        for sign,suffix in [(-1,"-"),(1,"+")]:
         ui.Button(("Rotate " if rotate else "Move ")+name+suffix,
          clicked_fn=lambda a=axis,s=sign,r=rotate:self.command(lambda:self.controller.jog(a,s,r)))
     with ui.HStack():
      ui.Button("Gripper OPEN",clicked_fn=lambda:self.command(lambda:self.controller.gripper(True)))
      ui.Button("Gripper CLOSE (uncalibrated)",clicked_fn=lambda:self.command(lambda:self.controller.gripper(False)))
     self.robot_label=ui.Label("Robot: not initialized",word_wrap=True)
     self.pose_label=ui.Label("Pose: waiting for physical step",word_wrap=True)
     self.ee_label=ui.Label("Hand xyz/rotation: waiting for initialization",word_wrap=True)
     ui.Label("Values are actual contact/rigid-body measurements; force excludes friction. Not a wrist FT sensor.",word_wrap=True)
     for rule in CONFIG["rules"]:
      self.metric_labels[rule["key"]]=ui.Label(rule.get("label", rule.get("key","?"))+": unavailable",word_wrap=True)
     ui.Label("GRAY = unavailable/unconfigured/inactive; GREEN = below demo warning; YELLOW = warning; RED = stop. Red stop is latched.",word_wrap=True)
     ui.Label("Grasp regions / absolute seating depth / true hook engagement / material damage are NOT calibrated. No stage implies task success.",word_wrap=True)
     ui.Label("Safety stop cancels further targets and pauses timeline. This is not a certified robot brake.",word_wrap=True)
     ui.Button("Close controller + monitor",clicked_fn=self.close)
 def command(self,fn):
  try:fn();self.user_message=""
  except Exception as e:
   self.user_message=str(e)
   if hasattr(self,"status_label"):self.status_label.text=str(e)
 def change_step(self,index):
  if self.controller.armed:self.controller.stop("STEP CHANGED - REARM")
  self.step_index=index
 def initialize_robot(self):
  if not self.timeline.is_playing() or self.stats["physics_steps"]<2:raise RuntimeError("Click Play and wait two physics steps first")
  st=self.stage
  for p in st.Traverse():
   a=p.GetAttribute("node:type")
   if a and "IsaacArticulationController" in str(a.Get()):
    raise RuntimeError("Another OmniGraph articulation controller exists; disable it before manual control: "+str(p.GetPath()))
  self.controller.initialize()
  try:self.check_hand_model()
  except Exception:
   self.controller.ready=False;raise
  self.slip_reference=None
  if hasattr(self,"hand_pose"):del self.hand_pose
 def check_hand_model(self):
  import omni.physics.tensors as tensors
  candidates=[p for p in self.stage.Traverse() if str(p.GetPath()).startswith(CONFIG["paths"]["robot"]+"/") and p.GetName()=="panda_hand" and p.HasAPI(UsdPhysics.RigidBodyAPI)]
  if len(candidates)!=1:raise RuntimeError("Cannot identify physical panda_hand for FK validation")
  sv=tensors.create_simulation_view("numpy",getattr(self,"simulation_stage_id",-1))
  hand=sv.create_rigid_body_view(str(candidates[0].GetPath()))
  physical=hand.get_transforms()[0]
  pos,R=self.controller.pose()
  pe=float(np.linalg.norm(pos-physical[:3]))
  re=float(np.linalg.norm((Rotation.from_matrix(R).inv()*Rotation.from_quat(physical[3:])).as_rotvec()))
  self.model_error=(pe,re)
  if pe>.0005 or re>math.radians(.5):
   self.controller.ready=False
   raise RuntimeError("Robot URDF/physical hand mismatch: %.3f mm, %.3f deg"%(pe*1000,math.degrees(re)))
 def arm_robot(self):
  if not self.ack:raise RuntimeError("Acknowledge prototype thresholds first")
  if not self.timeline.is_playing():raise RuntimeError("Resume Play before arming")
  if not self.engine.acknowledge(self.metrics,self.step_index,self.dt or .001):raise RuntimeError("Current red condition must clear before rearming")
  self.controller.arm();self.paused_for_safety=False
 def user_stop(self):
  if self.controller.ready:self.controller.stop()
  self.timeline.pause()
 def timeline_event(self,event):
  if event.type==int(omni.timeline.TimelineEventType.STOP):
   self.controller.armed=False;self.controller.ready=False;self.views=None;self.slip_reference=None
   if hasattr(self,"hand_pose"):del self.hand_pose
 def on_contacts(self,headers,data,friction):
  try:
   for h in headers:
    a=str(PhysicsSchemaTools.intToSdfPath(h.collider0));b=str(PhysicsSchemaTools.intToSdfPath(h.collider1))
    if not any(k in a+b for k in [base.SOCKET,base.DIMM,CONFIG["paths"]["robot"],"/World/Control"]):continue
    for i in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
     c=data[i]
     if a.startswith(base.SOCKET) and b.startswith(base.SOCKET):
      self.internal_overlap_mm=max(self.internal_overlap_mm,max(0.,-float(c.separation)*1000));continue
     self.pending.append({"a":a,"b":b,"point":list(c.position),"normal":list(c.normal),
       "impulse":np.asarray(c.impulse,dtype=float).tolist(),"reported_dt_s":self.dt,
       "separation_mm":float(c.separation)*1000})
  except Exception as e:self.errors.append(repr(e))
 def physics_step(self,dt):
  if dt<=0:return
  self.dt=float(dt);self.t+=dt;self.stats["physics_steps"]+=1
  try:
   self.measure()
   if self.safety["stop"] and self.controller.armed:
    self.controller.stop("SAFETY: "+",".join(self.safety["latched_reasons"]))
    self.pause_requested=True
   else:
    self.controller.tick(dt)
    if self.controller.error and self.controller.error.startswith(("IK ","EE ")):
     self.metrics["sensor_error"]=self.controller.error;self.safety=self.engine.evaluate(self.metrics,self.step_index,dt);self.pause_requested=True
  except Exception as e:
   self.errors.append(repr(e));self.metrics["sensor_error"]=repr(e)
   self.safety=self.engine.evaluate(self.metrics,self.step_index,dt)
   if self.controller.ready:self.controller.stop("MEASUREMENT/CONTROL ERROR")
   self.pause_requested=True
 def measure(self):
  # Preserve the dt attached to each actual report even if simulation dt changes.
  for c in self.pending:
   dt=c.get("reported_dt_s") or self.dt
   c["raw_impulse"]=c["impulse"]
   c["impulse"]=(np.asarray(c["impulse"])*self.dt/dt).tolist()
  super().measure();self.pause_requested=False
  p=self.body_poses();cache=UsdGeom.XformCache()
  axis=np.asarray(self.last["axis"])
  metrics={"left_press_N":0.,"right_press_N":0.,"left_torque_Nm":0.,"right_torque_Nm":0.,
   "finger_left_N":0.,"finger_right_N":0.,"axial_contact_N":0.,"lateral_contact_N":0.,
   "tilt_deg":self.last.get("tilt_deg"),"end_difference_mm":self.last.get("end_height_difference_mm"),
   "dimm_moment_Nm":0. if "DIMM" in p else None,"dimm_contact_N":0.,"unexpected_robot_contact_N":0.,
   "penetration_mm":max(self.internal_overlap_mm,max(0,-self.last["min_separation_mm"])),"slip_mm":None,"forbidden_contact":None}
  for side in ["Left","Right"]:
   angle=self.last.get(side.lower()+"_angle_deg")
   metrics[side.lower()+"_angle_abs_deg"]=None if angle is None else abs(angle)
  hinges={}
  for side in ["Left","Right"]:
   if side not in p:continue
   j=self.stage.GetPrimAtPath(base.SOCKET+"/Joints/"+side+"Hinge")
   q=p[side];R=Rotation.from_quat(q[3:])
   pivot=q[:3]+R.apply(np.array(j.GetAttribute("physics:localPos1").Get()))
   jr=rot(j.GetAttribute("physics:localRot1").Get())
   idx={"X":0,"Y":1,"Z":2}[j.GetAttribute("physics:axis").Get()]
   u=np.eye(3)[idx];hinges[side]=(pivot,(R*jr).apply(u))
  grasp_points=[];environment_moment=np.zeros(3)
  for c in self.last.get("contacts",[]):
   a,b=c["a"],c["b"];F=np.array(c["force_N"]);n=c["normal_force_N"];pt=np.array(c["point"])
   for side,(pivot,u) in hinges.items():
    body=base.SOCKET+"/"+side+"Latch"
    if (a.startswith(body) or b.startswith(body)) and not (a.startswith(base.SOCKET) and b.startswith(base.SOCKET)):
     metrics[side.lower()+"_press_N"]+=n
     signed=F if a.startswith(body) else -F
     metrics[side.lower()+"_torque_Nm"]+=float(np.dot(np.cross(pt-pivot,signed),u))
   if unexpected_robot_contact(a,b,CONFIG["paths"]["robot"],base.DIMM,base.SOCKET,self.step_index):
    metrics["unexpected_robot_contact_N"]+=n
   kind=dimm_contact_kind(a,b,base.DIMM,CONFIG["paths"]["robot"])
   if kind=="environment":
    metrics["dimm_contact_N"]+=n
    metrics["axial_contact_N"]+=abs(float(np.dot(F,axis)))
    metrics["lateral_contact_N"]+=float(np.linalg.norm(F-np.dot(F,axis)*axis))
    if "DIMM" in p:
     signed=F if a.startswith(base.DIMM) else -F
     environment_moment+=np.cross(pt-p["DIMM"][:3],signed)
   elif kind in ("finger_left","finger_right"):
    metrics[kind+"_N"]+=n;grasp_points.append(pt)
  if "DIMM" in p:metrics["dimm_moment_Nm"]=float(np.linalg.norm(environment_moment))
  # Calibration boxes are in the actual DIMM body local frame, SI meters.
  regions=CONFIG["grasp_regions"]
  if regions["calibrated"] and "DIMM" in p:
   q=p["DIMM"];R=Rotation.from_quat(q[3:]);bad=False
   for pt in grasp_points:
    local=R.inv().apply(pt-q[:3])
    inside=lambda box:bool(np.all(local>=box["min_m"]) and np.all(local<=box["max_m"]))
    if any(inside(b) for b in regions["forbidden_boxes"]) or not any(inside(b) for b in regions["allowed_boxes"]):bad=True
   metrics["forbidden_contact"]=float(bad)
  if self.controller.ready:
   hp,hR=self.controller.pose()
   if hasattr(self,"hand_pose"):metrics["ee_speed_mm_s"]=float(np.linalg.norm(hp-self.hand_pose[0])/self.dt)*1000
   self.hand_pose=(np.array(hp),np.array(hR))
   metrics["tracking_error_mm"]=float(np.linalg.norm(hp-self.controller.goal[0]))*1000
   q=self.controller.robot.get_joint_positions()
   metrics["gripper_width_m"]=float(np.sum(q[self.controller.fingers]))
   if "DIMM" in p:
    rel=Rotation.from_matrix(hR).inv().apply(p["DIMM"][:3]-hp)
    if metrics["finger_left_N"]>.05 and metrics["finger_right_N"]>.05 and self.slip_reference is None:self.slip_reference=rel
    if self.slip_reference is not None:metrics["slip_mm"]=float(np.linalg.norm(rel-self.slip_reference)*1000)
  self.internal_overlap_mm=0.
  self.metrics=metrics;self.safety=self.engine.evaluate(metrics,self.step_index,self.dt)
  self.detail.write(json.dumps({"time_s":self.t,"physics_dt_s":self.dt,"stage":self.step_index,
   "observation":observation(metrics),"safety":{"stop":self.safety["stop"],"reasons":self.safety["latched_reasons"],"colors":{r["key"]:r["color"] for r in self.safety["rows"]}},
   "armed":self.controller.armed,"command":None if self.controller.last_action is None else self.controller.last_action.tolist(),
   "joint_positions":None if not self.controller.ready else self.controller.robot.get_joint_positions().tolist(),
   "task_success":False,"ik":getattr(self.controller,"ik_diagnostics",None),"friction_impulses_included":False})+"\n")
 def draw(self):
  if self.draw_iface is None:return
  starts=[];ends=[];colors=[];points=[];pcolors=[]
  for c in self.last.get("contacts",[]):
   F=np.array(c["force_N"]);mag=np.linalg.norm(F)
   if mag<.002:continue
   p=np.array(c["point"]);v=F/mag;e=p+v*min(.025,.002*mag)
   co=COLORS["RED" if mag>=3 else "YELLOW" if mag>=1.5 else "GREEN"]
   h=np.cross(v,[0,0,1])
   if np.linalg.norm(h)<.01:h=np.cross(v,[0,1,0])
   h/=np.linalg.norm(h)
   for a,b in [(p,e),(e,e-v*.001+h*.0005),(e,e-v*.001-h*.0005)]:
    starts.append(tuple(a));ends.append(tuple(b));colors.append(co)
   points.append(tuple(p));pcolors.append(co)
  center=self.last.get("dimm_center")
  if center is not None:
   pos=np.array(center)
   starts.append(tuple(pos));ends.append(tuple(pos+np.array(self.last["axis"])*.04));colors.append((.1,.7,1.,1.))
  if self.controller.ready:
   hp,hR=self.controller.pose()
   for i,co in enumerate([(1,0,0,1),(0,1,0,1),(0,0,1,1)]):
    starts.append(tuple(hp));ends.append(tuple(hp+hR[:,i]*.025));colors.append(co)
  self.draw_iface.clear_lines();self.draw_iface.clear_points()
  if starts:self.draw_iface.draw_lines(starts,ends,colors,[2.]*len(starts))
  if points:self.draw_iface.draw_points(points,pcolors,[7.]*len(points))
 def update(self,event):
  if self.closed:return
  if hasattr(self,"ui") and self.ui and not self.ui.visible:self.user_stop();self.close();return
  if omni.usd.get_context().get_stage()!=self.stage:self.controller.ready=False;self.controller.armed=False;self.close();return
  if self.pause_requested:
   self.pause_requested=False;self.paused_for_safety=True;self.stats["stop_count"]+=1;self.timeline.pause()
  now=time.monotonic()
  if now-self.last_refresh<.05:return
  self.last_refresh=now;self.draw()
  if not hasattr(self,"status_label"):return
  self.status_label.text=("STOP: "+", ".join(self.safety.get("latched_reasons",[])) if self.safety["stop"] else "MONITORING - not task acceptance")+" | "+self.user_message
  self.status_label.style={"color":rgba(COLORS["RED"] if self.safety["stop"] else COLORS["YELLOW"])}
  self.robot_label.text="Robot %s | armed %s | commands %s | %s"%(self.controller.ready,self.controller.armed,self.controller.commands,self.controller.error or "")
  self.pose_label.text="Actual dt %.6f s | L angle %s | R angle %s | log %s"%(self.dt or 0,self.last.get("left_angle_deg"),self.last.get("right_angle_deg"),self.logdir.name)
  if self.controller.ready and hasattr(self,"hand_pose"):
   p,R=self.hand_pose;angles=Rotation.from_matrix(R).as_euler("xyz",degrees=True)
   self.ee_label.text="Hand xyz [mm] %s | Euler xyz [deg] %s | gripper %.3f mm"%(np.round(p*1000,3).tolist(),np.round(angles,2).tolist(),self.metrics.get("gripper_width_m",0)*1000)
  for row in self.safety.get("rows",[]):
   label=self.metric_labels.get(row["key"])
   if label:
    value="unavailable" if row["value"] is None else "%.4f"%row["value"]
    lim="UNCONFIGURED" if row["limit"] is None else str(row["limit"])
    display="RED" if row["key"] in self.safety["latched_reasons"] else row["color"]
    label.text="%s: %s %s | limit %s | %s | peak %s | exposure %.3f s"%(row["label"],value,row["unit"],lim,display,row.get("peak"),row["exposure_s"])
    label.style={"color":rgba(COLORS[display])}
 def close(self):
  if getattr(self,"closed",True):return
  if self.controller.ready:self.controller.stop("PANEL CLOSED")
  self.events=None
  if self.draw_iface:self.draw_iface.clear_lines();self.draw_iface.clear_points()
  self.detail.close()
  super().close()

if __name__=="__main__":
 if "_franka_manual_safety" in globals():_franka_manual_safety.close()
 _franka_manual_safety=ManualSafetyPanel()
