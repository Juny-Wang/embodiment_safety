"""Run from Isaac Script Editor. Enable keyboard explicitly; capture within the Isaac application."""
from pathlib import Path
import sys,importlib.util,copy,json,time
PACKAGE=Path("/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_manual_safety_v002")
if str(PACKAGE) not in sys.path:sys.path.insert(0,str(PACKAGE))
# --- Plan D modules (local calib + StepFSM + session physics) ---
import importlib
import importlib.util as _plan_d_ilu
import hashlib
# Drop stale Script Editor caches (plain name + underscored loaders + pyc ghosts).
for _stale in list(sys.modules):
 if _stale in (
  "plan_d_runtime", "_plan_d_runtime",
  "step_fsm", "_plan_d_step_fsm",
  "baseline_logger", "_plan_d_baseline_logger",
  "cameras_setup", "_plan_d_cameras_setup",
  "camera_viewports", "_plan_d_camera_viewports",
  "grasp_hold_diag", "_plan_d_grasp_hold_diag",
 ) or _stale.startswith(("plan_d_runtime", "step_fsm", "baseline_logger", "cameras_setup", "camera_viewports", "grasp_hold_diag")):
  sys.modules.pop(_stale, None)
def _plan_d_load(_name,_file, require_attr=None):
 _path=PACKAGE/_file
 if not _path.is_file():
  raise FileNotFoundError("Missing Plan D file: "+str(_path))
 _src=_path.read_text(encoding="utf-8")
 _sha=hashlib.sha256(_src.encode("utf-8")).hexdigest()[:16]
 # Compile from absolute path text so Kit cannot serve a stale finder cache.
 _mod=type(sys)(_name)
 _mod.__file__=str(_path)
 _mod.__dict__["__builtins__"]=__builtins__
 sys.modules[_name]=_mod
 exec(compile(_src, str(_path), "exec"), _mod.__dict__)
 if require_attr and not hasattr(_mod, require_attr):
  raise RuntimeError(
   "Loaded %s (sha=%s, %d bytes) but missing %s — refuse silent noop."
   % (_path, _sha, len(_src), require_attr))
 print("[keyboard_panel] loaded", _file, "as", _name, "sha", _sha,
       ("has "+require_attr) if require_attr else "")
 return _mod
_step_fsm=_plan_d_load("_plan_d_step_fsm","step_fsm.py")
plan_d_runtime=_plan_d_load("_plan_d_runtime","plan_d_runtime.py",
                            require_attr="evaluate_insertion_clearance")
# Pin the function object so later physics_step cannot see a swapped module without it.
_EVAL_INSERTION_CLEARANCE=plan_d_runtime.evaluate_insertion_clearance
# Also bind plain name so late imports see the fresh module.
sys.modules["plan_d_runtime"]=plan_d_runtime
sys.modules["step_fsm"]=_step_fsm
baseline_logger=_plan_d_load("_plan_d_baseline_logger","baseline_logger.py")
sys.modules["baseline_logger"]=baseline_logger
cameras_setup=_plan_d_load("_plan_d_cameras_setup","cameras_setup.py",
                           require_attr="apply_session_cameras")
sys.modules["cameras_setup"]=cameras_setup
camera_viewports=_plan_d_load("_plan_d_camera_viewports","camera_viewports.py",
                              require_attr="open_camera_split")
sys.modules["camera_viewports"]=camera_viewports
grasp_hold_diag=_plan_d_load("_plan_d_grasp_hold_diag","grasp_hold_diag.py",
                             require_attr="GraspHoldDiag")
sys.modules["grasp_hold_diag"]=grasp_hold_diag
print("[keyboard_panel] plan_d_runtime OK; evaluate_insertion_clearance id=",
      hex(id(_EVAL_INSERTION_CLEARANCE)))
StepFSM,STEP_LABELS,PLAN_D_STEPS=_step_fsm.StepFSM,_step_fsm.STEP_LABELS,_step_fsm.STEPS
_GEOM_CALIB=json.loads((PACKAGE/"geometry_calib.json").read_text())
spec=importlib.util.spec_from_file_location("_keyboard_monitor",PACKAGE/"launch_panel.py")
monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
# Reload the controller by absolute path even in a long-running Script Editor session.
_teleop_spec=importlib.util.spec_from_file_location("_keyboard_teleop_current",PACKAGE/"franka_teleop.py")
_teleop_module=importlib.util.module_from_spec(_teleop_spec);_teleop_spec.loader.exec_module(_teleop_module)
monitor.FrankaTeleop=_teleop_module.FrankaTeleop
# No stage selection: all configured thresholds remain active, contact gates retained.
monitor.CONFIG=copy.deepcopy(monitor.CONFIG)
# Merge LOCAL-frame geometry calib into grasp/seat config (does not bake world poses).
monitor.CONFIG=plan_d_runtime.merge_calib_into_safety(monitor.CONFIG, _GEOM_CALIB)
# Reload authored safety_config plan_d / traffic_light if present on disk.
try:
 _disk=json.loads((PACKAGE/"safety_config.json").read_text())
 for k in ("plan_d","traffic_light","warning_ratio","warning_clear_ratio","insertion_clearance","cameras"):
  if k in _disk: monitor.CONFIG[k]=_disk[k]
except Exception:
 pass
for rule in monitor.CONFIG["rules"]:
 # Keep all steps active for keyboard diagnostic (warn/gate layer handles step focus).
 rule["steps"]=[0,1,2,3,4,5,6]
 if rule["key"]=="tracking_error_mm":rule["limit"]=1000.0
# Usable free-space jogging (~180 mm/s, 60 deg/s). Contact slows to 5 mm/s / 5 deg/s.
# Tracking envelope 100 mm; IK residual is best-effort (no hard stop).
monitor.CONFIG["control"].update(
 tracking_error_stop_m=0.10,
 translation_speed_m_s=0.180,
 rotation_speed_rad_s=1.0471975512,
 arm_joint_speed_rad_s=1.0,
 gripper_speed_m_s=0.06,
 ik_position_tolerance_m=0.002,
 ik_rotation_tolerance_rad=0.05235987756,
 ik_max_evaluations=80,
)
FREE_TRANS_M_S=0.180
FREE_ROT_RAD_S=1.0471975512
CONTACT_TRANS_M_S=0.005
CONTACT_ROT_RAD_S=0.0872664626
GOAL_LEAD_M=0.05
old_contact=monitor.unexpected_robot_contact
def free_contact(a,b,robot,dimm,socket,step):
 return old_contact(a,b,robot,dimm,socket,0) and old_contact(a,b,robot,dimm,socket,1)
monitor.unexpected_robot_contact=free_contact
import carb.input,omni.appwindow
import omni.ui as ui
import numpy as np
from scipy.spatial.transform import Rotation
# Resolve the sibling file directly; Kit's running import finder may cache directory listings.
_input_path=PACKAGE/"keyboard_input.py"
if not _input_path.is_file():
 raise FileNotFoundError("Missing remote keyboard mapping file: "+str(_input_path))
_input_spec=importlib.util.spec_from_file_location("_franka_keyboard_input",_input_path)
_input_module=importlib.util.module_from_spec(_input_spec)
_input_spec.loader.exec_module(_input_module)
KEYS,vectors=_input_module.KEYS,_input_module.vectors
OPEN_KEYS=getattr(_input_module,"OPEN_KEYS",{"LEFT_BRACKET","O"})
CLOSE_KEYS=getattr(_input_module,"CLOSE_KEYS",{"RIGHT_BRACKET","P"})
gripper_sign=getattr(_input_module,"gripper_sign",None)
if gripper_sign is None:
 def gripper_sign(held):
  o=bool(held & OPEN_KEYS); c=bool(held & CLOSE_KEYS)
  return 0 if o==c else (1 if o else -1)

# Live panel registry: Script Editor re-runs leave ghost windows + physics_step callbacks.
_PANEL_REGISTRY=globals().get("_PANEL_REGISTRY") or []
def _destroy_franka_control_windows():
 """Hide any leftover Franka Control windows (close() only sets visible=False)."""
 try:
  # omni.ui has no stable global window list across Kit versions; best-effort.
  import omni.ui as _ui
  for _attr in ("Workspace",):
   _ws=getattr(_ui, _attr, None)
   if _ws is None: continue
   _get=getattr(_ws, "get_windows", None) or getattr(_ws, "get_window", None)
 except Exception:
  pass
 # Fall back: destroy windows we still hold references to via registry / globals.
 for _name in ("_franka_keyboard","_franka_manual_safety","_franka_keyboard_prev"):
  _obj=globals().get(_name)
  if _obj is None: continue
  try:
   if getattr(_obj, "ui", None) is not None:
    _obj.ui.visible=False
    if hasattr(_obj.ui, "destroy"):
     try: _obj.ui.destroy()
     except Exception: pass
  except Exception:
   pass
def _destroy_all_keyboard_panels(reason="reload"):
 global _PANEL_REGISTRY
 print("[keyboard_panel] destroy_all:", reason, "n=", len(_PANEL_REGISTRY))
 for _panel in list(_PANEL_REGISTRY):
  try: _panel.close()
  except Exception as _e:
   print("[keyboard_panel] close err:", repr(_e))
  try:
   # Force-drop physics/update subs even if close() partially failed.
   _panel.physics_sub=None; _panel.update_sub=None; _panel.contact_sub=None
   if getattr(_panel, "key_subscription", None) is not None:
    try: _panel.input.unsubscribe_to_input_events(_panel.key_subscription)
    except Exception: pass
    _panel.key_subscription=None
   if getattr(_panel, "ui", None) is not None:
    _panel.ui.visible=False
    if hasattr(_panel.ui, "destroy"):
     try: _panel.ui.destroy()
     except Exception: pass
  except Exception:
   pass
 _PANEL_REGISTRY.clear()
 _destroy_franka_control_windows()
 for _name in ("_franka_keyboard","_franka_manual_safety"):
  if _name in globals():
   try: globals()[_name].close()
   except Exception: pass
   globals()[_name]=None

class SoftFrankaTeleop(monitor.FrankaTeleop):
 """Manual jogging: IK residual is best-effort, never disarms the arm."""
 def stop(self,reason="USER STOP"):
  if isinstance(reason,str) and reason.startswith(("IK ","EE ")):
   # Soft fault: keep armed, pull command to measured pose to avoid windup.
   self.error=reason
   if self.ready:
    try:
     self.sync_goal();self.cart_command=(self.goal[0].copy(),self.goal[1])
     self.last_solution=None
    except Exception:
     pass
   return
  # Never wipe commanded gripper width on stop; fingers hold via applied targets.
  saved_width=self.width_goal; was_armed=self.armed
  super().stop(reason)
  if saved_width is not None and self.ready:
   self.width_goal=float(saved_width)
  # Soft release / unfocus paths keep the arm armed so gripper can finish.
  if reason in ("KEYBOARD HOLD","UNFOCUS HOLD") and was_armed and self.ready:
   self.armed=True; self.error=None
 def tick(self,dt):
  if not self.ready:return
  self.elapsed+=dt
  # Finger-only drive when disarmed but a CLOSE/OPEN target is still pending
  # (soft_fault / brief unfocus must not leave jaws stuck open near the DIMM).
  if not self.armed:
   if self.width_goal is None or self.last_action is None:return
   if self.elapsed<self.cfg["control"]["ik_period_s"]:return
   h=min(self.elapsed,.05);self.elapsed=0
   q=np.asarray(self.last_action).copy()
   each=self.width_goal/2.; finger_step=max(float(self.cfg["control"].get("gripper_speed_m_s",0.06) or 0.06),0.06)*h/2.
   q[self.fingers]=q[self.fingers]+np.clip(each-q[self.fingers],-finger_step,finger_step)
   self.apply_targets(q); self.last_action=q.copy(); self.commands+=1; return
  if self.elapsed<self.cfg["control"]["ik_period_s"]:return
  h=min(self.elapsed,.05);self.elapsed=0
  p,Rm=self.pose();R=Rotation.from_matrix(Rm);gp,gR=self.goal
  d=gp-p;dist=np.linalg.norm(d)
  if dist>self.cfg["control"]["tracking_error_stop_m"]:
   # Soft: resync rather than hard-stop keyboard mode; still drive gripper.
   self.error="EE TRACKING HIGH - resync";self.sync_goal();self.cart_command=(self.goal[0].copy(),self.goal[1])
   h=min(h,.05); q=np.asarray(self.last_action if self.last_action is not None else self.get_joint_positions()).copy()
   each=self.width_goal/2.; finger_step=max(float(self.cfg["control"].get("gripper_speed_m_s",0.06) or 0.06),0.06)*h/2.
   q[self.fingers]=q[self.fingers]+np.clip(each-q[self.fingers],-finger_step,finger_step)
   self.apply_targets(q); self.last_action=q.copy(); self.commands+=1; return
  cp,cR=self.cart_command;cd=gp-cp;cdist=np.linalg.norm(cd)
  p_cmd=cp+cd*min(1.,self.cfg["control"]["translation_speed_m_s"]*h/max(cdist,1e-12))
  rv=(gR*cR.inv()).as_rotvec();a=np.linalg.norm(rv)
  R_cmd=Rotation.from_rotvec(rv*min(1.,self.cfg["control"]["rotation_speed_rad_s"]*h/max(a,1e-12)))*cR
  def errors(x):
   pp,rr=self.solver.compute_forward_kinematics(self.cfg["control"]["ee_frame"],np.asarray(x,dtype=float))
   return np.asarray(pp)-p_cmd,(Rotation.from_matrix(rr)*R_cmd.inv()).as_rotvec()
  pos_tol=self.cfg["control"].get("ik_position_tolerance_m",1e-6)
  rot_tol=self.cfg["control"].get("ik_rotation_tolerance_rad",1e-4)
  candidate=self.last_solution
  pe,re=errors(candidate) if candidate is not None else (np.full(3,np.inf),np.full(3,np.inf))
  ok=candidate is not None and np.linalg.norm(pe)<=pos_tol and np.linalg.norm(re)<=rot_tol
  used_cache=bool(ok)
  if not ok:
   from scipy.optimize import least_squares
   def residual(x):
    dp,dr=errors(x);return np.r_[dp*1000.,dr*100.]
   def jacobian(x):
    columns=[]
    for k in range(len(x)):
     hi=x.copy();lo=x.copy();hi[k]+=1e-5;lo[k]-=1e-5
     columns.append((residual(hi)-residual(lo))/2e-5)
    return np.column_stack(columns)
   seed=candidate if candidate is not None else self.get_joint_positions()[self.arm_indices].astype(float)
   seed=np.clip(np.asarray(seed,dtype=float),self.lower+1e-8,self.upper-1e-8)
   result=least_squares(residual,seed,bounds=(self.lower,self.upper),jac=jacobian,
     max_nfev=self.cfg["control"].get("ik_max_evaluations",40),xtol=1e-10,ftol=1e-10,gtol=1e-10)
   candidate=result.x;self.fallback_count+=1;pe,re=errors(candidate)
   ok=bool(result.success)
  self.ik_diagnostics={"local_solver_success":bool(ok),"used_verified_cache":used_cache,"position_error_m":float(np.linalg.norm(pe)),
    "orientation_error_rad":float(np.linalg.norm(re)),"fallback_count":self.fallback_count}
  if candidate is None or (not np.isfinite(candidate).all()):
   self.error="IK NONFINITE - holding";return
  candidate=np.clip(np.asarray(candidate,dtype=float),self.lower,self.upper)
  if np.linalg.norm(pe)>pos_tol or np.linalg.norm(re)>rot_tol:
   self.error="IK RESIDUAL HIGH - best effort"
  else:
   if self.error and self.error.startswith(("IK ","EE ")):self.error=None
  self.last_solution=candidate.copy()
  self.cart_command=(p_cmd,R_cmd);desired=np.clip(candidate+self.hold_bias[self.arm_indices],self.lower,self.upper)
  q=np.asarray(self.last_action).copy()
  desired=np.asarray(desired,dtype=float)
  indices=self.arm_indices
  delta=np.clip(desired-q[indices],-self.cfg["control"]["arm_joint_speed_rad_s"]*h,self.cfg["control"]["arm_joint_speed_rad_s"]*h)
  q[indices]+=delta
  measured=self.robot.get_joint_positions()
  each=self.width_goal/2.
  finger_step=max(float(self.cfg["control"].get("gripper_speed_m_s",0.06) or 0.06),0.06)*h/2.
  q[self.fingers]=self.last_action[self.fingers]+np.clip(each-self.last_action[self.fingers],-finger_step,finger_step)
  self.apply_targets(q)
  self.last_action=q.copy();self.last_q=np.array(measured).copy();self.commands+=1

monitor.FrankaTeleop=SoftFrankaTeleop

class WarningOnlyEngine(monitor.SafetyEngine):
 """Manual-panel policy: thresholds and soft faults warn/red-display only; Esc still stops."""
 def evaluate(self,metrics,step,dt):
  # Soft faults must not latch as hard sensor_error stops.
  soft=metrics.pop("_soft_fault",None)
  had=metrics.pop("sensor_error",None)
  result=super().evaluate(metrics,step,dt)
  if had is not None:metrics["sensor_error"]=had
  if soft is not None:metrics["_soft_fault"]=soft
  # Re-attach soft reasons for display without forcing stop.
  extra=[]
  if soft:extra.append("soft_fault")
  if had:extra.append("sensor_warn")
  if extra:
   result["latched_reasons"]=list(dict.fromkeys(list(result.get("latched_reasons") or [])+extra))
   # Keep measurement rows red-capable via warning flag; do not invent fake rule rows.
  result["would_stop"]=bool(result.get("stop"))
  result["warning_only"]=True
  result["warning"]=bool(result.get("latched_reasons") or soft or had)
  result["stop"]=False
  result["soft_fault"]=soft or had
  return result
 def acknowledge(self,metrics,step,dt):
  self.latched=False;self.reasons=[];self.peaks={}
  result=self.evaluate(metrics,step,dt)
  return not result["stop"]

monitor.SafetyEngine=WarningOnlyEngine

class KeyboardPanel(monitor.ManualSafetyPanel):
 def __init__(self,show_ui=True):
  self.held=set();self.keyboard_enabled=False;self.was_moving=False;self.key_subscription=None;self.key_count=0;self.last_key="none";self.key_reason="ENABLE KEYBOARD required";self.capture_log=None
  # Runtime authoring, including PhysX USD updates, belongs to the session.
  _stage=monitor.omni.usd.get_context().get_stage()
  _stage.SetEditTarget(_stage.GetSessionLayer())
  # Manual preview only: session-layer override, never save physics changes to root.
  from pxr import UsdPhysics, PhysxSchema
  for _prim in _stage.Traverse():
   if _prim.IsA(UsdPhysics.Scene):
    PhysxSchema.PhysxSceneAPI.Apply(_prim).CreateTimeStepsPerSecondAttr(240)
  # Opening is permitted throughout the composed hinge range.
  # 0.5 deg is a preview solver tolerance, not a material rating.
  for _side in ("Left","Right"):
   _joint=UsdPhysics.RevoluteJoint(_stage.GetPrimAtPath(
    monitor.CONFIG["paths"]["socket"]+"/Joints/"+_side+"Hinge"))
   if not _joint:raise RuntimeError("Missing latch hinge: "+_side)
   _low=float(_joint.GetLowerLimitAttr().Get())
   _high=float(_joint.GetUpperLimitAttr().Get())
   if not np.isfinite([_low,_high]).all():raise RuntimeError("Invalid hinge limits")
   for _rule in monitor.CONFIG["rules"]:
    if _rule["key"]==_side.lower()+"_angle_abs_deg":
     _rule["limit"]=max(abs(_low),abs(_high))+.5
  # Plan D session overlay: weak latch + thin contact + visual-only dense pins.
  try:
   self._plan_d_report=plan_d_runtime.apply_session_plan_d(_stage, _GEOM_CALIB, monitor.CONFIG)
  except Exception as _e:
   self._plan_d_report={"ok":False,"errors":[repr(_e)]}
  # Session cameras + right-side mini viewports.
  try:
   self._cameras_report=cameras_setup.apply_session_cameras(_stage, monitor.CONFIG, _GEOM_CALIB)
   print("[keyboard_panel] cameras", self._cameras_report.get("paths"), "ok", self._cameras_report.get("ok"),
         "err", self._cameras_report.get("errors"))
  except Exception as _ce:
   self._cameras_report={"ok":False,"errors":[repr(_ce)]}
   print("[keyboard_panel] cameras FAILED", self._cameras_report)
  try:
   self._cam_split_report=camera_viewports.open_camera_split(
    _stage, monitor.CONFIG, getattr(self,"_cameras_report",None))
   print("[keyboard_panel] cam_split", self._cam_split_report)
  except Exception as _se:
   self._cam_split_report={"ok":False,"errors":[repr(_se)]}
   print("[keyboard_panel] cam_split FAILED", self._cam_split_report)
  self.fsm=StepFSM(_GEOM_CALIB, monitor.CONFIG)
  # Item4: optional baseline CSV logger (env BASELINE_LOG=1). Disabled = no-op.
  try:
   _bl=baseline_logger.get_logger(create_if_env=True)
   if _bl is not None:
    baseline_logger.attach(self, enable=True, trial_tag=_bl.trial_tag)
   else:
    self._baseline_logger=None
  except Exception:
   self._baseline_logger=None
  super().__init__(show_ui=show_ui)
  # Ensure soft teleop class is the live controller instance.
  if not isinstance(self.controller,SoftFrankaTeleop):
   cfg=self.controller.cfg;self.controller=SoftFrankaTeleop(cfg)
  self.step_index=0
  self.input=carb.input.acquire_input_interface()
  self.appwindow=omni.appwindow.get_default_app_window()
  self.keyboard=self.appwindow.get_keyboard()
  self.capture_log=(self.logdir/"keyboard_events.jsonl").open("w",buffering=1)
  self.key_subscription_kb=None
  if self.keyboard is not None or True:
   # Subscribe even if get_keyboard() is briefly None; on_input accepts any KEYBOARD.
   self.key_subscription=self.input.subscribe_to_input_events(self.on_input,order=-100000)
  try: _PANEL_REGISTRY.append(self)
  except Exception: pass
 def show_cam_split(self):
  """Recreate the three mini camera viewports (safe to click anytime)."""
  try:
   _stage=monitor.omni.usd.get_context().get_stage()
   if not getattr(self,"_cameras_report",None) or not (self._cameras_report or {}).get("ok"):
    self._cameras_report=cameras_setup.apply_session_cameras(_stage, monitor.CONFIG, _GEOM_CALIB)
    print("[keyboard_panel] cameras(re)", self._cameras_report)
   self._cam_split_report=camera_viewports.open_camera_split(
    _stage, monitor.CONFIG, getattr(self,"_cameras_report",None))
   print("[keyboard_panel] cam_split(re)", self._cam_split_report)
   n=len((self._cam_split_report or {}).get("windows") or [])
   self.user_message="Camera split windows: %d"%n
  except Exception as e:
   self.user_message="Camera split failed: "+repr(e)
   print("[keyboard_panel] cam_split FAILED", repr(e))
 def build_ui(self):
  self.ui=ui.Window("Franka Control",width=620,height=650)
  self.quick_labels={}
  with self.ui.frame:
   with ui.VStack(spacing=8):
    ui.Label("1  PLAY     2  ENABLE KEYBOARD     3  HOLD ARROW KEYS",height=28,style={"font_size":18})
    with ui.HStack(height=64,spacing=8):
     ui.Button("ENABLE KEYBOARD",clicked_fn=lambda:self.command(self.enable_keyboard),
      style={"background_color":0xff386b26,"font_size":22})
     ui.Button("EMERGENCY STOP",clicked_fn=self.user_stop,
      style={"background_color":0xff3030aa,"font_size":22})
    self.control_state_label=ui.Label("NOT ENABLED - click Play, then ENABLE KEYBOARD",
     height=48,word_wrap=True,style={"font_size":20})
    self.key_live_label=ui.Label("Keys: wait ENABLE | held= | cmds=0 | focused=?",
     height=28,word_wrap=True,style={"font_size":16})
    self.reason_label=ui.Label("",height=52,word_wrap=True)
    ui.Label("Arrows: X/Y    PgUp/PgDn: Z    Shift + keys: rotate",
     height=24)
    ui.Label("[ / O]: open   ] / P: close   Esc: emergency stop",height=24)
    ui.Button("SHOW CAM SPLIT",clicked_fn=self.show_cam_split,height=36,
     style={"background_color":0xff5a3d7a,"font_size":16})
    with ui.HStack(height=48,spacing=8):
     ui.Button("Gripper OPEN",clicked_fn=lambda:self.command(self.gripper_open),
      style={"background_color":0xff2f6f9f,"font_size":18})
     ui.Button("Gripper CLOSE",clicked_fn=lambda:self.command(self.gripper_close),
      style={"background_color":0xff6f4f2f,"font_size":18})
     ui.Button("GRASP DIAG",clicked_fn=lambda:self.command(self.start_grasp_diag),
      style={"background_color":0xff3d5a7a,"font_size":16})
    self.grasp_diag_label=ui.Label("Grasp diag: idle (CLOSE auto-arms 1.5s)",height=40,word_wrap=True)
    ui.Label("FREE: 180 mm/s | 60 deg/s | contact: 5 mm/s | lead 50 mm",height=24)
    ui.Label("Plan D gates: not aligned -> block down; not dual-latched -> block release | Esc stops",height=24)
    with ui.HStack(height=32):
     ui.Label("Step:",width=50)
     self.step_combo=ui.ComboBox(0,*STEP_LABELS)
     self.step_combo.model.add_item_changed_fn(lambda m,i:self.change_step(m.get_item_value_model().as_int))
    self.gate_label=ui.Label("Gates: --",height=40,word_wrap=True)
    for key,title in [("left_press_N","Left latch"),("right_press_N","Right latch"),
                      ("axial_contact_N","DIMM axial"),("lateral_contact_N","DIMM lateral")]:
     self.quick_labels[key]=(ui.Label(title+": --",height=24),title)
    self.motion_label=ui.Label("Hand speed: -- | Gripper: --",height=26)
    with ui.ScrollingFrame():
     with ui.VStack(spacing=4):
      with ui.CollapsableFrame("Keyboard status / troubleshooting",collapsed=True):
       with ui.VStack():
        self.key_label=ui.Label("",word_wrap=True,height=60)
        self.status_label=ui.Label("",word_wrap=True,height=50)
        self.robot_label=ui.Label("",word_wrap=True,height=45)
        self.pose_label=ui.Label("",word_wrap=True,height=45)
        self.ee_label=ui.Label("",word_wrap=True,height=55)
      with ui.CollapsableFrame("All safety measurements / limits",collapsed=True):
       with ui.VStack():
        ui.Label("GRAY: unavailable or uncalibrated. All safety checks stay active (warn-only).",word_wrap=True,height=35)
        for rule in monitor.CONFIG["rules"]:
         self.metric_labels[rule["key"]]=ui.Label(rule.get("label", rule.get("key","?"))+": unavailable",word_wrap=True,height=48)
 def gripper_open(self):
  if not self.controller.ready:raise RuntimeError("Play / initialize first")
  if not self.controller.armed:self.controller.arm()
  if hasattr(self,"fsm"):
   dual,why=self.fsm.dual_latched(self.metrics)
   if self.fsm.step_name in ("LATCH_LOCK","RELEASE") and not dual:
    self.user_message="OPEN blocked: not dual-latched ("+why+")";return
  self.controller.gripper(True);self.user_message="Gripper OPEN commanded"
 def gripper_close(self):
  if not self.controller.ready:raise RuntimeError("Play / initialize first")
  if not self.controller.armed:self.controller.arm()
  fl=float(self.metrics.get("finger_left_N") or 0); fr=float(self.metrics.get("finger_right_N") or 0)
  lim=float(monitor.CONFIG.get("plan_d",{}).get("grip_force_limit_N",15.0))
  width=self.metrics.get("gripper_width_m")
  dual_touch=fl>=0.05 and fr>=0.05
  # Only refuse when a stable dual-finger over-limit grasp is already holding.
  if max(fl,fr)>=lim and dual_touch and (width is None or float(width)<0.025):
   self.user_message="CLOSE blocked: dual-finger grip force limit (%.1fN)"%lim;return
  # Button close must be usable even if soft_fault / slow config defaults.
  self.controller.cfg["control"]["gripper_speed_m_s"]=max(
    float(self.controller.cfg["control"].get("gripper_speed_m_s",0.06) or 0.06),0.06)
  self.controller.gripper(False)
  self._arm_grasp_diag("close")
  self.user_message="Gripper CLOSE commanded (width_goal=0, speed>=0.06); grasp diag armed"
 def start_grasp_diag(self):
  """Manual arm: sample current hold without commanding CLOSE."""
  if not self.controller.ready:raise RuntimeError("Play / initialize first")
  self._arm_grasp_diag("manual")
  self.user_message="Grasp diag armed (manual)"
 def _arm_grasp_diag(self, reason="close"):
  try:
   if not hasattr(self, "_grasp_diag") or self._grasp_diag is None:
    self._grasp_diag = grasp_hold_diag.GraspHoldDiag()
   self._grasp_diag.arm(float(getattr(self, "t", 0.0) or 0.0), reason=reason)
   if hasattr(self, "grasp_diag_label"):
    self.grasp_diag_label.text = self._grasp_diag.summary
  except Exception as e:
   self.user_message = "Grasp diag arm failed: " + repr(e)
 def _dimm_world_pos(self):
  try:
   poses = self.body_poses()
   if poses and "DIMM" in poses:
    q = poses["DIMM"]
    return (float(q[0]), float(q[1]), float(q[2]))
  except Exception:
   pass
  try:
   from pxr import UsdGeom, Usd
   stage = monitor.omni.usd.get_context().get_stage()
   path = (monitor.CONFIG.get("paths") or {}).get("dimm", "/World/MatchedDIMM")
   prim = stage.GetPrimAtPath(path)
   if not prim or not prim.IsValid():
    return None
   tc = Usd.TimeCode.Default()
   try:
    import omni.timeline
    tl = omni.timeline.get_timeline_interface()
    tc = Usd.TimeCode(float(tl.get_current_time()))
   except Exception:
    pass
   tr = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(tc).ExtractTranslation()
   return (float(tr[0]), float(tr[1]), float(tr[2]))
  except Exception:
   return None
 def _tick_grasp_diag(self, dt):
  g = getattr(self, "_grasp_diag", None)
  if g is None or not g.active:
   if g is not None and hasattr(self, "grasp_diag_label") and g.summary:
    self.grasp_diag_label.text = g.summary
   return
  try:
   res = g.feed(float(getattr(self, "t", 0.0) or 0.0), self.metrics, self._dimm_world_pos(), dt=float(dt))
   if hasattr(self, "grasp_diag_label"):
    self.grasp_diag_label.text = g.summary
   if res is not None:
    self.user_message = g.summary
    print("[grasp_hold_diag]", res.get("verdict"), res.get("stats"))
  except Exception as e:
   self.user_message = "Grasp diag feed failed: " + repr(e)
 def change_step(self,index):
  if self.controller.armed:self.controller.stop("STEP CHANGED - REARM")
  self.step_index=int(index)
  if hasattr(self,"fsm"):self.fsm.set_step(self.step_index)
  self.user_message="Step -> "+(PLAN_D_STEPS[self.step_index] if 0<=self.step_index<len(PLAN_D_STEPS) else "?")
 def enable_keyboard(self):
  self.clear_keys(disable=True)
  # Drop previous soft/hard latches so ENABLE always restarts cleanly.
  try:
   self.engine.latched=False;self.engine.reasons=[];self.engine.peaks={}
  except Exception:
   pass
  if hasattr(self,"fsm"):
   self.fsm.acknowledge()
  self.metrics.pop("sensor_error",None);self.metrics.pop("_soft_fault",None)
  self.initialize_robot()
  self.ack=True
  self.arm_robot()
  # Refresh keyboard handle (remote sessions may replace the device object).
  try:
   self.keyboard=self.appwindow.get_keyboard()
  except Exception:
   pass
  if self.keyboard is None:
   # Still subscribe: on_input no longer requires a fixed keyboard identity.
   print("[keyboard_panel] WARN: appwindow.get_keyboard() is None; subscribing anyway")
  if self.key_subscription is not None:self.input.unsubscribe_to_input_events(self.key_subscription)
  self.key_subscription=self.input.subscribe_to_input_events(self.on_input,order=-100000)
  # Also try keyboard-specific API when available (some Kit builds deliver only here).
  self.key_subscription_kb=getattr(self,"key_subscription_kb",None)
  if self.key_subscription_kb is not None:
   try: self.input.unsubscribe_to_keyboard_events(self.keyboard, self.key_subscription_kb)
   except Exception:
    try: self.input.unsubscribe_to_keyboard_events(self.key_subscription_kb)
    except Exception: pass
   self.key_subscription_kb=None
  if self.keyboard is not None and hasattr(self.input, "subscribe_to_keyboard_events"):
   try:
    self.key_subscription_kb=self.input.subscribe_to_keyboard_events(self.keyboard, self.on_key)
   except Exception as _ke:
    print("[keyboard_panel] keyboard_events subscribe skipped:", repr(_ke))
  self.keyboard_enabled=True;self.key_reason="READY: keyboard captured";self.user_message="";self.log_key("enabled")
  print("[keyboard_panel] ENABLE ok; sub=", self.key_subscription, "kb_sub=", getattr(self,"key_subscription_kb",None))
 def clear_keys(self,disable=False,unfocus=False):
  self.held.clear();self.was_moving=False
  if disable and not unfocus:
   self.keyboard_enabled=False
   if self.controller.ready and self.controller.armed:self.controller.stop("USER STOP")
  elif self.controller.ready and self.controller.armed:
   # Stay armed; freeze Cartesian command; keep width_goal.
   try:
    self.controller.sync_goal();self.controller.cart_command=(self.controller.goal[0].copy(),self.controller.goal[1])
   except Exception:
    pass
   if unfocus:
    # Do not disarm — gripper open/close must finish after brief focus loss (remote desktop).
    self.controller.stop("UNFOCUS HOLD")
 def log_key(self,action):
  if self.capture_log:
   self.capture_log.write(json.dumps({"wall_time":time.time(),"sim_time":self.t,"action":action,"key":self.last_key,
    "enabled":self.keyboard_enabled,"armed":self.controller.armed,"held":sorted(self.held),
    "app_focused":self.appwindow.is_focused(),"reason":self.key_reason,"commands":self.controller.commands})+"\n")
 def on_input(self,event):
  # Accept ANY keyboard device. Remote desktop / multi-device setups change the
  # Keyboard handle after ENABLE; filtering by identity silently drops all keys.
  if event.deviceType!=carb.input.DeviceType.KEYBOARD:return True
  ev=event.event
  try:
   if getattr(ev, "keyboard", None) is not None:
    self.keyboard=ev.keyboard
  except Exception:
   pass
  return self.on_key(ev)
 def on_key(self,event,*unused):
  # Isaac / carb may deliver event.input as KeyboardInput enum OR plain str
  # (remote desktop / some Kit builds). Never assume .name exists.
  _inp=getattr(event,"input",None)
  if isinstance(_inp,str): key=_inp
  else: key=str(getattr(_inp,"name",_inp) or "")
  kind=getattr(event,"type",None)
  _kname=str(getattr(kind,"name",kind) or "")
  if key not in KEYS and key not in ("ESCAPE","LEFT_SHIFT","RIGHT_SHIFT"):return True
  self.key_count+=1;self.last_key=key+":"+_kname
  if kind==carb.input.KeyboardEventType.KEY_RELEASE:
   captured=self.keyboard_enabled or key in self.held
   self.held.discard(key);self.log_key("release");return not captured
  if not self.keyboard_enabled:
   self.key_reason="DISABLED: click ENABLE KEYBOARD";self.log_key("rejected");return True
  if not self.timeline.is_playing():
   self.clear_keys(disable=True);self.key_reason="timeline paused";self.log_key("rejected");return True
  # Remote desktop / Script Editor often reports is_focused()==False while keys
  # still arrive. Never reject motion for focus; Esc / pause / DISABLE still stop.
  if key=="ESCAPE":
   self.user_stop();self.log_key("emergency_stop");return False
  if kind in (carb.input.KeyboardEventType.KEY_PRESS,carb.input.KeyboardEventType.KEY_REPEAT):
   self.held.add(key);self.key_reason="ACCEPTED";self.log_key("accepted")
  return False
 def user_stop(self):
  self.clear_keys(disable=True)
  super().user_stop()
 def timeline_event(self,event):
  if event.type in (int(monitor.omni.timeline.TimelineEventType.PAUSE),int(monitor.omni.timeline.TimelineEventType.STOP)):
   self.clear_keys(disable=True)
  super().timeline_event(event)
 def keyboard_tick(self,dt):
  if not self.keyboard_enabled:return
  if not self.timeline.is_playing():
   self.clear_keys(disable=True);self.user_message="Keyboard disabled: timeline paused; ENABLE KEYBOARD again";return
  # Intentionally ignore appwindow.is_focused() (ToDesk/Sunlogin false negatives).
  tv,rv=vectors(self.held)
  gsign=gripper_sign(self.held)
  moving=any(tv) or any(rv) or (gsign!=0)
  if not moving:
   if self.was_moving:self.clear_keys()
   return
  if not self.controller.armed:
   try:self.controller.arm()
   except Exception as e:
    self.user_message="Arm failed: "+repr(e);return
  self.was_moving=True
  c=self.controller;cfg=c.cfg["control"];p,R=c.goal
  contact=max(self.metrics.get("finger_left_N",0) or 0,self.metrics.get("finger_right_N",0) or 0)>.05
  cfg["translation_speed_m_s"]=CONTACT_TRANS_M_S if contact else FREE_TRANS_M_S
  cfg["rotation_speed_rad_s"]=CONTACT_ROT_RAD_S if contact else FREE_ROT_RAD_S
  # Keep gripper rate explicit even if safety_config had a tiny default.
  cfg["gripper_speed_m_s"]=max(float(cfg.get("gripper_speed_m_s",0.06) or 0.06),0.06)
  # Plan D StepFSM gates (block down / block release / contact-limited close).
  if hasattr(self,"fsm"):
   tv,gsign,self._last_gate=self.fsm.apply_motion_gate(tv,gsign,self.metrics)
  c.goal=(p+np.asarray(tv)*cfg["translation_speed_m_s"]*dt,
          Rotation.from_rotvec(np.asarray(rv)*cfg["rotation_speed_rad_s"]*dt)*R)
  # Bound target lead using measured FK so joint-rate limits can catch up.
  actual_pos,actual_R=c.pose()
  lead=c.goal[0]-actual_pos
  distance=float(np.linalg.norm(lead))
  if distance>GOAL_LEAD_M:
   c.goal=(actual_pos+lead*(GOAL_LEAD_M/distance),c.goal[1])
  if gsign!=0:
   c.width_goal=float(np.clip(c.width_goal+gsign*cfg["gripper_speed_m_s"]*dt,0,.08))
 def physics_step(self,dt):
  """Drive articulation while keeping threshold/IK faults as warnings (no auto-pause)."""
  if dt<=0:return
  self.dt=float(dt);self.t+=dt;self.stats["physics_steps"]+=1
  try:
   # Drop stale clearance/align RED before motion gates (same-frame PgDn).
   if hasattr(self,"fsm"):
    step_name=(PLAN_D_STEPS[self.step_index] if 0<=self.step_index<len(PLAN_D_STEPS) else "")
    if step_name not in ("ALIGN","INSERT"):
     self.fsm.light.clear_red("insertion_clearance")
     self.fsm.light.clear_red("align_xy_m")
     if self.fsm.halt and not (self.fsm.light.reasons or self.metrics.get("latch_overforce")):
      self.fsm.halt=False
      self.fsm.halt_reasons=[r for r in self.fsm.halt_reasons if r not in ("insertion_clearance","align_xy_m")]
   self.keyboard_tick(dt)
  except Exception as e:
   self.user_message="Keyboard error: "+repr(e);self.user_stop();return
  try:
   self.measure()
   # Plan D: alignment / overforce color / traffic-light RED latch (gates use these).
   try:
    align=plan_d_runtime.compute_alignment_metrics(monitor.omni.usd.get_context().get_stage(), _GEOM_CALIB, monitor.CONFIG)
    self.metrics.update({k:v for k,v in align.items() if k!="dimm_in_socket_local"})
    # Item3: prefer measure tilt; else primary formula (same as base class). Never silently use legacy board-Z fallback as tilt_deg.
    if self.metrics.get("tilt_deg") is None and align.get("tilt_deg_primary") is not None:
     self.metrics["tilt_deg"]=align["tilt_deg_primary"]
    # Pose-based end_difference if measure did not provide one
    if self.metrics.get("end_difference_mm") is None and align.get("end_difference_mm") is not None:
     self.metrics["end_difference_mm"]=align["end_difference_mm"]
    clr=_EVAL_INSERTION_CLEARANCE(self.metrics, _GEOM_CALIB, monitor.CONFIG)
    self.metrics.update(clr)
    lim=float(monitor.CONFIG.get("plan_d",{}).get("latch_overforce_N",3.0))
    lf=self.metrics.get("left_press_N"); rf=self.metrics.get("right_press_N")
    over=plan_d_runtime.set_latch_overforce_color(monitor.omni.usd.get_context().get_stage(), lf, rf, lim)
    self.metrics["latch_overforce"]=1.0 if over.get("latch_overforce") else 0.0
    # Angle-gated soft closed hold: near closed keep spring; opened / contacted -> viscous only.
    try:
     _hold=plan_d_runtime.update_latch_soft_hold(
      monitor.omni.usd.get_context().get_stage(), monitor.CONFIG, self.metrics, _GEOM_CALIB)
     self.metrics["latch_hold_left"]=(_hold.get("sides") or {}).get("left",{}).get("mode")
     self.metrics["latch_hold_right"]=(_hold.get("sides") or {}).get("right",{}).get("mode")
    except Exception as _he:
     self.metrics["latch_hold_err"]=repr(_he)
    if hasattr(self,"fsm"):
     self.fsm.step_index=self.step_index
     # Feed traffic light on key force rules (RED immediate).
     # Clearance/align RED only on ALIGN/INSERT — earlier steps (OPEN_LATCH/GRASP/EXTRACT)
     # sit far from the slot and must not latch halt/block_down/block_release.
     step_name=(PLAN_D_STEPS[self.step_index] if 0<=self.step_index<len(PLAN_D_STEPS) else "")
     clearance_active=step_name in ("ALIGN","INSERT")
     ic_ratio=self.metrics.get("insertion_clearance_ratio")
     for key,val,limit,active in (
      ("left_press_N", lf, lim, True),
      ("right_press_N", rf, lim, True),
      ("axial_contact_N", self.metrics.get("axial_contact_N"), 5.0, True),
      ("lateral_contact_N", self.metrics.get("lateral_contact_N"), 2.0, True),
      ("insertion_clearance", ic_ratio, 1.0, clearance_active),
      # Compat: keep align_xy visible to light if clearance missing (ALIGN/INSERT only)
      ("align_xy_m", self.metrics.get("align_xy_m") if ic_ratio is None else None, float(_GEOM_CALIB.get("alignment",{}).get("xy_tol_m",0.0025)), clearance_active and ic_ratio is None),
     ):
      self.fsm.light.evaluate_rule(key, None if val is None else float(val), limit, active, float(dt))
     # Drop stale clearance/align RED if we left ALIGN/INSERT (e.g. after ENABLE on OPEN_LATCH).
     if not clearance_active:
      self.fsm.light.clear_red("insertion_clearance")
      self.fsm.light.clear_red("align_xy_m")
      if self.fsm.halt and not (self.fsm.light.reasons or self.metrics.get("latch_overforce")):
       self.fsm.halt=False
       self.fsm.halt_reasons=[r for r in self.fsm.halt_reasons if r not in ("insertion_clearance","align_xy_m")]
     self._last_gate=self.fsm.evaluate_gates(self.metrics)
     # Item4 baseline logger (no-op unless BASELINE_LOG / attach)
     try:
      baseline_logger.maybe_log(self, float(dt))
     except Exception:
      pass
   except Exception as _pe:
    self.user_message="Plan D enrich warn: "+repr(_pe)
   # Never auto-pause for warn-only / soft IK. Esc -> user_stop still pauses.
   self.pause_requested=False
   # Map soft IK/EE messages to warning display, not sensor hard-stop.
   err=self.controller.error
   if err and isinstance(err,str) and err.startswith(("IK ","EE ")):
    self.metrics["_soft_fault"]=err
    self.metrics.pop("sensor_error",None)
    self.safety=self.engine.evaluate(self.metrics,self.step_index,dt)
    self.user_message=err
   elif self.metrics.get("sensor_error"):
    # Genuine measure exceptions: warn + keep going with zeroed contact already in measure().
    soft=self.metrics.get("sensor_error")
    self.metrics["_soft_fault"]=soft
    self.metrics.pop("sensor_error",None)
    self.safety=self.engine.evaluate(self.metrics,self.step_index,dt)
    self.user_message="Sensor/measure warning: "+str(soft)
  except Exception as e:
   self.errors.append(repr(e))
   # Missing contact/force API etc.: warn only, mock zeros already come from measure defaults.
   self.metrics["_soft_fault"]=repr(e)
   self.metrics.pop("sensor_error",None)
   try:self.safety=self.engine.evaluate(self.metrics,self.step_index,dt)
   except Exception:pass
   self.user_message="Measurement warning (motion continues): "+repr(e)
   self.pause_requested=False
  # Grasp diag still advances if enrich failed but metrics exist.
  try:
   self._tick_grasp_diag(float(dt))
  except Exception:
   pass
  # Drive articulation OUTSIDE the measure try: soft_fault / measure errors must not
  # skip finger CLOSE (jaws stuck open near DIMM was the user-visible failure).
  try:
   if self.controller.ready:
    # Contact-limited clamp: freeze further inward once dual-finger grasp over limit.
    try:
     lim=float(monitor.CONFIG.get("plan_d",{}).get("grip_force_limit_N",15.0))
     fl=float(self.metrics.get("finger_left_N") or 0); fr=float(self.metrics.get("finger_right_N") or 0)
     if fl>=0.05 and fr>=0.05 and max(fl,fr)>=lim and self.controller.last_action is not None:
      measured_w=float(np.sum(self.controller.last_action[self.controller.fingers]))
      if self.controller.width_goal is not None and self.controller.width_goal<measured_w-1e-4:
       self.controller.width_goal=measured_w
    except Exception:
     pass
    width_pending=(self.controller.width_goal is not None and self.controller.last_action is not None and
      abs(float(self.controller.last_action[self.controller.fingers].sum())-float(self.controller.width_goal))>1e-4)
    if not self.controller.armed and (self.keyboard_enabled or width_pending):
     try:self.controller.arm()
     except Exception:
      pass
    if self.controller.armed or self.keyboard_enabled or width_pending:
     self.controller.tick(dt)
  except Exception as _te:
   self.metrics["_soft_fault"]="tick: "+repr(_te)
 def update(self,event):
  if self.keyboard_enabled and not self.timeline.is_playing():
   self.clear_keys(disable=True);self.user_message="Keyboard disabled: timeline paused; ENABLE KEYBOARD again"
  # Do not clear keys on focus flicker (remote desktop).
  # Suppress parent auto-pause side effects from stale pause_requested.
  self.pause_requested=False
  super().update(event)
  if hasattr(self,"key_label"):
   self.key_label.text="Keys %d | %s | enabled=%s | %s | held=%s | targets=%d"%(self.key_count,self.last_key,self.keyboard_enabled,self.key_reason,",".join(sorted(self.held)),self.controller.commands)
  self.refresh_compact_ui()
 def refresh_compact_ui(self):
  if self.closed or not hasattr(self,"control_state_label"):return
  stopped=False  # warn-only panel never shows SAFETY STOP from thresholds/IK
  enabled=self.keyboard_enabled and self.controller.armed and self.timeline.is_playing()
  soft=self.safety.get("soft_fault") or self.metrics.get("_soft_fault")
  if self.user_message and not enabled and not self.keyboard_enabled:
   title="ACTION REQUIRED";color="YELLOW"
  elif not self.timeline.is_playing():
   title="PAUSED - click Play, then ENABLE KEYBOARD";color="YELLOW"
  elif enabled:
   title=("KEYBOARD ENABLED - WARNING" if (self.safety.get("warning") or soft) else "KEYBOARD ENABLED - hold keys to move");color="YELLOW" if (self.safety.get("warning") or soft) else "GREEN"
  else:
   title="NOT ENABLED - click ENABLE KEYBOARD above";color="YELLOW"
  self.control_state_label.text=title
  self.control_state_label.style={"color":monitor.rgba(monitor.COLORS[color]),"font_size":20}
  reasons=list(self.safety.get("latched_reasons",[]) or [])
  if soft and "soft_fault" not in reasons:reasons.append("soft_fault")
  self.reason_label.text=("Warning (motion allowed): "+", ".join(reasons)) if reasons else (
   self.user_message or self.controller.error or ("Release keys to hold. Esc stops." if enabled else
    "Play, then ENABLE KEYBOARD. Esc emergency-stops."))
  self.reason_label.style={"color":monitor.rgba(monitor.COLORS["YELLOW" if (reasons or self.user_message) else "GREEN"])}
  if hasattr(self,"status_label"):
   self.status_label.text=("WARNING ONLY | "+", ".join(reasons)) if reasons else "WARNING ONLY | thresholds display only"
  rows={r["key"]:r for r in self.safety.get("rows",[])}
  for key,(label,title) in self.quick_labels.items():
   row=rows.get(key,{})
   value=row.get("value");limit=row.get("limit")
   label.text=title+": "+("--" if value is None else "%.2f N"%value)+("  / limit %.2f N"%limit if limit is not None else "")
   color="RED" if key in reasons else row.get("color","GRAY")
   label.style={"color":monitor.rgba(monitor.COLORS[color])}
  speed=self.metrics.get("ee_speed_mm_s");width=self.metrics.get("gripper_width_m")
  cmd=FREE_TRANS_M_S*1000
  self.motion_label.text="Hand speed: %s mm/s (cmd %.0f) | Gripper: %s mm"%(
   "--" if speed is None else "%.2f"%speed,cmd,"--" if width is None else "%.1f"%(width*1000))
  if hasattr(self,"key_live_label"):
   try: _foc=bool(self.appwindow.is_focused())
   except Exception: _foc=None
   self.key_live_label.text="Keys %d | %s | held=%s | cmds=%d | armed=%s | focused=%s | %s"%(
    self.key_count, self.last_key, ",".join(sorted(self.held)) or "-",
    int(getattr(self.controller,"commands",0) or 0),
    bool(getattr(self.controller,"armed",False)),
    _foc, self.key_reason)
  if hasattr(self,"gate_label"):
   g=getattr(self,"_last_gate",None) or (self.fsm.last_gate if hasattr(self,"fsm") else {})
   step=g.get("step","?")
   self.gate_label.text="Step %s | align=%s dual=%s | block_down=%s block_release=%s block_close=%s | %s"%(
    step, g.get("aligned"), g.get("dual_latched"), g.get("block_down"), g.get("block_release"), g.get("block_close"),
    ",".join(g.get("reasons") or []) or "ok")
   col="RED" if g.get("block_down") or g.get("block_release") or (hasattr(self,"fsm") and self.fsm.halt) else "GREEN"
   self.gate_label.style={"color":monitor.rgba(monitor.COLORS[col])}
 def close(self):
  self.clear_keys(disable=True)
  if self.key_subscription is not None:
   try: self.input.unsubscribe_to_input_events(self.key_subscription)
   except Exception: pass
   self.key_subscription=None
  if getattr(self,"key_subscription_kb",None) is not None:
   try:
    if self.keyboard is not None:
     self.input.unsubscribe_to_keyboard_events(self.keyboard, self.key_subscription_kb)
    else:
     self.input.unsubscribe_to_keyboard_events(self.key_subscription_kb)
   except Exception: pass
   self.key_subscription_kb=None
  if self.capture_log:self.capture_log.close();self.capture_log=None
  try:
   _bl=getattr(self,"_baseline_logger",None)
   if _bl is not None: _bl.close()
  except Exception:
   pass
  try:
   if self in _PANEL_REGISTRY: _PANEL_REGISTRY.remove(self)
  except Exception: pass
  try:
   if getattr(self,"ui",None) is not None:
    self.ui.visible=False
    if hasattr(self.ui,"destroy"):
     try: self.ui.destroy()
     except Exception: pass
  except Exception: pass
  # Drop physics/update subscriptions (Feedback.close only nulls; be explicit).
  try: self.physics_sub=None
  except Exception: pass
  try: self.update_sub=None
  except Exception: pass
  try: self.contact_sub=None
  except Exception: pass
  super().close()

# Always (re)start on Script Editor exec — do not rely on __name__=="__main__".
_destroy_all_keyboard_panels("startup")
_franka_keyboard=KeyboardPanel()
# KeyboardPanel.__init__ already registers into _PANEL_REGISTRY.
print("[keyboard_panel] started clean panel id=", hex(id(_franka_keyboard)),
      "registry_n=", len(_PANEL_REGISTRY))
