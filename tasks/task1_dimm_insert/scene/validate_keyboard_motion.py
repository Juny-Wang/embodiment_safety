"""Isolated actual articulated Franka drive test. No runtime joint-state teleport."""
from pathlib import Path
import os,sys,json,csv,hashlib,signal,math
ROOT=Path("/mnt/data2/wjy_SERL/franka_sim");OUT=ROOT/"outputs/T1_rl/franka_manual_safety_v002"
for n in ["runtime/cache","runtime/config","runtime/data","runtime/tmp"]:(OUT/n).mkdir(parents=True,exist_ok=True)
for k,n in [("XDG_CACHE_HOME","cache"),("XDG_CONFIG_HOME","config"),("XDG_DATA_HOME","data"),("TMPDIR","tmp")]:os.environ[k]=str(OUT/"runtime"/n)
sys.argv=sys.argv[:1]+["--portable-root",str(OUT/"runtime")]
from isaacsim import SimulationApp
app=SimulationApp({"headless":True,"create_new_stage":False,"enable_crashreporter":False,"limit_cpu_threads":8,
 "extra_args":["--/app/settings/persistent=false","--/app/settings/loadUserConfig=false",f"--/log/file={OUT}/keyboard_motion_native.log"]})
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics,PhysxSchema,UsdUtils,Gf,Sdf,PhysicsSchemaTools
import omni.usd,omni.physx,omni.physics.tensors as tensors
sys.path.insert(0,str(OUT))
from franka_teleop import FrankaTeleop
CONFIG=json.loads((OUT/"safety_config.json").read_text())
MAIN=ROOT/"outputs/T1_rl/t1_dimm_insert_train.usda"
ASSET=ROOT/"outputs/T1_rl/socket_franka_install_v008/socket_print_v007_installed.usda"
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [MAIN,ASSET]}
CACHE=ROOT/"outputs/T1_rl/franka_latch_open_test_v1/robot_cache/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
assert CACHE.exists()
omni.usd.get_context().open_stage(str(OUT/"robot_drive_test.usda"))
for _ in range(4):app.update()
st=omni.usd.get_context().get_stage()
px=omni.physx.get_physx_interface();px.set_thread_count(0)
sim=omni.physx.get_physx_simulation_interface();sid=UsdUtils.StageCache.Get().Insert(st).ToLongInt()
forces=[];t=0
def contact(headers,data):
 for h in headers:
  a=str(PhysicsSchemaTools.intToSdfPath(h.collider0));b=str(PhysicsSchemaTools.intToSdfPath(h.collider1))
  if "ControlBox" in a+b and "ControlFloor" in a+b:
   for i in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
    j=np.array(data[i].impulse);force=float(abs(j))/.001 if j.ndim==0 else float(np.linalg.norm(j))/.001
    forces.append((t,force))
sub=sim.subscribe_contact_report_events(contact);sim.attach_stage(sid)
def step():
 global t
 sim.simulate(.001,t);sim.fetch_results();t+=.001
for _ in range(1000):step()
sv=tensors.create_simulation_view("numpy",sid);sv.set_subspace_roots("/")
import keyboard_panel as kp
CONFIG=kp.monitor.CONFIG
robot=kp.monitor.FrankaTeleop(CONFIG);names=robot.initialize(sv)
hand_paths=[str(p.GetPath()) for p in st.Traverse() if p.GetName()=="panda_hand" and p.HasAPI(UsdPhysics.RigidBodyAPI)]
assert len(hand_paths)==1,hand_paths
hand=sv.create_rigid_body_view(hand_paths[0])
initial_hand=hand.get_transforms()[0].copy();start,R=robot.pose()
pe=float(np.linalg.norm(initial_hand[:3]-start));re=float(np.linalg.norm((Rotation.from_matrix(R).inv()*Rotation.from_quat(initial_hand[3:])).as_rotvec()))
assert pe<.0005 and re<math.radians(.5),("FK mismatch",pe,re)
import types,carb.input
panel=kp.KeyboardPanel.__new__(kp.KeyboardPanel)
panel.controller=robot;panel.held=set();panel.keyboard_enabled=True;panel.was_moving=False
panel.key_count=0;panel.last_key="";panel.key_reason="";panel.capture_log=None
panel.appwindow=types.SimpleNamespace(is_focused=lambda:True)
panel.timeline=types.SimpleNamespace(is_playing=lambda:True)
panel.safety={"stop":False};panel.metrics={};panel.t=0
inp=carb.input.acquire_input_interface();provider=inp.get_input_provider()
keyboard=provider.create_keyboard("Franka isolated keyboard validation")
seen=[]
panel.keyboard=keyboard
downstream=inp.subscribe_to_input_events(lambda e: seen.append(e.event.input.name) or True,order=0)
capture=inp.subscribe_to_input_events(panel.on_input,order=-100000)
def event(name,pressed):
 provider.buffer_keyboard_key_event(keyboard,carb.input.KeyboardEventType.KEY_PRESS if pressed else carb.input.KeyboardEventType.KEY_RELEASE,getattr(carb.input.KeyboardInput,name),0)
 inp.distribute_buffered_events()
def advance(n):
 for i in range(n):
  panel.keyboard_tick(.001);robot.tick(.001);step()
  if robot.error and robot.error not in ("KEYBOARD HOLD",):raise RuntimeError(str((robot.error,robot.ik_diagnostics)))
results=[]
for name,axis,sign in [("UP",0,1),("DOWN",0,-1),("LEFT",1,1),("RIGHT",1,-1),("PAGE_UP",2,1),("PAGE_DOWN",2,-1)]:
 start=hand.get_transforms()[0].copy();event(name,True);advance(1000)
 end=hand.get_transforms()[0].copy()
 event(name,False);advance(1);n=robot.commands;advance(100)
 delta=(end[:3]-start[:3])*1000
 results.append({"key":name,"delta_mm":delta.tolist(),"pass":bool(delta[axis]*sign>1.0),"no_targets_after_release":robot.commands==n})
 print("TRANSLATION",results[-1],flush=True)
for name,axis in [("UP",0),("LEFT",1),("PAGE_UP",2)]:
 start=hand.get_transforms()[0].copy()
 event("LEFT_SHIFT",True);event(name,True);advance(1000)
 end=hand.get_transforms()[0].copy()
 event(name,False);event("LEFT_SHIFT",False);advance(100)
 rv=(Rotation.from_quat(end[3:])*Rotation.from_quat(start[3:]).inv()).as_rotvec()*180/math.pi
 results.append({"key":"SHIFT+"+name,"rotation_world_deg":rv.tolist(),"pass":bool(rv[axis]>1.)})
 print("ROTATION",results[-1],flush=True)
# Physical release and whole-app loss of focus stop further drive commands.
event("UP",True);advance(50)
panel.appwindow.is_focused=lambda:False
advance(1);n=robot.commands;advance(100)
focus_pass=not panel.keyboard_enabled and not panel.held and robot.commands==n
result={"cases":results,"all_motions_pass":all(r["pass"] for r in results),
 "ordered_input_consumed":len(seen)==0,"downstream_keys":seen,"input_events":panel.key_count,
 "app_focus_loss_stops":focus_pass,"robot_commands":robot.commands,
 "runtime_joint_pose_writes":False,"physical_keyboard_remote_desktop_tested":False,
 "tracking_stop_m":CONFIG["control"]["tracking_error_stop_m"],"free_speed_m_s":.01,
 "source_files_unchanged":all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())}
(OUT/"keyboard_motion_acceptance.json").write_text(json.dumps(result,indent=2));print("KEYBOARD_MOTION_RESULT",json.dumps(result),flush=True)
inp.unsubscribe_to_input_events(capture);inp.unsubscribe_to_input_events(downstream);provider.destroy_keyboard(keyboard)
sub=None;sim.detach_stage()
signal.signal(signal.SIGALRM,lambda *a:os._exit(0));signal.alarm(5);app.close()
