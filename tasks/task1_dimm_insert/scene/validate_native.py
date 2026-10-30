"""Isolated actual articulated Franka drive test. No runtime joint-state teleport."""
from pathlib import Path
import os,sys,json,csv,hashlib,signal,math
ROOT=Path("/mnt/data2/wjy_SERL/franka_sim");OUT=ROOT/"outputs/T1_rl/franka_manual_safety_v002"
for n in ["runtime/cache","runtime/config","runtime/data","runtime/tmp"]:(OUT/n).mkdir(parents=True,exist_ok=True)
for k,n in [("XDG_CACHE_HOME","cache"),("XDG_CONFIG_HOME","config"),("XDG_DATA_HOME","data"),("TMPDIR","tmp")]:os.environ[k]=str(OUT/"runtime"/n)
sys.argv=sys.argv[:1]+["--portable-root",str(OUT/"runtime")]
from isaacsim import SimulationApp
app=SimulationApp({"headless":True,"create_new_stage":False,"enable_crashreporter":False,"limit_cpu_threads":8,
 "extra_args":["--/app/settings/persistent=false","--/app/settings/loadUserConfig=false",f"--/log/file={OUT}/native.log"]})
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
candidate=Usd.Stage.CreateNew(str(OUT/"manual_franka_scene.usda"))
UsdGeom.SetStageMetersPerUnit(candidate,1);UsdGeom.SetStageUpAxis(candidate,"Z")
world=UsdGeom.Xform.Define(candidate,"/World");candidate.SetDefaultPrim(world.GetPrim())
candidate.DefinePrim("/World/Robot","Xform").GetReferences().SetReferences([Sdf.Reference(str(CACHE))])
candidate.GetRootLayer().subLayerPaths=[str(MAIN)]
home=[0,-.785398,0,-2.356194,0,1.570796,.785398]
joint_count=0
for p in list(candidate.Traverse()):
 n=p.GetName()
 if str(p.GetPath()).startswith("/World/Robot/") and n.startswith("panda_joint") and n[-1:] in list("1234567"):
  q=home[int(n[-1])-1]
  PhysxSchema.JointStateAPI.Apply(p,"angular").CreatePositionAttr(math.degrees(q))
  drive=UsdPhysics.DriveAPI(p,"angular")
  if drive:drive.GetTargetPositionAttr().Set(math.degrees(q))
  joint_count+=1
 elif str(p.GetPath()).startswith("/World/Robot/") and n in ["panda_finger_joint1","panda_finger_joint2"]:
  PhysxSchema.JointStateAPI.Apply(p,"linear").CreatePositionAttr(.04)
  drive=UsdPhysics.DriveAPI(p,"linear")
  if drive:drive.GetTargetPositionAttr().Set(.04)
assert joint_count==7,("Could not configure authored diagnostic home",joint_count)
ph=candidate.GetPrimAtPath("/World/physicsScene")
if ph:PhysxSchema.PhysxSceneAPI.Apply(ph).CreateTimeStepsPerSecondAttr(1000)
candidate.GetRootLayer().customLayerData={"status":"MANUAL_DIAGNOSTIC_NOT_MECHANICAL_OR_TRAINING_ACCEPTED",
 "robot_initial_pose":"authored home; no runtime teleport","local_robot_cache":str(CACHE)}
candidate.GetRootLayer().Save()
# Native free-space robot drive test references ONLY robot, not socket or tray.
st=Usd.Stage.CreateNew(str(OUT/"robot_drive_test.usda"))
UsdGeom.SetStageMetersPerUnit(st,1);UsdGeom.SetStageUpAxis(st,"Z")
UsdGeom.Xform.Define(st,"/World")
st.DefinePrim("/World/Robot","Xform").GetReferences().AddReference(str(OUT/"manual_franka_scene.usda"),"/World/Robot")
ph=UsdPhysics.Scene.Define(st,"/World/physicsScene");ph.CreateGravityMagnitudeAttr(9.81);ph.CreateGravityDirectionAttr((0,0,-1))
ps=PhysxSchema.PhysxSceneAPI.Apply(ph.GetPrim());ps.CreateTimeStepsPerSecondAttr(1000);ps.CreateSolverTypeAttr("TGS");ps.CreateEnableGPUDynamicsAttr(False)
for name,z,dynamic in [("ControlFloor",.005,False),("ControlBox",.018,True)]:
 c=UsdGeom.Cube.Define(st,"/World/"+name);c.CreateSizeAttr(.01);c.AddTranslateOp().Set((1,0,z))
 UsdPhysics.CollisionAPI.Apply(c.GetPrim())
 if dynamic:
  UsdPhysics.RigidBodyAPI.Apply(c.GetPrim());UsdPhysics.MassAPI.Apply(c.GetPrim()).CreateMassAttr(.02)
  PhysxSchema.PhysxContactReportAPI.Apply(c.GetPrim()).CreateThresholdAttr(0.)
st.GetRootLayer().Save()
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
robot=FrankaTeleop(CONFIG);names=robot.initialize(sv)
hand_paths=[str(p.GetPath()) for p in st.Traverse() if p.GetName()=="panda_hand" and p.HasAPI(UsdPhysics.RigidBodyAPI)]
assert len(hand_paths)==1,hand_paths
hand=sv.create_rigid_body_view(hand_paths[0])
initial_hand=hand.get_transforms()[0].copy();start,R=robot.pose()
pe=float(np.linalg.norm(initial_hand[:3]-start));re=float(np.linalg.norm((Rotation.from_matrix(R).inv()*Rotation.from_quat(initial_hand[3:])).as_rotvec()))
assert pe<.0005 and re<math.radians(.5),("FK mismatch",pe,re)
robot.arm();robot.jog(0,1)
rows=[]
for i in range(1300):
 robot.tick(.001);step()
 if i%10==0:
  p,R=robot.pose();rows.append([t,"jog",*p.tolist(),*robot.robot.get_joint_positions().tolist()])
jog_end=hand.get_transforms()[0].copy()
print("JOG_STATUS",json.dumps({"armed":robot.armed,"reason":robot.error,"commands":robot.commands,"initial_q":robot.initial_q.tolist(),"current_q":robot.get_joint_positions().tolist(),"movement_m":(jog_end[:3]-initial_hand[:3]).tolist()}),flush=True)
assert robot.armed,("Controller stopped during jog",robot.error)
robot.gripper(False)
width0=float(sum(robot.robot.get_joint_positions()[robot.fingers]))
for i in range(500):
 robot.tick(.001);step()
width1=float(sum(robot.robot.get_joint_positions()[robot.fingers]));print("GRIP_DONE",width0,width1,robot.error,flush=True)
robot.stop("VALIDATION STOP");count=robot.commands;stop_hand=hand.get_transforms()[0].copy()
for _ in range(200):robot.tick(.001);step()
end_hand=hand.get_transforms()[0].copy()
movement=(jog_end[:3]-initial_hand[:3]).astype(float)
result={"native_robot_drive_ready":bool(movement[0]>.00010 and np.linalg.norm(movement)<.001),
 "measured_movement_m":movement.tolist(),"fk_physical_position_error_mm":pe*1000,"fk_physical_orientation_error_deg":math.degrees(re),
 "gripper_width_before_m":width0,"gripper_width_after_m":width1,"gripper_drive_responded":width1<width0-.0002,
 "stop_no_further_targets":robot.commands==count,"post_stop_displacement_mm":float(np.linalg.norm(end_hand[:3]-stop_hand[:3]))*1000,
 "normal_contact_positive":bool(forces and max(f[1] for f in forces)>0),"positive_contact_peak_N":max([f[1] for f in forces],default=0),
 "runtime_joint_pose_writes":False,"joint_names":names,"robot_commands":robot.commands,
 "original_files_unchanged":all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()),
 "source_hashes":hashes,"full_mechanical_task_accepted":False,"threshold_status":CONFIG["status"]}
with (OUT/"robot_drive.csv").open("w",newline="") as f:
 w=csv.writer(f);w.writerow(["time_s","phase","hand_x_m","hand_y_m","hand_z_m",*names]);w.writerows(rows)
with (OUT/"positive_contact.csv").open("w",newline="") as f:
 w=csv.writer(f);w.writerow(["time_s","normal_force_N"]);w.writerows(forces)
print("STOP_DONE",result,flush=True)
sub=None;hand=None;sv=None;sim.detach_stage()
print("DETACHED_FREE_SPACE",flush=True)
# Build the actual user panel without Play, so no command can be sent.
omni.usd.get_context().open_stage(str(OUT/"manual_franka_scene.usda"))
for _ in range(2):app.update()
import launch_panel
print("CANDIDATE_OPENED",flush=True)
panel=launch_panel.ManualSafetyPanel(show_ui=True)
print("PANEL_CREATED",flush=True)
result["gui_panel_constructed"]=panel.ui is not None
result["auto_armed_on_load"]=panel.controller.armed
panel.close()
(OUT/"native_acceptance.json").write_text(json.dumps(result,indent=2))
print("NATIVE_RESULT",json.dumps(result),flush=True)
signal.signal(signal.SIGALRM,lambda *a:os._exit(0));signal.alarm(5);app.close()
