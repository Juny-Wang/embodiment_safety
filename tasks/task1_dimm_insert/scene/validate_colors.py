"""Actual force-ramp on standard boxes; not DIMM damage or task acceptance."""
from pathlib import Path
import os,sys,json,csv,signal
OUT=Path("/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_manual_safety_v002")
for k,n in [("XDG_CACHE_HOME","cache"),("XDG_CONFIG_HOME","config"),("XDG_DATA_HOME","data"),("TMPDIR","tmp")]:os.environ[k]=str(OUT/"runtime"/n)
sys.argv=sys.argv[:1]+["--portable-root",str(OUT/"runtime")]
from isaacsim import SimulationApp
app=SimulationApp({"headless":True,"create_new_stage":False,"enable_crashreporter":False,"limit_cpu_threads":8,
 "extra_args":["--/app/settings/persistent=false","--/app/settings/loadUserConfig=false",f"--/log/file={OUT}/colors_native.log"]})
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema,UsdUtils,Gf,PhysicsSchemaTools
import numpy as np,omni.physx,omni.physics.tensors as tensors
sys.path.insert(0,str(OUT));from safety_core import SafetyEngine
st=Usd.Stage.CreateNew(str(OUT/"standard_color_force_bench.usda"));UsdGeom.SetStageMetersPerUnit(st,1);UsdGeom.SetStageUpAxis(st,"Z")
UsdGeom.Xform.Define(st,"/World")
ph=UsdPhysics.Scene.Define(st,"/World/physicsScene");ph.CreateGravityMagnitudeAttr(9.81);ph.CreateGravityDirectionAttr((0,0,-1))
ps=PhysxSchema.PhysxSceneAPI.Apply(ph.GetPrim());ps.CreateTimeStepsPerSecondAttr(1000);ps.CreateSolverTypeAttr("TGS")
mat=UsdPhysics.MaterialAPI.Apply(st.DefinePrim("/World/TestMaterial","Material"));mat.CreateStaticFrictionAttr(.3);mat.CreateDynamicFrictionAttr(.25);mat.CreateRestitutionAttr(0)
for name,z,dynamic in [("Floor",.005,False),("Box",.018,True)]:
 p=UsdGeom.Cube.Define(st,"/World/"+name);p.CreateSizeAttr(.01);p.AddTranslateOp().Set((0,0,z))
 UsdPhysics.CollisionAPI.Apply(p.GetPrim());UsdShade.MaterialBindingAPI.Apply(p.GetPrim()).Bind(UsdShade.Material(mat.GetPrim()),materialPurpose="physics")
 if dynamic:
  UsdPhysics.RigidBodyAPI.Apply(p.GetPrim());UsdPhysics.MassAPI.Apply(p.GetPrim()).CreateMassAttr(.02)
  PhysxSchema.PhysxContactReportAPI.Apply(p.GetPrim()).CreateThresholdAttr(0)
st.GetRootLayer().Save()
cfg={"status":"STANDARD_BOX_DEMONSTRATION_ONLY","rules":[{"key":"normal_N","label":"Box normal contact","unit":"N","limit":3.,"steps":[0]}],
 "warning_ratio":.8,"warning_clear_ratio":.7,"contact_gate_N":.05}
engine=SafetyEngine(cfg);sim=omni.physx.get_physx_simulation_interface();omni.physx.get_physx_interface().set_thread_count(0)
frame=[]
def contact(headers,data):
 for h in headers:
  for i in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
   c=data[i];j=np.array(c.impulse);F=np.array(c.normal)*float(j)/.001 if j.ndim==0 else j/.001
   frame.append(abs(float(np.dot(F,c.normal))))
sub=sim.subscribe_contact_report_events(contact);sid=UsdUtils.StageCache.Get().Insert(st).ToLongInt();sim.attach_stage(sid)
sim.simulate(.001,0);sim.fetch_results()
sv=tensors.create_simulation_view("numpy",sid);box=sv.create_rigid_body_view("/World/Box")
idx=PhysicsSchemaTools.sdfPathToInt("/World/Box");rows=[];counter=1
for load in [0.,1.5,2.5,3.5,0.]:
 for i in range(200):
  frame.clear();sim.wake_up(sid,idx)
  if load:
   p=box.get_transforms()[0,:3]
   sim.apply_force_at_pos(sid,idx,Gf.Vec3f(0,0,-load),Gf.Vec3f(*map(float,p)),"Force")
  sim.simulate(.001,counter*.001);sim.fetch_results();counter+=1
  force=sum(frame);result=engine.evaluate({"normal_N":force},0,.001);r=result["rows"][0]
  display="RED" if result["stop"] else r["color"]
  rows.append([counter*.001,load,force,r["color"],display,result["stop"],r["peak"]])
with (OUT/"actual_force_colors.csv").open("w",newline="") as f:
 w=csv.writer(f);w.writerow(["time_s","applied_downward_N","measured_normal_N","raw_color","display_color","latched_stop","peak_N"]);w.writerows(rows)
seen=sorted(set(r[3] for r in rows))
result={"actual_contact_colors_seen":seen,"green_yellow_red_from_real_contact":all(x in seen for x in ["GREEN","YELLOW","RED"]),
 "unloaded_stop_remains_latched":bool(rows[-1][5]),"unloaded_raw_color":rows[-1][3],
 "unloaded_display_color":rows[-1][4],"peak_normal_N":max(r[2] for r in rows),
 "physics_material":"/World/TestMaterial","static_friction":.3,"dynamic_friction":.25,"restitution":0,
 "friction_response_validated":False,"threshold_N":3,"threshold_status":"STANDARD_BOX_DEMONSTRATION_ONLY",
 "full_robot_task_accepted":False}
sub=None;sim.detach_stage();(OUT/"color_acceptance.json").write_text(json.dumps(result,indent=2))
print("COLOR_RESULT",json.dumps(result),flush=True)
signal.signal(signal.SIGALRM,lambda *a:os._exit(0));signal.alarm(5);app.close()
