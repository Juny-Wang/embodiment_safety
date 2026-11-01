from pathlib import Path
import json,sys
from isaacsim import SimulationApp
app=SimulationApp({"headless":True})
import omni.usd,omni.physics.tensors as tensors
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import UsdGeom,UsdPhysics,Gf,PhysxSchema
from isaacsim.core.api import SimulationContext
R=Path("/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene")
omni.usd.get_context().open_stage(str(R/"riser_io_single_wire_fastening_v004.usda"))
stage=omni.usd.get_context().get_stage();stage.SetEditTarget(stage.GetSessionLayer())
for prim in stage.GetPseudoRoot().GetChildren():
 if prim.GetName() not in ["World","NutPlate_Coarse","ScrewTray"]:prim.SetActive(False)
for prim in stage.GetPrimAtPath("/World").GetChildren():
 if prim.GetName()!="physicsScene":prim.SetActive(False)
for i in [2,3,4]:stage.GetPrimAtPath("/NutPlate_Coarse/Nut_"+str(i)).SetActive(False)
# Tray nests are static fixtures at a different location.
sp="/ScrewTray/Nest_1/Screw_1"
x=UsdGeom.Xformable(stage.GetPrimAtPath(sp));x.ClearXformOpOrder();x.AddTranslateOp().Set(Gf.Vec3d(.168,.119,.016));x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1,Gf.Vec3d(0)))
j=UsdPhysics.Joint.Define(stage,"/World/ProbeSpindle");j.CreateBody1Rel().SetTargets([sp]);j.CreateLocalPos0Attr(Gf.Vec3f(-.092,-.071,.766));j.CreateLocalPos1Attr(Gf.Vec3f(0))
j.CreateLocalRot0Attr(Gf.Quatf(.70710678,Gf.Vec3f(0,-.70710678,0)));j.CreateLocalRot1Attr(Gf.Quatf(.70710678,Gf.Vec3f(0,-.70710678,0)))
for dof in ["transY","transZ","rotY","rotZ"]:
 a=UsdPhysics.LimitAPI.Apply(j.GetPrim(),dof);a.CreateLowAttr(1);a.CreateHighAttr(-1)
drive=UsdPhysics.DriveAPI.Apply(j.GetPrim(),"rotX");drive.CreateTypeAttr("force");drive.CreateStiffnessAttr(0);drive.CreateDampingAttr(0.);drive.CreateMaxForceAttr(.01);drive.CreateTargetVelocityAttr(-45)
from pxr import Usd
for prim in list(Usd.PrimRange(stage.GetPrimAtPath(sp))):
 if prim.GetName().startswith("Thread_") or prim.GetName() in ["Core","Tip","HeadCollider"]:prim.SetActive(False)
UsdPhysics.CollisionAPI(stage.GetPrimAtPath(sp+"/Mesh")).CreateCollisionEnabledAttr(True)
stage.GetPrimAtPath("/World/physicsScene").GetAttribute("physxScene:enableGPUDynamics").Set(True)
sim=SimulationContext(physics_dt=1/480,rendering_dt=1/60,backend="torch",device="cuda:0");sim.initialize_physics();sim.play()
import torch
t=lambda a:torch.as_tensor(a,device="cuda:0")
view=tensors.create_simulation_view("torch");view.set_subspace_roots("/")
v=view.create_rigid_body_view(sp);ix=torch.tensor([0],dtype=torch.int32,device="cuda:0")
v.set_transforms(t(np.array([[-.092,-.071,.766,0,0,0,1]],np.float32)),ix);v.set_velocities(t(np.zeros((1,6),np.float32)),ix)
contact=view.create_rigid_contact_view(sp,["/NutPlate_Coarse/Nut_1/Mesh"],8192)
rows=[];yaw=0.;lastyaw=0.
for k in range(480*15):
 pose=v.get_transforms()[0].cpu().numpy();vel=v.get_velocities()[0].cpu().numpy();rot=Rotation.from_quat(pose[3:]);rv=rot.as_rotvec()
 phase=k//(480*3)
 desired_speed=0 if phase==0 else -.8
 force=np.array([0,0,-.3],np.float32)
 axis=rot.apply([0,0,1]);torque=np.array([0.,0.,-.001])
 v.apply_forces_and_torques_at_position(t(force[None]),t(np.asarray(torque,np.float32)[None]),None,ix,True)
 sim.step(render=False)
 pyaw=float(rot.as_euler("xyz")[2]);yaw+=float((pyaw-lastyaw+np.pi)%(2*np.pi)-np.pi);lastyaw=pyaw
 if k%24==0:
  f,p,n,d,c,st=[a.cpu().numpy() for a in contact.get_contact_data(1/480)];number=int(c.sum());minimum=float(d[:number].min()) if number else 0.
  rows.append({"t":k/480,"z":float(pose[2]),"xy":pose[:2].tolist(),"yaw":yaw,"tilt":float(np.degrees(np.arccos(np.clip(axis[2],-1,1)))),"min_sep":minimum,"applied_torque":float(torque[2])})
  if k%480==0:print(rows[-1],flush=True)
(R/"single_wire_fastening_v004/thread_gpu_probe.json").write_text(json.dumps(rows,indent=2))
app.close()
