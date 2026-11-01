from pathlib import Path
import json
from isaacsim import SimulationApp
app=SimulationApp({'headless':True})
import omni.usd,omni.physics.tensors as tensors
import numpy as np
from pxr import UsdGeom,UsdPhysics,Gf,Sdf
from isaacsim.core.api import SimulationContext
root=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene')
omni.usd.get_context().new_stage();s=omni.usd.get_context().get_stage()
UsdGeom.SetStageMetersPerUnit(s,1);UsdGeom.SetStageUpAxis(s,'Z')
scene=UsdPhysics.Scene.Define(s,'/PhysicsScene');scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81)
asset=root/'assets/task2_wire/usd/printable_v001/ring_terminal_surrogate.usda'
for i,(r,off) in enumerate([(.0034,0.),(.005,0.),(.005,.0004),(.006,0.)]):
 x=i*.06
 ring=UsdGeom.Xform.Define(s,f'/Ring_{i}');ring.GetPrim().GetReferences().AddReference(str(asset));ring.AddTranslateOp().Set(Gf.Vec3d(x,0,.80))
 joint=UsdPhysics.FixedJoint.Define(s,f'/Mount_{i}');joint.CreateBody1Rel().SetTargets([ring.GetPath()]);joint.CreateLocalPos0Attr(Gf.Vec3f(x,0,.80))
 c=UsdGeom.Cylinder.Define(s,f'/Probe_{i}');c.CreateRadiusAttr(r);c.CreateHeightAttr(.003);c.AddTranslateOp().Set(Gf.Vec3d(x+off,0,.825))
 UsdPhysics.RigidBodyAPI.Apply(c.GetPrim());UsdPhysics.MassAPI.Apply(c.GetPrim()).CreateMassAttr(.005);UsdPhysics.CollisionAPI.Apply(c.GetPrim())
 c.GetPrim().AddAppliedSchema('PhysxCollisionAPI');c.GetPrim().CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.00015)
sim=SimulationContext(physics_dt=1/480,rendering_dt=1/60,backend='numpy',device='cpu');sim.initialize_physics();sim.play()
for _ in range(4):sim.step(render=False)
v=tensors.create_simulation_view('numpy');v.set_subspace_roots('/');b=v.create_rigid_body_view('/Probe_*')
for _ in range(240):sim.step(render=False)
positions=b.get_transforms()[:,:3].tolist()
r={'probe_radii_m':[.0034,.005,.005,.006],'offset_m':[0,0,.0004,0.],'positions':positions,'passed_open_hole':[x[2]<.78 for x in positions]}
(root/'single_wire_fastening_v003/hole_probe.json').write_text(json.dumps(r,indent=2))
print(r,flush=True);app.close()
