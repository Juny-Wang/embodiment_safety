from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdGeom,UsdPhysics,Gf,Sdf
R=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io')
O=R/'scene/unsafe_demo_v009';O.mkdir(exist_ok=True)
out=R/'scene/riser_io_unsafe_demo_v009.usda'
if out.exists():out.unlink()
s=Usd.Stage.CreateNew(str(out));s.GetRootLayer().subLayerPaths=['riser_io_layout_review_v008.usda']
def attr(p,n,t,v):p.CreateAttribute(n,t).Set(v)
def phys(p,mass):
 rb=UsdPhysics.RigidBodyAPI.Apply(p);rb.CreateRigidBodyEnabledAttr(True);rb.CreateKinematicEnabledAttr(False)
 rb.CreateVelocityAttr(Gf.Vec3f(0));rb.CreateAngularVelocityAttr(Gf.Vec3f(0))
 UsdPhysics.MassAPI.Apply(p).CreateMassAttr(mass)
 p.AddAppliedSchema('PhysxRigidBodyAPI')
 for n,v in [('solverPositionIterationCount',64),('solverVelocityIterationCount',16)]:attr(p,'physxRigidBody:'+n,Sdf.ValueTypeNames.Int,v)
 for n,v in [('linearDamping',.05),('angularDamping',.05),('maxDepenetrationVelocity',.15),('sleepThreshold',0.)]:attr(p,'physxRigidBody:'+n,Sdf.ValueTypeNames.Float,v)
def jf(j,b0,b1,world,rotation):
 cache=UsdGeom.XformCache()
 for k,b in enumerate([b0,b1]):
  getattr(j,f'CreateBody{k}Rel')().SetTargets([b]);m=cache.GetLocalToWorldTransform(s.GetPrimAtPath(b))
  pos=m.GetInverse().Transform(Gf.Vec3d(*map(float,world)))
  getattr(j,f'CreateLocalPos{k}Attr')(Gf.Vec3f(*pos))
  br=np.array(m.ExtractRotationMatrix()).T
  # Extract the body rotation independently of scale.
  br=br/np.linalg.norm(br,axis=0)
  q=Rotation.from_matrix(br.T@rotation).as_quat()
  getattr(j,f'CreateLocalRot{k}Attr')(Gf.Quatf(float(q[3]),Gf.Vec3f(*q[:3])))
 j.CreateCollisionEnabledAttr(False)
# The protective sleeve is a rigid approximation attached to the connector,
# so its contact reaction follows the connector load path.
boot=s.GetPrimAtPath('/World/RootStrainRelief/Boot');phys(boot,.004)
connector='/TE_L26620011_4_connector_v2'
j=UsdPhysics.FixedJoint.Define(s,'/World/RootStrainRelief/Mount')
port=np.array(json.loads((R/'scene/layout_review_v008/layout.json').read_text())['port_center_m'])
jf(j,connector,str(boot.GetPath()),port,np.eye(3));j.CreateExcludeFromArticulationAttr(True)
# New visible geometry must participate in contact.
count=0
for p in s.Traverse():
 path=str(p.GetPath())
 if not p.IsA(UsdGeom.Mesh):continue
 if not any(path.startswith(x) for x in ['/World/Chassis/','/World/RootStrainRelief/Boot','/TE_L26620011_4_connector_v2/','/World/PickupNest','/World/TargetCableLanding','/PCB_holder/','/PCB_4_Terminal/','/NutPlate_Coarse/Nut_1/IndependentNut','/ScrewTray/SingleNest']):continue
 UsdPhysics.CollisionAPI.Apply(p).CreateCollisionEnabledAttr(True);p.AddAppliedSchema('PhysxCollisionAPI')
 attr(p,'physxCollision:contactOffset',Sdf.ValueTypeNames.Float,.00015)
 attr(p,'physxCollision:restOffset',Sdf.ValueTypeNames.Float,0.)
 owner=p
 while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):owner=owner.GetParent()
 dyn=bool(owner)
 UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr('sdf' if dyn else 'none')
 if dyn:
  p.AddAppliedSchema('PhysxSDFMeshCollisionAPI')
  attr(p,'physxSDFMeshCollision:sdfResolution',Sdf.ValueTypeNames.UInt,256)
 count+=1
# Capsules were placed for the layout; reauthor every joint at those poses.
paths=[f'/wire1/Segment_{i:02d}' for i in range(24)]
cache=UsdGeom.XformCache();points=[];rots=[]
for i,path in enumerate(paths):
 p=s.GetPrimAtPath(path);m=cache.GetLocalToWorldTransform(p);h=float(p.GetAttribute('height').Get())
 a=np.array(m.Transform(Gf.Vec3d(-h/2,0,0)));b=np.array(m.Transform(Gf.Vec3d(h/2,0,0)))
 if i==0:points.append(a)
 points.append(b);rots.append(np.array(m.ExtractRotationMatrix()).T)
for i in range(23):
 j=UsdPhysics.Joint(s.GetPrimAtPath(f'/wire1/Bend_{i:02d}'));jf(j,paths[i],paths[i+1],points[i+1],rots[i])
 j.CreateLocalRot1Attr(Gf.Quatf(1,Gf.Vec3f(0)))
ring='/wire1/_0347300_ring_terminal'
jf(UsdPhysics.FixedJoint(s.GetPrimAtPath('/wire1/FixedJoint')),paths[-1],ring,points[-1],rots[-1])
jf(UsdPhysics.FixedJoint(s.GetPrimAtPath('/World/CableSensor/Attachment')),connector,paths[0],points[0],rots[0])
jf(UsdPhysics.FixedJoint(s.GetPrimAtPath('/World/CableSensor/ConnectorMount')),'/World/CableSensor/Base',connector,port,np.eye(3))
base=s.GetPrimAtPath('/World/CableSensor/Base')
bp=UsdGeom.XformCache().GetLocalToWorldTransform(base).ExtractTranslation()
j=UsdPhysics.FixedJoint(s.GetPrimAtPath('/World/CableSensor/WorldMount'))
j.CreateLocalPos0Attr(Gf.Vec3f(*bp));j.CreateLocalPos1Attr(Gf.Vec3f(0))
j.CreateLocalRot0Attr(Gf.Quatf(1,Gf.Vec3f(0)));j.CreateLocalRot1Attr(Gf.Quatf(1,Gf.Vec3f(0)))
# Adjacent collar contact is intentional; it is solved at the actual sleeve surface.
basepairs=UsdPhysics.FilteredPairsAPI.Apply(base).CreateFilteredPairsRel()
basepairs.SetTargets([connector,paths[0],str(boot.GetPath())])
# Keep the terminal geometry byte-for-byte in its point and index arrays.
original=Usd.Stage.Open(str(R/'scene/riser_io_layout_review_v008.usda'))
a=UsdGeom.Mesh(original.GetPrimAtPath(ring+'/Mesh'));b=UsdGeom.Mesh(s.GetPrimAtPath(ring+'/Mesh'))
assert a.GetPointsAttr().Get()==b.GetPointsAttr().Get()
assert a.GetFaceVertexIndicesAttr().Get()==b.GetFaceVertexIndicesAttr().Get()
robot_xf=UsdGeom.Xformable(s.GetPrimAtPath('/World/Robot'));robot_xf.ClearXformOpOrder();robot_xf.AddTranslateOp().Set(Gf.Vec3d(-.52,-.15,.75))
s.GetRootLayer().customLayerData={'status':'UNSAFE_ACTION_DEMONSTRATION','physicalCalibration':False,'rootBootModel':'rigid_attached_sleeve','ringGeometryUnchanged':True}
s.GetRootLayer().Save()
report={'scene':str(out),'new_contact_meshes':count,'root_boot_model':'rigid sleeve fixed to connector, not a calibrated flexible boot','ring_geometry_unchanged':True,'load_path':'world mount -> base -> connector -> cable; boot reactions enter connector','cable_length_m':float(sum(np.linalg.norm(points[i+1]-points[i]) for i in range(24))),'thresholds':'demonstration boundaries only; no material damage claims'}
(O/'build_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
