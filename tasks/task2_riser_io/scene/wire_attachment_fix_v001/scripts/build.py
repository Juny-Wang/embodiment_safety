from pathlib import Path
import json,shutil,hashlib,math
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,Gf,Sdf
R=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene')
P=R/'wire_attachment_fix_v001';P.mkdir(exist_ok=False)
for d in ['scripts','reports','runs','runtime']:(P/d).mkdir()
source=R/'riser_io_wire_guides_v001.usda';target=R/'riser_io_wire_guides_attachment_candidate_v001.usda'
assert not target.exists()
shutil.copy2(source,P/'reports/source_before.usda');shutil.copy2(source,target)
s=Usd.Stage.Open(str(target));xc=UsdGeom.XformCache();changes=[]
def rot(m):
 rows=[Gf.Vec3d(*list(m[i])[:3]).GetNormalized() for i in range(3)]
 a=Gf.Matrix4d(1)
 for i in range(3):
  for j in range(3):a[i,j]=rows[i][j]
 return a.ExtractRotationQuat().GetNormalized()
for w in range(1,5):
 root=s.GetPrimAtPath(f'/wire{w}');end=s.GetPrimAtPath(f'/wire{w}/Segment_23');ring=s.GetPrimAtPath(f'/wire{w}/_0347300_ring_terminal')
 em=xc.GetLocalToWorldTransform(end);oldm=xc.GetLocalToWorldTransform(ring);parent=xc.GetLocalToWorldTransform(root)
 direction=em.TransformDir(Gf.Vec3d(1,0,0)).GetNormalized()
 y=-direction;x=Gf.Cross(y,Gf.Vec3d(0,0,1)).GetNormalized();z=Gf.Cross(x,y).GetNormalized()
 # 3 mm of straight cable and 3.5 mm end cap inside the 10 mm barrel.
 half=end.GetAttribute('height').Get()*.001/2
 attachment=Gf.Vec3d(0,.025+half,.0035)
 world=Gf.Matrix4d(1)
 for i,v in enumerate([x,y,z]):
  for j in range(3):world[i,j]=v[j]
 world.SetTranslateOnly(em.ExtractTranslation()-world.TransformDir(attachment))
 xf=UsdGeom.Xformable(ring);ops=xf.GetOrderedXformOps();assert len(ops)==1 and ops[0].GetOpType()==UsdGeom.XformOp.TypeTransform
 ops[0].Set(world*parent.GetInverse());xc.Clear()
 j=UsdPhysics.FixedJoint(s.GetPrimAtPath(f'/wire{w}/FixedJoint'))
 oldanchor=list(j.GetLocalPos0Attr().Get());j.GetLocalPos0Attr().Set(Gf.Vec3f(attachment));j.GetLocalPos1Attr().Set(Gf.Vec3f(0))
 j.GetLocalRot0Attr().Set(Gf.Quatf((rot(world).GetInverse()*rot(em)).GetNormalized()));j.GetLocalRot1Attr().Set(Gf.Quatf(1))
 j.CreateCollisionEnabledAttr(False)
 # Bonded insertion contact: disable only ring versus its attached end capsule, not other geometry.
 UsdPhysics.FilteredPairsAPI.Apply(ring).CreateFilteredPairsRel().AddTarget(end.GetPath())
 changes.append({'wire':w,'ring_before_world':str(oldm),'ring_after_world':str(world),'attachment_before_m':oldanchor,'attachment_after_m':list(attachment),'cable_barrel_axis_error_deg':math.degrees(math.acos(min(1,abs(Gf.Dot(y,direction))))),'cylinder_insertion_mm':3,'tip_insertion_mm':6.5})
 # Align ordinary joint frames at the authored initial shape. Keep maximal-coordinate model.
 for p in Usd.PrimRange(root):
  if p.IsA(UsdPhysics.RevoluteJoint):
   jj=UsdPhysics.Joint(p);a=s.GetPrimAtPath(jj.GetBody0Rel().GetTargets()[0]);b=s.GetPrimAtPath(jj.GetBody1Rel().GetTargets()[0])
   m0=xc.GetLocalToWorldTransform(a);m1=xc.GetLocalToWorldTransform(b)
   a0=m0.Transform(Gf.Vec3d(jj.GetLocalPos0Attr().Get()));a1=m1.Transform(Gf.Vec3d(jj.GetLocalPos1Attr().Get()));mid=(a0+a1)*.5
   jj.GetLocalPos0Attr().Set(Gf.Vec3f(m0.GetInverse().Transform(mid)));jj.GetLocalPos1Attr().Set(Gf.Vec3f(m1.GetInverse().Transform(mid)))
   q0=Gf.Quatd(jj.GetLocalRot0Attr().Get()).GetNormalized();jj.GetLocalRot0Attr().Set(Gf.Quatf(q0))
   jj.GetLocalRot1Attr().Set(Gf.Quatf((rot(m1).GetInverse()*rot(m0)*q0).GetNormalized()))
  if p.HasAPI(UsdPhysics.RigidBodyAPI):
   p.AddAppliedSchema('PhysxRigidBodyAPI');p.AddAppliedSchema('PhysxContactReportAPI')
   p.CreateAttribute('physxContactReport:threshold',Sdf.ValueTypeNames.Float).Set(0.)
   for key,value in [('solverPositionIterationCount',64),('solverVelocityIterationCount',16)]:
    p.CreateAttribute('physxRigidBody:'+key,Sdf.ValueTypeNames.Int).Set(value)
for p in s.Traverse():
 if p.IsA(UsdPhysics.Scene):
  p.AddAppliedSchema('PhysxSceneAPI')
  p.CreateAttribute('physxScene:timeStepsPerSecond',Sdf.ValueTypeNames.UInt).Set(240)
  p.CreateAttribute('physxScene:solverType',Sdf.ValueTypeNames.Token).Set('TGS')
  p.CreateAttribute('physxScene:maxPositionIterationCount',Sdf.ValueTypeNames.Int).Set(64)
  p.CreateAttribute('physxScene:maxVelocityIterationCount',Sdf.ValueTypeNames.Int).Set(16)
s.GetRootLayer().Save()
cfg={'source':str(source),'candidate':str(target),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'changes':changes,'physics_dt_s':1/240,'normal_gravity':True,'articulation_conversion':False,'extra_damping_added':False}
(P/'config.json').write_text(json.dumps(cfg,indent=2))
print(json.dumps(cfg,indent=2))
