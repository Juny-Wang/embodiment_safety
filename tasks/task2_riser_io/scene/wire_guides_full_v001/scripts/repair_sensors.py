"""Repair copy-only sensor prerequisites, preserving every initial mesh pose."""
from pathlib import Path
import json,shutil,hashlib
from pxr import Usd,UsdGeom,UsdPhysics,Gf,Sdf
P=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/wire_guides_full_v001')
cfg=json.loads((P/'config.json').read_text());path=Path(cfg['scene'])
backup=P/'reports/copy_before_sensor_repair.usda'
assert not backup.exists()
shutil.copy2(path,backup)
stage=Usd.Stage.Open(str(path));xc=UsdGeom.XformCache();changes=[];roots=[]
for w in range(1,5):
 root=f'/wire{w}';body=stage.GetPrimAtPath(root+'/Segment_00');matrix=xc.GetLocalToWorldTransform(body)
 body.RemoveAPI(UsdPhysics.ArticulationRootAPI)
 UsdPhysics.RigidBodyAPI(body).CreateKinematicEnabledAttr(False)
 if body.HasAttribute('physxRigidBody:kinematicEnabled'):body.GetAttribute('physxRigidBody:kinematicEnabled').Set(False)
 # A real world attachment replaces the invalid kinematic articulation root.
 path=f'/Task2WireAnchors/Wire{w}'
 anchor=UsdPhysics.FixedJoint.Define(stage,path)
 anchor.CreateBody1Rel().SetTargets([body.GetPath()])
 anchor.CreateLocalPos0Attr(Gf.Vec3f(*matrix.ExtractTranslation()))
 anchor.CreateLocalPos1Attr(Gf.Vec3f(0))
 q=Gf.Transform(matrix).GetRotation().GetQuat().GetNormalized()
 anchor.CreateLocalRot0Attr(Gf.Quatf(q));anchor.CreateLocalRot1Attr(Gf.Quatf(1))
 UsdPhysics.ArticulationRootAPI.Apply(anchor.GetPrim())
 anchor.GetPrim().AddAppliedSchema('PhysxArticulationAPI')
 for key,value in [('solverPositionIterationCount',64),('solverVelocityIterationCount',16)]:
  anchor.GetPrim().CreateAttribute('physxArticulation:'+key,Sdf.ValueTypeNames.Int).Set(value)
 anchor.GetPrim().CreateAttribute('physxArticulation:enabledSelfCollisions',Sdf.ValueTypeNames.Bool).Set(False)
 roots.append(path)
 for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
  if p.IsA(UsdGeom.Capsule):
   p.AddAppliedSchema('PhysxContactReportAPI');p.CreateAttribute('physxContactReport:threshold',Sdf.ValueTypeNames.Float).Set(0.)
  if not p.IsA(UsdPhysics.Joint):continue
  j=UsdPhysics.Joint(p);b0=j.GetBody0Rel().GetTargets();b1=j.GetBody1Rel().GetTargets()
  if not b0 or not b1:continue
  m0=xc.GetLocalToWorldTransform(stage.GetPrimAtPath(b0[0]));m1=xc.GetLocalToWorldTransform(stage.GetPrimAtPath(b1[0]))
  a=m0.Transform(Gf.Vec3d(j.GetLocalPos0Attr().Get()));b=m1.Transform(Gf.Vec3d(j.GetLocalPos1Attr().Get()));mid=(a+b)*.5
  q0=Gf.Transform(m0).GetRotation().GetQuat().GetNormalized();q1=Gf.Transform(m1).GetRotation().GetQuat().GetNormalized()
  jq0=Gf.Quatd(j.GetLocalRot0Attr().Get()).GetNormalized()
  old=j.GetLocalRot1Attr().Get()
  new=(q1.GetInverse()*q0*jq0).GetNormalized()
  j.GetLocalPos0Attr().Set(Gf.Vec3f(m0.GetInverse().Transform(mid)));j.GetLocalPos1Attr().Set(Gf.Vec3f(m1.GetInverse().Transform(mid)))
  j.GetLocalRot1Attr().Set(Gf.Quatf(new))
  changes.append({'prim':str(p.GetPath()),'anchor_error_before_mm':(a-b).GetLength()*1000,'maximum_anchor_shift_mm':(a-b).GetLength()*500,'localRot1_before':str(old),'localRot1_after':str(new),'initial_mesh_pose_changed':False})
cfg['articulation_roots']=roots
cfg['sensor_prerequisite_repair']='Copy only: dynamic Segment_00 + real fixed anchor; joint frames rebased to initial poses. Rest condition changed, not material-calibrated.'
stage.GetRootLayer().Save();(P/'config.json').write_text(json.dumps(cfg,indent=2))
(P/'reports/sensor_prerequisite_repair.json').write_text(json.dumps({'changes':changes,'roots':roots,'all_initial_mesh_poses_preserved':True,'note':cfg['sensor_prerequisite_repair']},indent=2))
print('SENSOR_REPAIRS',len(changes),roots)
