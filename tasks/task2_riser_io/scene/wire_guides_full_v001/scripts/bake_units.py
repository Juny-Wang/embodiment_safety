from pathlib import Path
import json,shutil
from pxr import Usd,UsdGeom,UsdPhysics,Gf
P=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/wire_guides_full_v001')
cfg=json.loads((P/'config.json').read_text());path=Path(cfg['scene'])
shutil.copy2(path,P/'reports/copy_before_unit_bake.usda')
shutil.copy2(P/'reports/full_smoke.json',P/'reports/full_smoke_before_unit_bake.json')
stage=Usd.Stage.Open(str(path));xc=UsdGeom.XformCache()
bodyworld={};jointworld={};scales={}
for w in range(1,5):
 for p in Usd.PrimRange(stage.GetPrimAtPath(f'/wire{w}')):
  if p.HasAPI(UsdPhysics.RigidBodyAPI):bodyworld[str(p.GetPath())]=xc.GetLocalToWorldTransform(p)
for p in stage.Traverse():
 if not p.IsA(UsdPhysics.Joint):continue
 j=UsdPhysics.Joint(p);targets=[j.GetBody0Rel().GetTargets(),j.GetBody1Rel().GetTargets()]
 if not any(str(t[0]) in bodyworld for t in targets if t):continue
 jointworld[str(p.GetPath())]=[(xc.GetLocalToWorldTransform(stage.GetPrimAtPath(ts[0])).Transform(Gf.Vec3d(a.Get())) if ts else Gf.Vec3d(a.Get())) for ts,a in zip(targets,[j.GetLocalPos0Attr(),j.GetLocalPos1Attr()])]
for w in range(1,5):UsdGeom.Xformable(stage.GetPrimAtPath(f'/wire{w}')).ClearXformOpOrder()
for path,m in bodyworld.items():
 p=stage.GetPrimAtPath(path);xf=UsdGeom.Xformable(p);xf.ClearXformOpOrder()
 transform=Gf.Transform(m)
 if p.IsA(UsdGeom.Capsule):
  scale=transform.GetScale();assert max(scale)-min(scale)<1e-8
  scales[path]=float(scale[0])
  cap=UsdGeom.Capsule(p);cap.GetRadiusAttr().Set(cap.GetRadiusAttr().Get()*scale[0]);cap.GetHeightAttr().Set(cap.GetHeightAttr().Get()*scale[0])
  r=cap.GetRadiusAttr().Get();half=cap.GetHeightAttr().Get()/2+r
  cap.GetExtentAttr().Set([Gf.Vec3f(-half,-r,-r),Gf.Vec3f(half,r,r)])
  transform.SetScale(Gf.Vec3d(1));m=transform.GetMatrix()
 xf.AddTransformOp(opSuffix='metres').Set(m)
xc.Clear()
for path,points in jointworld.items():
 j=UsdPhysics.Joint(stage.GetPrimAtPath(path))
 for targets,a,point in zip([j.GetBody0Rel().GetTargets(),j.GetBody1Rel().GetTargets()],[j.GetLocalPos0Attr(),j.GetLocalPos1Attr()],points):
  local=xc.GetLocalToWorldTransform(stage.GetPrimAtPath(targets[0])).GetInverse().Transform(point) if targets else point
  a.Set(Gf.Vec3f(local))
stage.GetRootLayer().Save()
(P/'reports/unit_bake.json').write_text(json.dumps({'capsule_baked_scales':scales,'preserves_world_geometry':True,'purpose':'Remove mm-scale parent from articulation; explicit metre shapes and joint anchors.'},indent=2))
print('BAKED_METRE_GEOMETRY',len(scales))
