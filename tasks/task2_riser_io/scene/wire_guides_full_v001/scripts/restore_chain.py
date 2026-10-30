from pathlib import Path
import json,shutil
from pxr import Usd,UsdGeom,Sdf
P=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/wire_guides_full_v001')
cfg=json.loads((P/'config.json').read_text());target=Path(cfg['scene'])
shutil.copy2(target,P/'reports/rejected_articulation_metres.usda')
shutil.copy2(P/'reports/full_smoke.json',P/'reports/rejected_articulation_metres.json')
shutil.copy2(P/'reports/copy_before_sensor_repair.usda',target)
stage=Usd.Stage.Open(str(target))
for w in range(1,5):
 root=stage.GetPrimAtPath(f'/wire{w}/Segment_00')
 # API was ignored by PhysX on a kinematic body. Removing it does not convert the original chain.
 from pxr import UsdPhysics
 root.RemoveAPI(UsdPhysics.ArticulationRootAPI)
 for p in Usd.PrimRange(stage.GetPrimAtPath(f'/wire{w}')):
  if p.IsA(UsdGeom.Capsule):
   p.AddAppliedSchema('PhysxContactReportAPI');p.CreateAttribute('physxContactReport:threshold',Sdf.ValueTypeNames.Float).Set(0.)
stage.GetRootLayer().Save()
cfg.pop('articulation_roots',None);cfg.pop('sensor_prerequisite_repair',None)
cfg['force_location']='Wire-to-guide actual normal contact only; near-end/internal joint reaction unavailable in retained maximal-coordinate chain.'
cfg['monitor_mode']='CONTACT_AND_BEND_ONLY_INTERNAL_LOAD_UNAVAILABLE'
cfg['diagnostic_articulation_promoted']=False
(P/'config.json').write_text(json.dumps(cfg,indent=2))
print('RESTORED_ORIGINAL_CHAIN_MODEL; experimental articulation rejected')
