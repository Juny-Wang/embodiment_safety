"""Full-scene passive smoke test; no robot commands, no source scene save."""
from pathlib import Path
import os,sys,json,runpy,traceback
P=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/wire_guides_full_v001')
for n in ['cache','data','config','tmp']:(P/'runtime'/n).mkdir(parents=True,exist_ok=True)
for k,n in [('XDG_CACHE_HOME','cache'),('XDG_DATA_HOME','data'),('XDG_CONFIG_HOME','config'),('TMPDIR','tmp')]:os.environ[k]=str(P/'runtime'/n)
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'width':1000,'height':700,'extra_args':['--portable','--portable-root',str(P/'runtime'),'--/log/file='+str(P/'runs/full_smoke_repaired_kit.log')]})
report={'full_scene_passive_smoke_pass':False,'robot_commanded':False,'manual_gui_verified':False}
try:
 import numpy as np
 import omni.usd,omni.physics.tensors as tensors
 from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Gf
 from isaacsim.core.api import SimulationContext
 cfg=json.loads((P/'config.json').read_text())
 omni.usd.get_context().open_stage(cfg['scene'])
 for _ in range(5):app.update()
 stage=omni.usd.get_context().get_stage()
 # Positive control exists only in this test's anonymous session, never in saved copy.
 with Usd.EditContext(stage,stage.GetSessionLayer()):
  c=UsdGeom.Cube.Define(stage,'/MonitorPositiveControl');c.CreateSizeAttr(.02)
  UsdGeom.Xformable(c).AddTranslateOp().Set(Gf.Vec3d(.4,-.2,.7601))
  UsdPhysics.RigidBodyAPI.Apply(c.GetPrim());UsdPhysics.MassAPI.Apply(c.GetPrim()).CreateMassAttr(.1);UsdPhysics.CollisionAPI.Apply(c.GetPrim())
  from pxr import PhysxSchema
  PhysxSchema.PhysxRigidBodyAPI.Apply(c.GetPrim()).CreateSleepThresholdAttr(0.)
  PhysxSchema.PhysxContactReportAPI.Apply(c.GetPrim()).CreateThresholdAttr(0.)
  UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath('/Task2WireGuides/Materials/PrintedContact')),materialPurpose='physics')
 sim=SimulationContext(physics_dt=1/960,rendering_dt=1/60,backend='numpy',device='cpu')
 sim.initialize_physics();sim.play()
 for _ in range(10):sim.step(render=True)
 view=tensors.create_simulation_view('numpy');view.set_subspace_roots('/')
 cv=view.create_rigid_contact_view('/MonitorPositiveControl',['/World/Table/Top'],256)
 runpy.run_path(str(P/'scripts/launch_monitor.py'))
 monitor=sys.modules['_task2_full_wire_monitor_v001'].monitor
 initial=None;last=None;frames=[];box=[]
 for step in range(960):
  sim.step(render=(step%16==0))
  f=float(np.linalg.norm(cv.get_net_contact_forces(1/960)[0]));box.append(f)
  if step%120==0:
   frames.append({'step':step,'box_N':f,'labels':{str(k):v.text for k,v in monitor.labels.items()}})
   print('SAMPLE',step,json.dumps(frames[-1]),flush=True)
  if initial is None and len(monitor.views)==4:
   initial={w:a.get_transforms()[ids,:3].copy() for w,(_,a,ids,_) in monitor.views.items()}
 for _ in range(3):sim.step(render=True)
 report['samples']=frames;report['latest']=monitor.latest
 report['control_mean_last_120_N']=float(np.mean(box[-120:]))
 report['positive_control_pass']=.7<report['control_mean_last_120_N']<1.25
 report['four_wire_sensors_finite']=len(monitor.latest)==4 and all(v.get('guide_normal_N') is not None and 'segment_centers_m' in v for v in monitor.latest.values())
 report['internal_load_ready']=False
 report['monitor_scope']='Guide contact and bend only; no internal-load measurement'
 report['full_scene_passive_smoke_pass']=report['positive_control_pass'] and report['four_wire_sensors_finite']
 report['monitor_log']=str(monitor.out/'measurements.jsonl')
 report['arrows_api_available']=monitor.dd is not None
 # Neither actual robot routing nor wire material stress calibrated by this smoke.
 report['route_contact_test_passed']=False;report['material_stress_ready']=False
 monitor.close();sim.stop()
except Exception:report['error']=traceback.format_exc()
finally:
 (P/'reports/full_smoke.json').write_text(json.dumps(report,indent=2,allow_nan=False))
 print('RESULT',json.dumps({k:v for k,v in report.items() if k not in ['latest','samples']}),flush=True)
 app.close()
