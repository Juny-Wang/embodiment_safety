"""Full-scene wire monitor. Run manually in Isaac Script Editor.
Reads actual PhysX rigid-body poses and guide contact forces. Session-layer feedback only.
Internal joint reaction and material stress are unavailable. Never commands Franka.
"""
from pathlib import Path
import sys,json,types,datetime
import numpy as np
import omni.usd,omni.physx,omni.kit.app,omni.timeline,omni.ui as ui
import omni.physics.tensors as tensors
from pxr import Usd,UsdGeom,Gf
P=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/wire_guides_full_v001')
# Load exact package-local metrics, avoiding modules with the same name.
ns={};exec(compile((P/'scripts/metrics.py').read_text(),str(P/'scripts/metrics.py'),'exec'),ns);evaluate=ns['evaluate']
CFG=json.loads((P/'config.json').read_text())
COLORS={'GREEN':(.12,.8,.25),'YELLOW':(1.,.7,.05),'RED':(1.,.08,.04),'GRAY':(.5,.5,.5)}
class Monitor:
 def __init__(self):
  self.stage=omni.usd.get_context().get_stage()
  if not self.stage or self.stage.GetRootLayer().realPath!=CFG['scene']:raise RuntimeError('Open '+CFG['scene'])
  self.timeline=omni.timeline.get_timeline_interface()
  self.views={};self.latest={};self.serial=0;self.used=-1;self.dt=None;self.dd=None;self.saved_colors={}
  self.out=P/'runs'/('gui_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%f'));self.out.mkdir(parents=True)
  self.log=(self.out/'measurements.jsonl').open('w')
  self.window=ui.Window('Task2 | Wire load (DEMO)',width=640,height=255);self.labels={}
  with self.window.frame:
   with ui.VStack(spacing=5):
    ui.Label('CONTACT + BEND ONLY | internal load / stress UNAVAILABLE',word_wrap=True)
    ui.Label('Wire | Near-end N | Guide N | Bend radius mm | State')
    for w in range(1,5):self.labels[w]=ui.Label(f'wire{w} | waiting for Play')
    ui.Label('Yellow: 70%; red: guide >=1 N or bend radius <35 mm.',word_wrap=True)
    ui.Label('Demo thresholds. No fracture prediction. Robot controls unchanged.',word_wrap=True)
    ui.Button('STOP MONITOR',clicked_fn=self.close)
  self.substep=omni.physx.get_physx_interface().subscribe_physics_step_events(self.on_step)
  self.sub=omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self.update)
  try:
   from isaacsim.util.debug_draw import _debug_draw
   self.dd=_debug_draw.acquire_debug_draw_interface()
  except Exception:pass
 def on_step(self,dt):
  if dt>0:self.dt=float(dt);self.serial+=1
 def color(self,w,state):
  with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
   for i in range(24):
    path=f'/wire{w}/Segment_{i:02d}';p=self.stage.GetPrimAtPath(path)
    if p:
     a=UsdGeom.Gprim(p).GetDisplayColorAttr()
     if path not in self.saved_colors:self.saved_colors[path]=a.Get()
     a.Set([Gf.Vec3f(*COLORS[state])])
 def update(self,event):
  if omni.usd.get_context().get_stage()!=self.stage:self.close();return
  if not self.timeline.is_playing():
   self.views={};self.latest={}
   for w,label in self.labels.items():
    label.text=f'wire{w} | PAUSED / no fresh measurement';self.color(w,'GRAY')
   if self.dd:self.dd.clear_lines()
   return
  if self.dt is None or self.serial==self.used:return
  self.used=self.serial;points=[];result={'time_s':float(self.timeline.get_current_time()),'physics_dt_s':self.dt,'wires':{}}
  for w in range(1,5):
   try:
    if w not in self.views:
     view=tensors.create_simulation_view('numpy');view.set_subspace_roots('/')
     art=view.create_rigid_body_view(f'/wire{w}/Segment_*')
     if art.count!=24:raise RuntimeError('24 wire rigid bodies not available')
     names=list(art.prim_paths)
     ids=[names.index(f'/wire{w}/Segment_{i:02d}') for i in range(24)]
     cv=view.create_rigid_contact_view(f'/wire{w}/Segment_*',CFG['guide_paths'],2048)
     self.views[w]=(view,art,ids,cv)
    view,art,ids,cv=self.views[w]
    positions=art.get_transforms()[ids,:3].copy()
    wrench=None
    total=None;localpoints=[];contact_error=None
    try:
     ff,pp,nn,ss,cc,starts=cv.get_contact_data(self.dt);total=0.
     for ind in np.ndindex(cc.shape):
      for k in range(int(starts[ind]),int(starts[ind]+cc[ind])):
       f=abs(float(np.asarray(ff[k]).reshape(-1)[0]));total+=f
       localpoints.append({'position_m':pp[k].tolist(),'normal':nn[k].tolist(),'normal_force_N':f,'separation_m':float(np.asarray(ss[k]).reshape(-1)[0]),'wire':w})
     if not np.isfinite(total):raise RuntimeError('nonfinite guide contact force')
    except Exception as e:total=None;localpoints=[];contact_error=str(e)
    m=evaluate(positions,wrench,total,CFG['thresholds'])
    m.update({'measurement_joint':None,'measurement_frame':None,'wrench_N_Nm':None,'segment_centers_m':positions.tolist(),'contact_points':localpoints,'contact_error':contact_error})
    if contact_error:m['state']='GRAY';m['reasons'].append('GUIDE_SENSOR_UNAVAILABLE')
    self.latest[w]=m;result['wires'][str(w)]=m;points+=localpoints
    radius='straight' if m['min_bend_radius_m'] is None else f"{m['min_bend_radius_m']*1000:.1f}"
    guide='N/A' if total is None else f'{total:.3f}'
    self.labels[w].text=f"wire{w} | N/A | {guide} N | {radius} mm | {m['state']}"
    self.color(w,m['state'])
   except Exception as e:
    self.views.pop(w,None);self.latest[w]={'state':'GRAY','error':str(e),'stress_Pa':None}
    result['wires'][str(w)]=self.latest[w];self.labels[w].text=f'wire{w} | UNAVAILABLE: {str(e)[:80]}';self.color(w,'GRAY')
  if self.dd:
   self.dd.clear_lines()
   starts=[];ends=[]
   for p in points:
    x=np.array(p['position_m']);normal=np.array(p['normal']);delta=normal*min(.05,.025*p['normal_force_N']);end=x+delta
    length=np.linalg.norm(delta)
    if length<1e-8:continue
    starts.append(tuple(x));ends.append(tuple(end));u=delta/length;v=np.cross(u,[0,0,1])
    if np.linalg.norm(v)<1e-6:v=np.cross(u,[0,1,0])
    v/=np.linalg.norm(v)
    for sign in [-1,1]:starts.append(tuple(end));ends.append(tuple(end-u*min(.004,length*.35)+sign*v*min(.002,length*.175)))
   if starts:self.dd.draw_lines(starts,ends,[(1.,.3,.02,1.)]*len(starts),[2.]*len(starts))
  self.log.write(json.dumps(result,allow_nan=False)+'\n');self.log.flush()
 def close(self):
  self.sub=None;self.substep=None;self.views={}
  if self.dd:self.dd.clear_lines()
  if omni.usd.get_context().get_stage()==self.stage:
   with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
    for path,value in self.saved_colors.items():
     p=self.stage.GetPrimAtPath(path)
     if p and value is not None:UsdGeom.Gprim(p).GetDisplayColorAttr().Set(value)
  if not self.log.closed:self.log.close()
  self.window.visible=False
key='_task2_full_wire_monitor_v001'
if key in sys.modules and getattr(sys.modules[key],'monitor',None):sys.modules[key].monitor.close()
holder=types.ModuleType(key);sys.modules[key]=holder;holder.monitor=Monitor()
print('Full Task2 wire monitor active: warning only, no Franka command.')
