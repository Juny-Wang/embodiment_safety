"""Run from Isaac Script Editor AFTER opening a wire_safety_v001 L0/L1/L2 stage.
Read PhysX tensors; only session-layer colors/arrows are authored. No robot control.
No internal material stress or connector-root force is claimed.
"""
from pathlib import Path
import sys,json,time,types,datetime
import numpy as np
import omni.usd,omni.kit.app,omni.timeline,omni.physx,omni.ui as ui
import omni.physics.tensors as tensors
from pxr import UsdGeom,Gf
P=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/wire_safety_v001')
if str(P/'scripts') not in sys.path:sys.path.insert(0,str(P/'scripts'))
from metrics import summarize
class Monitor:
 def __init__(self):
  self.stage=omni.usd.get_context().get_stage()
  allowed=[str(P/(n+'.usda')) for n in ['L0_sensor_bench','L1_one_guide','L2_two_guides']]
  if self.stage.GetRootLayer().realPath not in allowed:raise RuntimeError('Open an isolated L0/L1/L2 wire_safety_v001 stage first; main stage is not supported.')
  self.stage.SetEditTarget(self.stage.GetSessionLayer())
  self.timeline=omni.timeline.get_timeline_interface();self.art=None;self.dt=None;self.serial=0;self.used=-1;self.control_pass=False;self.control_samples=[];self.dd=None
  self.window=ui.Window('Task2 Wire Safety | DEMO',width=420,height=285)
  self.labels={}
  with self.window.frame:
   with ui.VStack(spacing=5):
    ui.Label('ISOLATED BENCH - warning only / no robot commands')
    for k,title in [('status','Status'),('near_end','Connector joint N'),('radius','Min bend radius mm'),('guide','Guide normal N'),('box','Box control N'),('root','Connector root / stress')]:
     with ui.HStack():
      ui.Label(title,width=190);self.labels[k]=ui.Label('unavailable')
    ui.Label('Thresholds are demonstration values, NOT damage limits.',word_wrap=True)
    ui.Button('STOP MONITOR',clicked_fn=self.close)
  self.out=P/'runs'/('gui_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%f'));self.out.mkdir()
  self.log=(self.out/'metrics.jsonl').open('w')
  self.substep=omni.physx.get_physx_interface().subscribe_physics_step_events(self.on_step)
  self.sub=omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self.update)
  try:
   from isaacsim.util.debug_draw import _debug_draw
   self.dd=_debug_draw.acquire_debug_draw_interface()
  except Exception:self.dd=None
 def on_step(self,dt):
  if dt>0:self.dt=float(dt);self.serial+=1
 def update(self,event):
  if not self.timeline.is_playing():
   self.labels['status'].text='PAUSED / no fresh measurements';self.art=None;self.control_pass=False;self.control_samples=[]
   return
  if self.dt is None or self.serial==self.used:return
  self.used=self.serial
  try:
   if self.art is None:
    self.view=tensors.create_simulation_view('numpy');self.view.set_subspace_roots('/')
    self.art=self.view.create_articulation_view('/World/Cable')
    names=list(self.art.shared_metatype.link_names);self.ids=[names.index('Segment_'+f'{i:02d}') for i in range(24)]
    self.cv=self.view.create_rigid_contact_view('/World/ControlBox',['/World/Table'],256)
    self.gv=self.view.create_rigid_contact_view('/World/Cable/Segment_*',['/World/Guides/Post_1']+(['/World/Guides/Post_2'] if self.stage.GetPrimAtPath('/World/Guides/Post_2') else []),2048) if self.stage.GetPrimAtPath('/World/Guides') else None
   p=self.art.get_link_transforms()[0,self.ids,:3].copy()
   wrench=self.art.get_link_incoming_joint_force()[0,self.ids[0],:].copy()
   if not np.isfinite(p).all() or not np.isfinite(wrench).all():
    raise RuntimeError('Nonfinite physics state; measurements invalid')
   box=float(np.linalg.norm(self.cv.get_net_contact_forces(self.dt)[0]))
   self.control_samples=(self.control_samples+[box])[-60:]
   self.control_pass=len(self.control_samples)>=30 and .7<float(np.mean(self.control_samples))<1.25
   m=summarize(p,wrench if self.control_pass else None)
   m['measurement_location']='/World/Cable/CableAttachment child frame; connector surrogate'
   m['connector_root_force_N']=None;m['control_box_N']=box;m['control_verified']=self.control_pass
   guide=None;points=[]
   if self.gv is not None:
    force,point,normal,separation,count,start=self.gv.get_contact_data(self.dt)
    guide=0.
    for ind in np.ndindex(count.shape):
     for k in range(int(start[ind]),int(start[ind]+count[ind])):
      f=float(np.asarray(force[k]).reshape(-1)[0]);guide+=abs(f)
      points.append({'position_m':point[k].tolist(),'normal':normal[k].tolist(),'normal_force_N':f,'separation_m':float(np.asarray(separation[k]).reshape(-1)[0])})
   m['guide_normal_N']=guide;m['physics_dt']=self.dt;m['time_s']=float(self.timeline.get_current_time());m['contact_points']=points
   self.labels['status'].text=m['state']+' | '+(', '.join(m['reasons']) or ('measured / no demo threshold exceeded' if self.control_pass else 'WAITING FOR BOX POSITIVE CONTROL'))
   self.labels['near_end'].text='unavailable' if m['connector_force_N'] is None else f"{m['connector_force_N']:.3f}"
   self.labels['radius'].text='straight' if m.get('straight') else ('unavailable' if m['min_bend_radius_m'] is None else f"{1000*m['min_bend_radius_m']:.1f}")
   self.labels['guide'].text='N/A (no guide)' if guide is None else f'{guide:.3f}'
   self.labels['box'].text=f'{box:.3f} / expected 0.981'
   self.labels['root'].text='unavailable / not material-calibrated'
   colors={'GRAY':(.5,.5,.5),'GREEN':(.15,.8,.25),'YELLOW':(1,.7,.05),'RED':(1,.1,.05)}
   for i in range(24):
    state=m['state']
    if self.control_pass and 0<i<23 and m['radii_m'] and m['radii_m'][i-1] is not None and m['radii_m'][i-1]<.035:state='RED'
    UsdGeom.Gprim(self.stage.GetPrimAtPath('/World/Cable/Segment_'+f'{i:02d}')).GetDisplayColorAttr().Set([Gf.Vec3f(*colors[state])])
   if self.dd:
    self.dd.clear_lines()
    if points:
     starts=[];ends=[]
     for c in points:
      v=np.asarray(c['position_m']);end=v+np.asarray(c['normal'])*np.clip(c['normal_force_N']*.025,-.05,.05)
      starts.append(tuple(v));ends.append(tuple(end))
      direction=end-v;length=np.linalg.norm(direction)
      if length>1e-8:
       direction/=length;side=np.cross(direction,[0,0,1])
       if np.linalg.norm(side)<1e-6:side=np.cross(direction,[0,1,0])
       side/=np.linalg.norm(side)
       for sign in [-1,1]:
        starts.append(tuple(end));ends.append(tuple(end-direction*min(.004,length*.35)+sign*side*min(.002,length*.175)))
     self.dd.draw_lines(starts,ends,[(1.,.35,.05,1.)]*len(starts),[2.]*len(starts))
   m['arrows_available']=self.dd is not None
   self.log.write(json.dumps(m,allow_nan=False)+'\n');self.log.flush()
  except Exception as ex:
   self.labels['status'].text='UNAVAILABLE: '+str(ex)[:120]
   for key,label in self.labels.items():
    if key!='status':label.text='unavailable'
   if self.dd:self.dd.clear_lines()
   for i in range(24):
    prim=self.stage.GetPrimAtPath('/World/Cable/Segment_'+f'{i:02d}')
    if prim:UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([Gf.Vec3f(.5,.5,.5)])
   self.art=None
 def close(self):
  self.sub=None;self.substep=None
  if self.dd:self.dd.clear_lines()
  if not self.log.closed:self.log.close()
  self.window.visible=False
name='_task2_wire_safety_monitor_v001'
if name in sys.modules and getattr(sys.modules[name],'monitor',None):
 sys.modules[name].monitor.close()
holder=types.ModuleType(name);sys.modules[name]=holder;holder.monitor=Monitor()
print('Wire monitor attached: no robot commands; warning-only; independent of Task2 keyboard panel.')
