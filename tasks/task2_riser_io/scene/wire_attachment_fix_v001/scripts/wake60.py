from pathlib import Path
import sys,os,json,csv,traceback
P=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/wire_attachment_fix_v001')
mode=sys.argv[1];sys.argv=sys.argv[:1];cfg=json.loads((P/'config.json').read_text())
for n in ['cache','config','data','tmp']:(P/'runtime'/n).mkdir(exist_ok=True)
for k,n in [('XDG_CACHE_HOME','cache'),('XDG_CONFIG_HOME','config'),('XDG_DATA_HOME','data'),('TMPDIR','tmp')]:os.environ[k]=str(P/'runtime'/n)
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'width':800,'height':600,'extra_args':['--portable','--portable-root',str(P/'runtime'),'--/log/file='+str(P/'runs'/('kit_'+mode+'.log'))]})
report={'mode':mode,'pass':False}
try:
 import numpy as np
 import omni.usd,omni.physics.tensors as tensors
 from pxr import Gf
 from isaacsim.core.api import SimulationContext
 omni.usd.get_context().open_stage(cfg['source'] if mode=='baseline' else cfg['candidate'])
 for _ in range(5):app.update()
 sim=SimulationContext(physics_dt=1/60,rendering_dt=1/60,backend='numpy',device='cpu')
 sim.initialize_physics();sim.play()
 for _ in range(4):sim.step(render=True)
 view=tensors.create_simulation_view('numpy');view.set_subspace_roots('/')
 views={}
 for w in range(1,5):
  wire=view.create_rigid_body_view(f'/wire{w}/Segment_*');paths=list(wire.prim_paths);ids=[paths.index(f'/wire{w}/Segment_{i:02d}') for i in range(24)]
  ring=view.create_rigid_body_view(f'/wire{w}/_0347300_ring_terminal');assert ring.count==1
  views[w]=(wire,ring,ids)
 def rotate(p,v):return np.array(Gf.Rotation(Gf.Quatd(float(p[6]),Gf.Vec3d(*[float(a) for a in p[3:6]]))).TransformDir(Gf.Vec3d(*v)))
 rows=[];poses={w:[] for w in range(1,5)}
 for step in range(1200):
  if mode=='wake60':
   for wire,ring,ids in views.values():
    f=np.array([[0.,0.,.1 if 600<=step<660 else 0.]],dtype=np.float32)
    ring.apply_forces_and_torques_at_position(f,None,None,np.array([0],dtype=np.uint32),True)
  sim.step(render=step%30==0)
  t=(step+1)/60
  for w,(wire,ring,ids) in views.items():
   pos=wire.get_transforms()[ids].copy();rp=ring.get_transforms()[0].copy();vel=wire.get_velocities()[ids,:3].copy();rv=ring.get_velocities()[0,:3].copy()
   if not all(np.isfinite(a).all() for a in [pos,rp,vel,rv]):raise ValueError('Nonfinite physics')
   end=pos[-1];axis=rotate(end,[1,0,0]);boreaxis=rotate(rp,[0,1,0]);bore=rp[:3]+rotate(rp,[0,.023,.0035]);delta=bore-end[:3]
   lateral=np.linalg.norm(delta-axis*np.dot(delta,axis))*1000
   angle=np.degrees(np.arccos(np.clip(abs(np.dot(axis,boreaxis)),0,1)))
   rows.append({'t':t,'wire':w,'inlet_lateral_mm':float(lateral),'inlet_axis_deg':float(angle),'max_wire_speed_mm_s':float(np.linalg.norm(vel,axis=1).max()*1000),'rms_wire_speed_mm_s':float(np.sqrt(np.mean(vel**2))*1000),'ring_speed_mm_s':float(np.linalg.norm(rv)*1000),'ring_x':float(rp[0]),'ring_y':float(rp[1]),'ring_z':float(rp[2])})
   if t>17:poses[w].append(pos[:,:3].copy())
  if step%120==0:print('STEP',mode,step,rows[-1],flush=True)
 with (P/'runs'/(mode+'.csv')).open('w') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 result={}
 for w in views:
  rs=[r for r in rows if r['wire']==w and r['t']>17];ps=np.array(poses[w])
  result[w]={'inlet_lateral_max_mm':max(r['inlet_lateral_mm'] for r in rs),'inlet_axis_max_deg':max(r['inlet_axis_deg'] for r in rs),'wire_speed_p95_mm_s':float(np.percentile([r['max_wire_speed_mm_s'] for r in rs],95)),'ring_speed_p95_mm_s':float(np.percentile([r['ring_speed_mm_s'] for r in rs],95)),'wire_position_range_max_mm':float(np.linalg.norm(np.ptp(ps,axis=0),axis=1).max()*1000)}
 for w in views:
  rs=[r for r in rows if r['wire']==w];ref=min(rs,key=lambda r:abs(r['t']-9.9));xyz=lambda r:np.array([r['ring_x'],r['ring_y'],r['ring_z']])
  result[w]['wake_displacement_mm']=max(float(np.linalg.norm(xyz(r)-xyz(ref))*1000) for r in rs if 10<=r['t']<=12)
 report.update({'wires':result,'pass':all(v['inlet_lateral_max_mm']<.5 and v['inlet_axis_max_deg']<1 and v['wire_speed_p95_mm_s']<1 and v['wire_position_range_max_mm']<.5 for v in result.values()),'duration_s':20,'steady_window_s':[17,20],'no_body_pose_or_velocity_writes':True,'normal_gravity':True,'robot_commanded':False})
 sim.stop()
except Exception:report['error']=traceback.format_exc()
finally:
 (P/'reports'/(mode+'.json')).write_text(json.dumps(report,indent=2));print('RESULT',json.dumps(report),flush=True);app.close()
