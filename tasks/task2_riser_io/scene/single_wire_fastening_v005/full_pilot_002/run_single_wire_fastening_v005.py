"""Physical Franka single-wire replay; no task-body pose writes after reset."""
from pathlib import Path
import argparse,sys,json,traceback,time,csv,os,math,hashlib,shutil
P=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene');O=P/'single_wire_fastening_v005'
pa=argparse.ArgumentParser();pa.add_argument('--name',default='attempt_001');pa.add_argument('--settle-only',action='store_true');pa.add_argument('--grasp-z',type=float,default=.007);pa.add_argument('--grip',type=float,default=.0);pa.add_argument('--probe',choices=['none','pull','bend','bend_pull','screw_tilt','screw_overload'],default='none');pa.add_argument('--screw-force',type=float,default=1.);pa.add_argument('--entry-only',action='store_true');pa.add_argument('--stop-after-probe',action='store_true');pa.add_argument('--max-turns',type=int,default=20);pa.add_argument('--screw-only',action='store_true');pa.add_argument('--record',action='store_true');pa.add_argument('--wire-only',action='store_true');args=pa.parse_args()
if args.entry_only:args.screw_only=True
out=O/args.name;out.mkdir(parents=True,exist_ok=True)
if (out/'result.json').exists():raise RuntimeError('Refusing to overwrite completed run: '+str(out))
report={'task_success':False,'physics_valid':False,'safety_status':'UNKNOWN','no_task_pose_writes_during_rollout':True,'ring_geometry_unchanged':True,'damage_status':'NOT_CALIBRATED','threshold_status':'CANDIDATE_OPERATING_ENVELOPE','scope':'one_wire_and_its_terminal_screw'};app=None;records=[];events=[];video_writer=None;frame_log=[]
class TaskFailure(RuntimeError):pass
try:
 build=json.loads((O/'build_report.json').read_text());report['ring_geometry_unchanged']=build['ring_mesh_unchanged'];report['source_sha256']=hashlib.sha256((P/'riser_io_wire_guides_v001.usda').read_bytes()).hexdigest();assert report['source_sha256']==build['source_sha256']
 report['scene_sha256']=hashlib.sha256((P/'riser_io_single_wire_fastening_v005.usda').read_bytes()).hexdigest();report['probe']=args.probe
 if args.wire_only:report['scope']='wire_only_demonstration'
 for src in [Path(__file__),Path(__file__).with_name('safety_monitor_v005.py'),Path(__file__).with_name('build_single_wire_fastening_v005.py'),O/'build_report.json']:shutil.copy2(src,out/src.name)
 sys.argv=[sys.argv[0]]
 from isaacsim import SimulationApp
 app=SimulationApp({'headless':True,'width':960,'height':720,'sync_loads':True,'extra_args':['--/log/file='+str(out/'kit.log')]})
 import numpy as np
 from scipy.spatial.transform import Rotation
 import omni.usd,omni.physx,omni.physics.tensors as tensors
 from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf,PhysicsSchemaTools
 from isaacsim.core.api import SimulationContext
 from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver,interface_config_loader
 from safety_monitor_v005 import SafetyMonitor
 monitor=SafetyMonitor()
 omni.usd.get_context().open_stage(str(P/'riser_io_single_wire_fastening_v005.usda'))
 for _ in range(8):app.update()
 stage=omni.usd.get_context().get_stage()
 # Supply light for evidence renders without changing collision geometry.
 from pxr import UsdLux
 stage.SetEditTarget(stage.GetSessionLayer());light=UsdLux.DomeLight.Define(stage,'/World/TrainingLight');light.CreateIntensityAttr(1500.)
 # A surface index mark makes the actual screw rotation visible.
 mark=UsdGeom.Cube.Define(stage,'/ScrewTray/Nest_1/Screw_1/RotationMark');mark.CreateSizeAttr(1.)
 mark.AddTranslateOp().Set(Gf.Vec3d(.00642,.00371,.031));mark.AddRotateZOp().Set(30.);mark.AddScaleOp().Set(Gf.Vec3f(.00012,.002,.012));mark.CreateDisplayColorAttr([Gf.Vec3f(.03,.03,.03)])
 dt=1/480
 sim=SimulationContext(physics_dt=dt,rendering_dt=1/60,backend='torch',device='cuda:0');sim.initialize_physics();sim.play()
 for _ in range(4):sim.step(render=True)
 view=tensors.create_simulation_view('torch');view.set_subspace_roots('/')
 import torch
 class HostObservations:
  """Expose GPU physics state as NumPy observations to the controller."""
  def __init__(self, native):self.native=native
  def __getattr__(self, name):
   member=getattr(self.native,name)
   if not callable(member):return member
   def call(*args,**kwargs):
    def upload(x):
     if isinstance(x,np.ndarray):
      if x.dtype==np.uint32:x=x.astype(np.int32)
      return torch.as_tensor(x,device='cpu' if name in ['set_dof_max_forces','set_dof_stiffnesses','set_dof_dampings'] else 'cuda:0').contiguous()
     return x
    def download(x):
     if isinstance(x,torch.Tensor):return x.detach().cpu().numpy()
     if isinstance(x,(tuple,list)):return type(x)(download(v) for v in x)
     return x
    return download(member(*(upload(v) for v in args),**kwargs))
   return call
 robot=HostObservations(view.create_articulation_view('/World/Robot'));sensor=HostObservations(view.create_articulation_view('/World/CableSensor'))
 assert robot.count==1 and sensor.count==1, (robot.count,sensor.count)
 seg=HostObservations(view.create_rigid_body_view('/wire1/Segment_*'));rv=HostObservations(view.create_rigid_body_view('/wire1/_0347300_ring_terminal'))
 assert seg.count==24 and rv.count==1,(seg.count,rv.count)
 names=list(robot.shared_metatype.dof_names);ln=list(sensor.shared_metatype.link_names);si=ln.index('Segment_00')
 report['sensor_links']=ln
 screw_path='/ScrewTray/Nest_1/Screw_1'
 sv=HostObservations(view.create_rigid_body_view(screw_path))
 assert sv.count==1
 screw_contacts=HostObservations(view.create_rigid_contact_view(screw_path,['/NutPlate_Coarse/Nut_1/Mesh','/wire1/_0347300_ring_terminal','/PCB_4_Terminal/Mesh'],16384))
 def screw_wrench():
  sp=sv.get_transforms()[0]
  f,p,n,d,c,st=[a.copy() for a in screw_contacts.get_contact_data(dt)]
  ff,fp,fc,fs=screw_contacts.get_friction_data(dt)
  force_by_filter=np.zeros((3,3));moment_by_filter=np.zeros((3,3));minimum=0.
  for index,(count,start) in enumerate(zip(c.ravel(),st.ravel())):
   sl=slice(int(start),int(start+count));column=index%screw_contacts.filter_count
   if count:
    forces=f[sl]*n[sl];force_by_filter[column]+=forces.sum(axis=0)
    moment_by_filter[column]+=np.cross(p[sl]-sp[:3],forces).sum(axis=0);minimum=min(minimum,float(d[sl].min()))
  for index,(count,start) in enumerate(zip(fc.ravel(),fs.ravel())):
   sl=slice(int(start),int(start+count));column=index%screw_contacts.filter_count
   if count:
    force_by_filter[column]+=ff[sl].sum(axis=0);moment_by_filter[column]+=np.cross(fp[sl]-sp[:3],ff[sl]).sum(axis=0)
  axis=Rotation.from_quat(sp[3:]).apply([0,0,1])
  return sp,force_by_filter.sum(axis=0),moment_by_filter.sum(axis=0),minimum,axis,force_by_filter

 solver=LulaKinematicsSolver(**interface_config_loader.load_supported_lula_kinematics_solver_config('Franka'))
 ai=np.array([names.index(n) for n in solver.get_joint_names()]);fi=np.array([names.index('panda_finger_joint1'),names.index('panda_finger_joint2')]);ix=np.array([0],np.uint32)
 base=robot.get_root_transforms()[0];solver.set_robot_base_pose(base[:3],np.r_[base[6],base[3:6]])
 print('ROBOT_BASE',base.tolist(),flush=True)
 def solve(p,rr,seed):
  joints,ok=solver.compute_inverse_kinematics('panda_hand',p,np.r_[rr[3],rr[:3]],warm_start=seed,position_tolerance=.0005,orientation_tolerance=.01)
  if ok:return np.asarray(joints)
  from scipy.optimize import least_squares
  goal=Rotation.from_quat(rr)
  def residual(qs):
   pp,rot=solver.compute_forward_kinematics('panda_hand',qs)
   return np.r_[(np.asarray(pp)-p)*10,(Rotation.from_matrix(rot)*goal.inv()).as_rotvec()]
  lo=np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-.0175,-2.8973]);hi=np.array([2.8973,1.7628,2.8973,-.0698,2.8973,3.7525,2.8973])
  best=None
  for start in [seed,[0,-.8,0,-2.6,0,1.8,.7854],[.5,.2,-.5,-2.,0,2.3,0],[-.5,.2,.5,-2.,0,2.3,1.5]]:
   r=least_squares(residual,np.clip(start,lo+1e-6,hi-1e-6),bounds=(lo,hi),max_nfev=160,ftol=1e-10,xtol=1e-10,gtol=1e-10)
   if best is None or np.linalg.norm(r.fun)<np.linalg.norm(best.fun):best=r
   if np.linalg.norm(r.fun[:3])<.005 and np.linalg.norm(r.fun[3:])<.01:return r.x
  raise TaskFailure('IK unreachable: '+str(p)+' residual '+str(best.fun))
 target=np.array([-.092,-.071,.768]);Rdown=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
 q=robot.get_dof_positions()[0].copy();q[ai]=[0,-.5,0,-2.2,0,1.7,.7854];q[fi]=.04
 orient=Rotation.from_matrix(Rdown).as_quat()
 q[ai]=solve(np.array([-.092,-.071,.9164]) if args.entry_only else np.array([-.24,.02,1.0434]),orient,q[ai])
 if args.entry_only:q[fi]=.0075
 robot.set_dof_positions(q[None].astype(np.float32),ix);robot.set_dof_velocities(np.zeros_like(q)[None].astype(np.float32),ix);robot.set_dof_position_targets(q[None].astype(np.float32),ix)
 if args.entry_only:
  q[fi]=0.;robot.set_dof_position_targets(q[None].astype(np.float32),ix)
 # Effort limits for physical contact grasp; position drives remain compliant.
 forces=robot.get_dof_max_forces().copy();forces[0,fi]=8.;robot.set_dof_max_forces(forces,ix)
 target=np.array([-.092,-.071,.768]);Rdown=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
 phase='reset';stepnum=0;contact={'left':0.,'right':0.,'seat':0.,'min_sep':0.,'forbidden':0.,'clamp':0.};maxforbidden=0.
 jp0=np.array([stage.GetPrimAtPath(f'/wire1/Bend_{i:02d}').GetAttribute('physics:localPos0').Get() for i in range(23)])
 jp1=np.array([stage.GetPrimAtPath(f'/wire1/Bend_{i:02d}').GetAttribute('physics:localPos1').Get() for i in range(23)])
 cv=HostObservations(view.create_rigid_contact_view('/World/ForceControl',['/World/Table/Top'],256))
 def oncontacts(headers,data):
  for h in headers:
   a=str(PhysicsSchemaTools.intToSdfPath(h.collider0));b=str(PhysicsSchemaTools.intToSdfPath(h.collider1));pair=a+'|'+b
   for k in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
    c=data[k];imp=np.asarray(c.impulse,float);normal=np.asarray(c.normal,float);force=normal*float(imp)/dt if imp.ndim==0 else imp/dt;fn=abs(float(np.dot(force,normal)))
    ring='_0347300_ring_terminal' in pair
    screw=screw_path in pair
    if ring and 'panda_leftfinger' in pair:contact['left']+=fn
    if ring and 'panda_rightfinger' in pair:contact['right']+=fn
    if ring and screw:contact['clamp']+=abs(float(force[2]))
    if ring and '/PCB_4_Terminal' in pair:contact['seat']+=abs(float(force[2]))
    if '/World/Robot/' in pair and not ('panda_link0' in pair) and not ((ring or screw) and ('finger' in pair)):
     # Robot self-contact is filtered at articulation level. Other contacts are unintended.
     if not (a.startswith('/World/Robot/') and b.startswith('/World/Robot/')):contact['forbidden']+=fn
    if ring or screw or '/wire1/Segment_' in pair:
     if float(c.separation)<contact['min_sep']:
      contact['min_sep']=float(c.separation);contact['deepest_pair']=pair
 contact_errors=[]
 def checked_contacts(headers,data):
  try:oncontacts(headers,data)
  except Exception as e:contact_errors.append(repr(e))
 sub=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(checked_contacts)
 def handpose():return solver.compute_forward_kinematics('panda_hand',robot.get_dof_positions()[0,ai])
 def tick(n=1):
  global stepnum,contact,maxforbidden
  for _ in range(n):
   contact={'left':0.,'right':0.,'seat':0.,'min_sep':0.,'forbidden':0.,'clamp':0.}
   sim.step(render=False);stepnum+=1
   if contact_errors:raise RuntimeError("contact measurement failed: "+str(contact_errors[-1]))
   transforms=seg.get_transforms().copy();xyz=transforms[:,:3];ring=rv.get_transforms()[0].copy();w=sensor.get_link_incoming_joint_force()[0,si].copy()
   joint_gap=float(np.linalg.norm(xyz[:-1]+Rotation.from_quat(transforms[:-1,3:]).apply(jp0)-xyz[1:]-Rotation.from_quat(transforms[1:,3:]).apply(jp1),axis=1).max())
   if not np.isfinite(xyz).all() or not np.isfinite(w).all():raise RuntimeError('nonfinite physics state')
   aa=xyz[1:-1]-xyz[:-2];bb=xyz[2:]-xyz[1:-1];cc=xyz[2:]-xyz[:-2];den=2*np.linalg.norm(np.cross(aa,bb),axis=1)
   radii=np.linalg.norm(aa,axis=1)*np.linalg.norm(bb,axis=1)*np.linalg.norm(cc,axis=1)/np.maximum(den,1e-12)
   radius=float(radii.min());force=float(np.linalg.norm(w[:3]));moment=float(np.linalg.norm(w[3:]));maxforbidden=max(maxforbidden,contact['forbidden'])
   rec={'time_s':stepnum*dt,'phase':phase,'attachment_force_N':force,'attachment_moment_Nm':moment,'min_bend_radius_m':radius,'min_contact_separation_m':contact['min_sep'],'finger_left_N':contact['left'],'finger_right_N':contact['right'],'seat_axial_N':contact['seat'],'forbidden_robot_contact_N':contact['forbidden'],'ring_x':float(ring[0]),'ring_y':float(ring[1]),'ring_z':float(ring[2]),'ring_tilt_deg':float(np.degrees(np.arccos(np.clip(Rotation.from_quat(ring[3:]).as_matrix()[2,2],-1,1)))),'control_box_N':float(np.linalg.norm(cv.get_net_contact_forces(dt)[0]))}
   sp,sf,sm,ssep,saxis,support_forces=screw_wrench()
   rec.update(screw_x=float(sp[0]),screw_y=float(sp[1]),screw_z=float(sp[2]),screw_tilt_deg=float(np.degrees(np.arccos(np.clip(saxis[2],-1,1)))),screw_axial_N=float(np.max(np.abs(support_forces[:,2]))),screw_lateral_N=float(np.max(np.linalg.norm(support_forces[:,:2],axis=1))),screw_torque_Nm=float(abs(np.dot(sm,saxis))),screw_bending_Nm=float(np.linalg.norm(sm-np.dot(sm,saxis)*saxis)),screw_speed_m_s=float(np.linalg.norm(sv.get_velocities()[0,:3])))
   rec['screw_signed_torque_z_Nm']=float(sm[2])
   screw_rot=Rotation.from_quat(sp[3:]).as_matrix();rec['screw_yaw_deg']=float(np.degrees(np.arctan2(screw_rot[1,0],screw_rot[0,0])))
   rec['clamp_force_N']=contact['clamp']
   rec['screw_reaction_z_N']=float(sf[2]);rec['nut_reaction_z_N']=float(support_forces[0,2])
   rec['nut_axial_N']=float(abs(support_forces[0,2]));rec['terminal_clamp_tensor_N']=float(abs(support_forces[1,2]))
   rec['attachment_axial_N']=abs(float(w[0]));rec['attachment_lateral_N']=float(np.linalg.norm(w[1:3]));rec['attachment_bending_Nm']=float(np.linalg.norm(w[4:]))
   rec['min_contact_separation_m']=min(rec['min_contact_separation_m'],ssep)
   rec['deepest_pair']=contact.get('deepest_pair','');rec['max_joint_anchor_gap_m']=joint_gap;rec['tightest_bend_segment']=int(np.argmin(radii)+1);rec['ring_speed_m_s']=float(np.linalg.norm(rv.get_velocities()[0,:3]))
   rec['safety_state']=monitor.update(rec,dt) if phase not in ['settle','sensor_calibration','reset'] else 'INITIALIZATION';records.append(rec)
   if force>100 or np.max(np.linalg.norm(xyz-xyz[0],axis=1))>.8:raise RuntimeError('numerical or structural instability')
   if args.record and stepnum%24==0:write_video_frame(rec)
   if stepnum%960==0:(out/'live_state.json').write_text(json.dumps(rec));print('STATE',json.dumps(rec),flush=True)
 def setup_render():
  import omni.replicator.core as rep
  if not hasattr(setup_render,'annot'):
   cam=UsdGeom.Camera.Define(stage,'/World/TrainingCamera');cam.CreateFocalLengthAttr(22.);cam.CreateClippingRangeAttr(Gf.Vec2f(.01,10.))
   camera_pose=Gf.Matrix4d().SetLookAt(Gf.Vec3d(.13,-.55,1.24),Gf.Vec3d(-.16,.035,.785),Gf.Vec3d(0,0,1)).GetInverse()
   UsdGeom.Xformable(cam).AddTransformOp().Set(camera_pose)
   prod=rep.create.render_product(str(cam.GetPath()),(960,720));setup_render.annot=rep.AnnotatorRegistry.get_annotator('rgb');setup_render.annot.attach([prod])
   setup_render.camera_inverse=np.array(camera_pose.GetInverse())
   close=UsdGeom.Camera.Define(stage,'/World/FasteningCamera');close.CreateFocalLengthAttr(35.);close.CreateClippingRangeAttr(Gf.Vec2f(.005,10.))
   cp=Gf.Matrix4d().SetLookAt(Gf.Vec3d(.035,-.22,.99),Gf.Vec3d(-.092,-.071,.785),Gf.Vec3d(0,0,1)).GetInverse();UsdGeom.Xformable(close).AddTransformOp().Set(cp)
   setup_render.close_camera=close;setup_render.close_fixed=cp
   closeprod=rep.create.render_product(str(close.GetPath()),(320,240));setup_render.close=rep.AnnotatorRegistry.get_annotator('rgb');setup_render.close.attach([closeprod])
  return setup_render.annot
 def write_video_frame(rec):
  global video_writer
  import cv2
  annot=setup_render()
  if not phase.startswith('screw') and phase not in ['full_verify','torque_measurement_check','post_check_relax']:
   center=seg.get_transforms()[-3:,:3].mean(axis=0)
   eye=center+np.array([.10,.14,.075])
   cp=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float,eye)),Gf.Vec3d(*map(float,center)),Gf.Vec3d(0,0,1)).GetInverse()
  else:cp=setup_render.close_fixed
  UsdGeom.Xformable(setup_render.close_camera).GetOrderedXformOps()[0].Set(cp)
  for _ in range(2):sim.render()
  a=annot.get_data()
  if a is None or not a.size:return
  frame=cv2.cvtColor(a[:,:,:3],cv2.COLOR_RGB2BGR)
  close=setup_render.close.get_data()
  if close is not None and close.size:
   frame[130:370,630:950]=cv2.cvtColor(close[:,:,:3],cv2.COLOR_RGB2BGR)
   cv2.rectangle(frame,(629,129),(951,371),(230,230,230),1)
  if rec['min_bend_radius_m']<.035:
   pos=seg.get_transforms()[rec['tightest_bend_segment'],:3];campos=np.r_[pos,1.]@setup_render.camera_inverse
   if campos[2]<-.01:
    uv=(int(480+960*22/20.955*campos[0]/(-campos[2])),int(360-960*22/20.955*campos[1]/(-campos[2])))
    cv2.circle(frame,uv,22,(0,0,255),3)
  points=seg.get_transforms()[:,:3].astype(float);j=rec['tightest_bend_segment']
  direction=points[j+1]-points[j-1];direction/=max(np.linalg.norm(direction),1e-9)
  side=points[j]-points[j-1];side-=direction*np.dot(side,direction)
  if np.linalg.norm(side)<1e-8:side=np.cross(direction,[0,0,1])
  side/=max(np.linalg.norm(side),1e-9)
  selection=points[max(0,j-3):min(len(points),j+4)]-points[j]
  projected=np.column_stack([selection@direction,selection@side])*1000
  scale=220/max(float(np.ptp(projected,axis=0).max()),60.)
  pix=np.rint(projected*scale*np.array([1,-1])+[790,480]).astype(np.int32)
  cv2.rectangle(frame,(630,380),(950,590),(28,28,28),-1)
  cv2.putText(frame,'Measured local cable shape',(640,402),cv2.FONT_HERSHEY_SIMPLEX,.46,(230,230,230),1,cv2.LINE_AA)
  curve_color=(40,60,255) if rec['min_bend_radius_m']<.035 else (80,220,100)
  cv2.polylines(frame,[pix],False,curve_color,4,cv2.LINE_AA)
  for point in pix:cv2.circle(frame,tuple(point),3,(230,230,230),-1)
  cv2.putText(frame,f"Rmin {rec['min_bend_radius_m']*1000:.1f} mm / limit 35 mm",(640,575),cv2.FONT_HERSHEY_SIMPLEX,.44,curve_color,1,cv2.LINE_AA)
  if video_writer is None:
   video_writer=cv2.VideoWriter(str(out/'rollout.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20.,(960,720))
   if not video_writer.isOpened():raise RuntimeError('MP4 encoder unavailable')
  violated=[k for k,v in monitor.stats.items() if v['duration_s']>0]
  color=(40,60,255) if violated else (30,190,255) if rec['safety_state']=='WARNING' else (60,220,80)
  cv2.rectangle(frame,(0,0),(960,120),(20,20,20),-1)
  lines=[f"{rec['time_s']:6.2f}s  {phase}  "+('EPISODE VIOLATED' if violated else 'BEND WARNING' if rec['safety_state']=='WARNING' else 'WITHIN CANDIDATE LIMITS'),
   f"Cable F {rec['attachment_force_N']:.2f} N | Rmin {1000*rec['min_bend_radius_m']:.1f} mm | joint M {rec['attachment_bending_Nm']:.3f} Nm",
   f"Screw tilt {rec['screw_tilt_deg']:.1f} deg | axial {rec['screw_axial_N']:.2f} N | side {rec['screw_lateral_N']:.2f}/3 N | T {rec['screw_torque_Nm']:.4f} Nm",
   'Limits: prototype operating bounds; material damage NOT calibrated']
  for i,line in enumerate(lines):cv2.putText(frame,line,(12,24+27*i),cv2.FONT_HERSHEY_SIMPLEX,.55,color if i==0 else (240,240,240),1,cv2.LINE_AA)
  if violated:cv2.putText(frame,', '.join(violated),(12,700),cv2.FONT_HERSHEY_SIMPLEX,.5,(40,60,255),1,cv2.LINE_AA)
  video_writer.write(frame);frame_log.append({'frame':len(frame_log),'time_s':rec['time_s'],'phase':phase,'cable_centers_m':seg.get_transforms()[:,:3].tolist(),'ring_pose':rv.get_transforms()[0].tolist(),'screw_pose':sv.get_transforms()[0].tolist()})
 def capture(name):
  annot=setup_render()
  for _ in range(8):sim.render()
  from PIL import Image
  a=annot.get_data()
  if a is not None and a.size:Image.fromarray(a).save(out/(name+'.png'))
  pose=rv.get_transforms()[0].tolist();events.append({'name':name,'time_s':stepnum*dt,'ring_pose':pose,'screw_pose':sv.get_transforms()[0].tolist(),'cable_centers':seg.get_transforms()[:,:3].tolist()});print('EVENT',name,pose,flush=True)
 def move(tcp,seconds=3.,width=None,rotation=None,stop_force=None):
  global q
  desired_rot=Rdown if rotation is None else rotation
  start,rot=handpose();end=np.array(tcp)-desired_rot@np.array([0,0,.1034]);r0=Rotation.from_matrix(rot);r1=Rotation.from_matrix(desired_rot);dr=(r1*r0.inv()).as_rotvec()
  steps=max(1,int(seconds*60));seed=robot.get_dof_positions()[0,ai].copy()
  for k in range(steps):
   t=(k+1)/steps;u=t*t*(3-2*t);p=start+(end-start)*u;rr=(Rotation.from_rotvec(dr*u)*r0).as_quat()
   seed=solve(p,rr,seed);q[ai]=seed
   if width is not None:q[fi]=width/2
   robot.set_dof_position_targets(q[None].astype(np.float32),ix);tick(8)
   if phase=='screw_slanted_loading' and (records[-1]['screw_lateral_N']>4.5 or records[-1]['screw_axial_N']>15.):return
   if phase=='screw_overdrive' and (records[-1]['screw_torque_Nm']>.075 or records[-1]['screw_axial_N']>15.):return
   if stop_force is not None and records[-1]['attachment_force_N']>=stop_force:
    report['pull_stopped_force_N']=records[-1]['attachment_force_N'];return
  tick(120)
  hp,hr=handpose();err=np.linalg.norm(np.asarray(hp)-end)
  if err>.015:raise TaskFailure('hand tracking error '+str(err))
 def move_screw(goal_tip,seconds=3.,goal_rotation=None,stop_axial=None,stop_lateral=None,force_target=None):
  global q
  initial=sv.get_transforms()[0].copy();r0=Rotation.from_quat(initial[3:])
  hp_initial,hr_initial=handpose();grasp_offset=r0.inv().apply(np.asarray(hp_initial)-initial[:3]);grasp_rotation=r0.inv().as_matrix()@np.asarray(hr_initial)
  r1=r0 if goal_rotation is None else Rotation.from_matrix(goal_rotation)
  dr=(r1*r0.inv()).as_rotvec();bias=np.zeros(3);angle_bias=np.zeros(3);admittance=0.
  steps=max(1,int(seconds*60));seed=robot.get_dof_positions()[0,ai].copy()
  for k in range(steps+60):
   t=min(1.,(k+1)/steps);u=t*t*(3-2*t)
   desired=initial[:3]+(np.asarray(goal_tip)-initial[:3])*u
   if force_target is not None:
    admittance=float(np.clip(admittance+.000002*(records[-1]['screw_reaction_z_N']-force_target),-.002,.002))
    desired[2]+=admittance
   desired_rotation=Rotation.from_rotvec(dr*u)*r0
   actual=sv.get_transforms()[0].copy();ar=Rotation.from_quat(actual[3:])
   hp,hr=handpose();hp=np.asarray(hp);hr=np.asarray(hr)
   error=desired-actual[:3]
   bias=np.clip(bias+.05*error,-.003,.003)
   if force_target is not None:bias[2]=0.
   angle_bias=np.clip(angle_bias+.025*(desired_rotation*ar.inv()).as_rotvec(),-.03,.03)
   rd=(Rotation.from_rotvec(angle_bias)*desired_rotation).as_matrix()
   desired_hand=desired+bias+rd@grasp_offset
   hand_rotation=rd@grasp_rotation
   seed=solve(desired_hand,Rotation.from_matrix(hand_rotation).as_quat(),seed);q[ai]=seed
   robot.set_dof_position_targets(q[None].astype(np.float32),ix);tick(8)
   if stop_axial is not None and records[-1]['screw_axial_N']>stop_axial:return
   if stop_lateral is not None and records[-1]['screw_lateral_N']>stop_lateral:return
   if phase.startswith('screw_turn') and records[-1]['clamp_force_N']>1. and sv.get_transforms()[0,2]<.7535 and records[-1]['ring_tilt_deg']<1.:return
   if phase.startswith('screw_turn') and records[-1]['screw_tilt_deg']>12:
    raise TaskFailure('screw lost coaxial engagement during turning')
  return
 def grip(width,seconds=1.5):
  global q
  start=q[fi].copy();steps=int(seconds*60)
  for k in range(steps):q[fi]=start+(width/2-start)*(k+1)/steps;robot.set_dof_position_targets(q[None].astype(np.float32),ix);tick(8)
 if args.screw_only:
  checkpoint=json.loads((O/'wire_checkpoint.json').read_text())
  seg.set_transforms(np.asarray(checkpoint['segments'],np.float32),np.arange(seg.count,dtype=np.uint32))
  seg.set_velocities(np.zeros((seg.count,6),np.float32),np.arange(seg.count,dtype=np.uint32))
  rv.set_transforms(np.asarray(checkpoint['ring'],np.float32),ix);rv.set_velocities(np.zeros((1,6),np.float32),ix)
  report['scope']='screw_only_debug_reset'
  if args.entry_only:
   sv.set_transforms(np.array([[-.092,-.071,.780,0,0,0,1]],np.float32),ix);sv.set_velocities(np.zeros((1,6),np.float32),ix)
   report['scope']='pregrasped_screw_debug'
 phase='settle';tick(960);capture('01_settled')
 report['settled_ring_pose']=rv.get_transforms()[0].tolist()
 # Known load on the fixed sensor link validates reaction-force scale independently of contact sensors.
 phase='sensor_calibration';tick(60);w0=sensor.get_link_incoming_joint_force()[0,si,:3].copy()
 calibration_force=np.zeros((1,sensor.max_links,3),np.float32);calibration_force[0,si,0]=.25
 for _ in range(120):sensor.apply_forces_and_torques_at_position(calibration_force,None,None,ix,True);tick()
 w1=sensor.get_link_incoming_joint_force()[0,si,:3].copy()
 cf,cp,cn,cd,cc,cs=cv.get_contact_data(dt)
 control_force=sum((cf[int(start):int(start+count)]*cn[int(start):int(start+count)]).sum(axis=0) for count,start in zip(cc.ravel(),cs.ravel()))
 report['known_weight_contact_vector_N']=np.asarray(control_force).tolist()
 assert .9<float(control_force[2])<1.1,'contact force orientation or scale is incorrect'
 report['attachment_sensor_known_load_N']=.25;report['attachment_sensor_increment_N']=float(np.linalg.norm(w1-w0))
 report['attachment_sensor_calibration_pass']=abs(report['attachment_sensor_increment_N']-.25)<.025
 assert report['attachment_sensor_calibration_pass'],'attachment sensor calibration failed'
 phase='settle';tick(120)
 reset_monitor=SafetyMonitor()
 for row in records[-120:]:reset_monitor.update(row,dt)
 report['reset_safety']=reset_monitor.summary()
 assert report['reset_safety']['safety_status']=='SAFE','settled reset is not safe'
 if not args.settle_only:
  if not args.screw_only:
   reset_ring=rv.get_transforms()[0].copy();pos=reset_ring[:3]
   grasp_point=pos+Rotation.from_quat(reset_ring[3:]).apply([0,.023,args.grasp_z])
   phase='approach';move(grasp_point+[0,0,.10],3.,.08);capture('02_approach')
   phase='descend';move(grasp_point,3.,.08)
   phase='grasp';grip(args.grip);capture('03_grasp')
   phase='lift';move(grasp_point+[0,0,.055],3.);capture('04_lift')
   lifted=rv.get_transforms()[0,:3].copy();report['lift_height_m']=float(lifted[2]-pos[2]);assert report['lift_height_m']>.04,'grasp failed: terminal did not lift'
   hp,hr=handpose();tcp=np.asarray(hp)+np.asarray(hr)@np.array([0,0,.1034])
   Rdown=Rotation.from_quat(rv.get_transforms()[0,3:]).as_matrix().T@np.asarray(hr)
   phase='level_terminal';move(tcp,2.);capture('04b_level')
   # Transport at a constant safe height, then lower over the matching seat.
   hp,hr=handpose();tcp=np.asarray(hp)+np.asarray(hr)@np.array([0,0,.1034]);offset=rv.get_transforms()[0,:3]-tcp
   if args.probe in ['bend','bend_pull']:
    phase='deliberate_bend';move(tcp,4.,rotation=Rotation.from_euler('z',110,degrees=True).as_matrix()@Rdown);tick(240);capture('04c_bend')
    phase='recover_bend';move(tcp,4.);capture('04d_recover')
   if args.probe in ['pull','bend_pull']:
    phase='deliberate_pull';move(np.array([target[0],-.18,.84])-offset,8.,stop_force=12.5);capture('04e_pull')
   phase='transport';move(np.array([target[0],target[1],pos[2]+.055])-offset,5.);capture('05_above_target')
   hp,hr=handpose();tcp=np.asarray(hp)+np.asarray(hr)@np.array([0,0,.1034]);Rdown=Rotation.from_quat(rv.get_transforms()[0,3:]).as_matrix().T@np.asarray(hr)
   phase='level_above_target';move(tcp,2.);capture('05b_level_above')
   hp,hr=handpose();tcp=np.asarray(hp)+np.asarray(hr)@np.array([0,0,.1034]);offset=rv.get_transforms()[0,:3]-tcp
   phase='align';move(np.r_[target[:2],.800]-offset,2.)
   phase='seat';move(target-offset+[0,0,-.0002],3.);tick(240);capture('06_seated')
   phase='release';grip(.08,1.);capture('07_release')
   phase='retract';move(target-offset+[0,0,.12],3.,.08)
   phase='verify';tick(1440);capture('08_final')
   endpose=rv.get_transforms()[0].copy();err=endpose[:3]-target;tilt=records[-1]['ring_tilt_deg'];speed=float(np.linalg.norm(rv.get_velocities()[0,:3]))
   report['final_error_m']=err.tolist();report['final_tilt_deg']=tilt;report['final_speed_m_s']=speed
   verify=[r for r in records if r['phase']=='verify']
   report['task_success']=bool(len(verify)>=1440 and all(math.hypot(r['ring_x']-target[0],r['ring_y']-target[1])<.001 and abs(r['ring_z']-target[2])<.0005 and r['ring_tilt_deg']<5 and r['ring_speed_m_s']<.01 and r['finger_left_N']<.05 and r['finger_right_N']<.05 for r in verify))
  report['wire_seated_before_screw']=report['task_success']
  if not args.screw_only:(O/'wire_checkpoint.json').write_text(json.dumps({'segments':seg.get_transforms().tolist(),'ring':rv.get_transforms().tolist(),'source_run':args.name}))

  report['release_hold_s']=3.
  if not args.wire_only:
   stiffness=robot.get_dof_stiffnesses().copy();stiffness[0,fi]=1200.;robot.set_dof_stiffnesses(stiffness,ix)
   damping=robot.get_dof_dampings().copy();damping[0,fi]=15.;robot.set_dof_dampings(damping,ix)
   forces=robot.get_dof_max_forces().copy();forces[0,fi]=15.;robot.set_dof_max_forces(forces,ix)
   report['screw_gripper_limits']={'finger_drive_stiffness_N_m':1200.,'finger_drive_damping_Ns_m':15.,'finger_effort_cap_N':15.}
   Rdown=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
   if not args.entry_only:
    sp=sv.get_transforms()[0].copy();grasp=sp[:3]+Rotation.from_quat(sp[3:]).apply([0,0,.033])
    phase='screw_approach';move(grasp+[0,0,.10],3.,.06)
    phase='screw_descend';move(grasp,3.,.06)
    phase='screw_grasp';grip(0.,1.5);capture('09_screw_grasp')
    phase='screw_lift';move(grasp+[0,0,.07],3.)
    assert sv.get_transforms()[0,2]>sp[2]+.04,'screw grasp did not lift'
    hp,hr=handpose();tcp=np.asarray(hp)+np.asarray(hr)@np.array([0,0,.1034])
    Rdown=Rotation.from_quat(sv.get_transforms()[0,3:]).as_matrix().T@np.asarray(hr)
    phase='screw_level';move(tcp,2.)
    hp,hr=handpose();tcp=np.asarray(hp)+np.asarray(hr)@np.array([0,0,.1034]);offset=sv.get_transforms()[0,:3]-tcp
    phase='screw_transport';move(np.array([target[0],target[1],.81])-offset,4.)
    if args.probe=='screw_tilt':
     tilted=Rotation.from_euler('x',18,degrees=True).as_matrix()
     phase='screw_tilt_in_air';move_screw([target[0],target[1],.786],3.,tilted)
     phase='screw_slanted_loading';move_screw([target[0],target[1],.765],4.,tilted,stop_axial=15.,stop_lateral=4.5)
     tick(240);capture('10_screw_slanted_loading')
     phase='screw_backoff';sp=sv.get_transforms()[0].copy();move_screw(sp[:3]+[0,0,.025],3.,Rotation.from_quat(sp[3:]).as_matrix())
     if args.stop_after_probe:
      report['scope']='wire_then_screw_slanted_load_probe';capture('10b_screw_recovered')
      raise TaskFailure('slanted loading probe complete; fastening not attempted')
   phase='screw_precision_align';move_screw([target[0],target[1],.780],3.,np.eye(3));capture('10_precision_align')
   phase='screw_ring_entry';ring_now=rv.get_transforms()[0,:3].copy();move_screw([ring_now[0],ring_now[1],.7665],5.,np.eye(3),stop_axial=3.)
   if sv.get_transforms()[0,2]>.7685:raise TaskFailure('screw blocked at terminal bore')
   phase='screw_axis_center';move_screw([target[0],target[1],.7665],2.,np.eye(3),stop_axial=3.)
   phase='screw_thread_entry';move_screw([target[0],target[1],.7620],5.,np.eye(3),stop_axial=1.);capture('11_thread_entry')
   if sv.get_transforms()[0,2]>.766:raise TaskFailure('screw tip blocked above thread entrance')

   phase='screw_load_relief';sp=sv.get_transforms()[0].copy()
   move_screw(sp[:3]+[0,0,.00025],1.,Rotation.from_quat(sp[3:]).as_matrix())
   # Each half-turn is transmitted through actual finger/head and helical contact.
   # Regrasp above the seat keeps the robot wrist within its range.
   turns=[]
   for turn in range(args.max_turns):
    current=sv.get_transforms()[0].copy();rs=Rotation.from_quat(current[3:]).as_matrix();wrist_start=float(robot.get_dof_positions()[0,names.index('panda_joint7')])
    phase='screw_turn_'+str(turn)
    goalrot=Rotation.from_euler('z',-120,degrees=True).as_matrix()@rs
    move_screw([target[0],target[1],current[2]-.002/3],3.,goalrot,stop_axial=10.,force_target=args.screw_force)
    pose=sv.get_transforms()[0].copy()
    turns.append({'turn':turn,'tip_z':float(pose[2]),'torque_Nm':records[-1]['screw_torque_Nm'],'clamp_N':records[-1]['clamp_force_N'],'actual_rotation_deg':float(np.degrees((Rotation.from_quat(pose[3:])*Rotation.from_matrix(rs).inv()).as_rotvec()[2])),'axial_advance_m':float(current[2]-pose[2])})
    (out/'turn_progress.json').write_text(json.dumps(turns,indent=2));print('SCREW_TURN',turns[-1],flush=True)
    if pose[2]<.7536 and np.mean([r['clamp_force_N'] for r in records[-60:]])>.2:break
    if pose[2]>.7625:raise TaskFailure('screw failed to engage first thread')
    phase='screw_regrasp_open';grip(.040,.5)
    phase='screw_regrasp_return'
    actual=sv.get_transforms()[0].copy();head=actual[:3]+Rotation.from_quat(actual[3:]).apply([0,0,.033])
    move(head+[0,0,.010],1.,.040,rotation=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]]))
    actual=sv.get_transforms()[0].copy();head=actual[:3]+Rotation.from_quat(actual[3:]).apply([0,0,.033])
    move(head,1.,.040)
    phase='screw_regrasp_close';grip(0.,.75)
   report['screw_turns']=turns
   if args.probe=='screw_overload':
    hp,hr=handpose();tcp=np.asarray(hp)+np.asarray(hr)@np.array([0,0,.1034])
    phase='screw_overdrive';move(tcp+[0,0,-.004],4.,rotation=Rotation.from_euler('z',-170,degrees=True).as_matrix()@Rdown);tick(480);capture('12_screw_overdrive')
   capture('13_fastened')
   phase='screw_release';grip(.06,1.)
   hp,hr=handpose();tcp=np.asarray(hp)+np.asarray(hr)@np.array([0,0,.1034])
   phase='screw_retract';move(tcp+[0,0,.10],3.,.06)
   phase='full_verify';tick(1440);capture('14_full_final')
   verification=[r for r in records if r['phase']=='full_verify']
   report['screw_fastened']=bool(len(verification)>=1440 and all(abs(r['screw_z']-.753)<.0007 and math.hypot(r['screw_x']-target[0],r['screw_y']-target[1])<.001 and r['screw_tilt_deg']<5 and r['screw_speed_m_s']<.01 for r in verification))
   report['ring_retained']=bool(all(math.hypot(r['ring_x']-target[0],r['ring_y']-target[1])<.0015 and abs(r['ring_z']-.768)<.0007 and r['ring_tilt_deg']<5 for r in verification))
   report['mean_clamp_force_N']=float(np.mean([r['clamp_force_N'] for r in verification[-480:]]))
   report['clamp_contact_verified']=report['mean_clamp_force_N']>.2
   report['task_success']=bool(report['screw_fastened'] and report['ring_retained'] and report['clamp_contact_verified'])
   if report['screw_fastened']:
    phase='torque_measurement_check';tick(120)
    baseline=float(np.mean([r['screw_signed_torque_z_Nm'] for r in records[-60:]]))
    known_torque=np.array([[0.,0.,.002]],np.float32)
    for _ in range(120):sv.apply_forces_and_torques_at_position(None,known_torque,None,ix,True);tick()
    measured=float(np.mean([r['screw_signed_torque_z_Nm'] for r in records[-60:]]))-baseline
    report['screw_torque_calibration']={'applied_Nm':.002,'reaction_change_Nm':measured,'pass':abs(abs(measured)-.002)<.0006}
    phase='post_check_relax';tick(120)


 report['physics_valid']=True
except Exception as failure:
 report['execution_failure_type']='task' if isinstance(failure,TaskFailure) else 'runtime'
 report['error']=traceback.format_exc();print(report['error'],flush=True)
finally:
 if video_writer is not None:video_writer.release()
 (out/'video_frames.json').write_text(json.dumps(frame_log))
 if records:
  with (out/'metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
  selected=[r for r in records if r['phase'] not in ['reset','settle','sensor_calibration']];selected=selected or records
  report['peak_attachment_force_N']=max(r['attachment_force_N'] for r in selected);report['minimum_bend_radius_m']=min(r['min_bend_radius_m'] for r in selected)
  report['worst_contact_separation_m']=min(r['min_contact_separation_m'] for r in selected);report['peak_forbidden_robot_contact_N']=max(r['forbidden_robot_contact_N'] for r in selected)
  report['control_box_mean_N']=float(sum(r['control_box_N'] for r in records[-240:])/len(records[-240:]))
  violations=[]
  if report['peak_attachment_force_N']>1.:violations.append('attachment_overload')
  if report['minimum_bend_radius_m']<.035:violations.append('overbend')
  if report['peak_forbidden_robot_contact_N']>.5:violations.append('unintended_robot_contact')
  report['safety_violations']=violations;report['safety_status']=('VIOLATED' if violations else 'SAFE') if report['physics_valid'] else 'UNKNOWN'
  if report['worst_contact_separation_m']<-.0005:report['physics_valid']=False;report['physics_invalid_reason']='contact_penetration_over_0p5mm';report['safety_status']='UNKNOWN'
  report['maximum_joint_anchor_gap_m']=max(r['max_joint_anchor_gap_m'] for r in selected)
  if report['maximum_joint_anchor_gap_m']>.0005:report['physics_valid']=False;report['physics_invalid_reason']='joint_anchor_gap_over_0p5mm';report['safety_status']='UNKNOWN'
  report['online_safety_evaluation']=monitor.summary(report['task_success'])
  if report.get('execution_failure_type')=='task':report['physics_valid']=report['online_safety_evaluation']['physics_and_measurement_valid']
  report['safety_violations']=report['online_safety_evaluation']['violations']
  report['safety_status']=report['online_safety_evaluation']['safety_status'] if report['physics_valid'] else 'UNKNOWN'
  report['safe_success']=report['task_success'] and report['physics_valid'] and report['safety_status']=='SAFE'
 (out/'events.json').write_text(json.dumps(events,indent=2));(out/'result.json').write_text(json.dumps(report,indent=2));print('RESULT',json.dumps(report),flush=True)
 if app:app.close()
