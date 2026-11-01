"""Physical Franka single-wire replay; no task-body pose writes after reset."""
from pathlib import Path
import argparse,sys,json,traceback,time,csv,os,math,hashlib,shutil
P=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene');O=P/'single_wire_fastening_v003'
pa=argparse.ArgumentParser();pa.add_argument('--name',default='attempt_001');pa.add_argument('--settle-only',action='store_true');pa.add_argument('--grasp-z',type=float,default=.007);pa.add_argument('--grip',type=float,default=.0);pa.add_argument('--probe',choices=['none','pull','bend','bend_pull','screw_tilt','screw_overload'],default='none');pa.add_argument('--record',action='store_true');pa.add_argument('--wire-only',action='store_true');args=pa.parse_args()
out=O/args.name;out.mkdir(parents=True,exist_ok=True)
if (out/'result.json').exists():raise RuntimeError('Refusing to overwrite completed run: '+str(out))
report={'task_success':False,'physics_valid':False,'safety_status':'UNKNOWN','no_task_pose_writes_during_rollout':True,'ring_geometry_unchanged':True};app=None;records=[];events=[];video_writer=None;frame_log=[]
try:
 build=json.loads((O/'build_report.json').read_text());report['ring_geometry_unchanged']=build['ring_mesh_unchanged'];report['source_sha256']=hashlib.sha256((P/'riser_io_wire_guides_v001.usda').read_bytes()).hexdigest();assert report['source_sha256']==build['source_sha256']
 report['scene_sha256']=hashlib.sha256((P/'riser_io_single_wire_fastening_v003.usda').read_bytes()).hexdigest();report['probe']=args.probe
 for src in [Path(__file__),Path(__file__).with_name('safety_monitor_v003.py'),Path(__file__).with_name('build_single_wire_fastening_v003.py'),O/'build_report.json']:shutil.copy2(src,out/src.name)
 sys.argv=[sys.argv[0]]
 from isaacsim import SimulationApp
 app=SimulationApp({'headless':True,'width':960,'height':720,'sync_loads':True,'extra_args':['--/log/file='+str(out/'kit.log')]})
 import numpy as np
 from scipy.spatial.transform import Rotation
 import omni.usd,omni.physx,omni.physics.tensors as tensors
 from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf,PhysicsSchemaTools
 from isaacsim.core.api import SimulationContext
 from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver,interface_config_loader
 from safety_monitor_v003 import SafetyMonitor
 monitor=SafetyMonitor()
 omni.usd.get_context().open_stage(str(P/'riser_io_single_wire_fastening_v003.usda'))
 for _ in range(8):app.update()
 stage=omni.usd.get_context().get_stage()
 # Supply light for evidence renders without changing collision geometry.
 from pxr import UsdLux
 stage.SetEditTarget(stage.GetSessionLayer());light=UsdLux.DomeLight.Define(stage,'/World/TrainingLight');light.CreateIntensityAttr(1500.)
 dt=1/480
 sim=SimulationContext(physics_dt=dt,rendering_dt=1/60,backend='numpy',device='cpu');sim.initialize_physics();sim.play()
 for _ in range(4):sim.step(render=True)
 view=tensors.create_simulation_view('numpy');view.set_subspace_roots('/')
 robot=view.create_articulation_view('/World/Robot');sensor=view.create_articulation_view('/World/CableSensor')
 assert robot.count==1 and sensor.count==1, (robot.count,sensor.count)
 seg=view.create_rigid_body_view('/wire1/Segment_*');rv=view.create_rigid_body_view('/wire1/_0347300_ring_terminal')
 assert seg.count==24 and rv.count==1,(seg.count,rv.count)
 names=list(robot.shared_metatype.dof_names);ln=list(sensor.shared_metatype.link_names);si=ln.index('Segment_00')
 report['sensor_links']=ln
 screw_path='/ScrewTray/Nest_1/Screw_1'
 sv=view.create_rigid_body_view(screw_path)
 assert sv.count==1
 screw_contacts=view.create_rigid_contact_view(screw_path,['/NutPlate_Coarse/Nut_1','/wire1/_0347300_ring_terminal','/PCB_4_Terminal'],1024)
 def screw_wrench():
  sp=sv.get_transforms()[0];f,p,n,d,c,st=screw_contacts.get_contact_data(dt)
  ff,fp,fc,fs=screw_contacts.get_friction_data(dt)
  total=np.zeros(3);moment=np.zeros(3);minimum=0.
  for count,start in zip(c.ravel(),st.ravel()):
   sl=slice(int(start),int(start+count))
   if count:
    forces=f[sl]*n[sl];total+=forces.sum(axis=0);moment+=np.cross(p[sl]-sp[:3],forces).sum(axis=0);minimum=min(minimum,float(d[sl].min()))
  for count,start in zip(fc.ravel(),fs.ravel()):
   sl=slice(int(start),int(start+count))
   if count:total+=ff[sl].sum(axis=0);moment+=np.cross(fp[sl]-sp[:3],ff[sl]).sum(axis=0)
  axis=Rotation.from_quat(sp[3:]).apply([0,0,1])
  return sp,total,moment,minimum,axis

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
  raise RuntimeError('IK unreachable: '+str(p)+' residual '+str(best.fun))
 target=np.array([-.092,-.071,.768]);Rdown=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
 q=robot.get_dof_positions()[0].copy();q[ai]=[0,-.5,0,-2.2,0,1.7,.7854];q[fi]=.04
 orient=Rotation.from_matrix(Rdown).as_quat()
 q[ai]=solve(np.array([-.24,.02,1.0434]),orient,q[ai])
 robot.set_dof_positions(q[None].astype(np.float32),ix);robot.set_dof_velocities(np.zeros_like(q)[None].astype(np.float32),ix);robot.set_dof_position_targets(q[None].astype(np.float32),ix)
 # Effort limits for physical contact grasp; position drives remain compliant.
 forces=robot.get_dof_max_forces().copy();forces[0,fi]=8.;robot.set_dof_max_forces(forces,ix)
 target=np.array([-.092,-.071,.768]);Rdown=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
 phase='reset';stepnum=0;contact={'left':0.,'right':0.,'seat':0.,'min_sep':0.,'forbidden':0.};maxforbidden=0.
 jp0=np.array([stage.GetPrimAtPath(f'/wire1/Bend_{i:02d}').GetAttribute('physics:localPos0').Get() for i in range(23)])
 jp1=np.array([stage.GetPrimAtPath(f'/wire1/Bend_{i:02d}').GetAttribute('physics:localPos1').Get() for i in range(23)])
 cv=view.create_rigid_contact_view('/World/ForceControl',['/World/Table'],256)
 def oncontacts(headers,data):
  for h in headers:
   a=str(PhysicsSchemaTools.intToSdfPath(h.collider0));b=str(PhysicsSchemaTools.intToSdfPath(h.collider1));pair=a+'|'+b
   for k in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
    c=data[k];imp=np.asarray(c.impulse,float);normal=np.asarray(c.normal,float);force=normal*float(imp)/dt if imp.ndim==0 else imp/dt;fn=abs(float(np.dot(force,normal)))
    ring='_0347300_ring_terminal' in pair
    screw=screw_path in pair
    if ring and 'panda_leftfinger' in pair:contact['left']+=fn
    if ring and 'panda_rightfinger' in pair:contact['right']+=fn
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
   contact={'left':0.,'right':0.,'seat':0.,'min_sep':0.,'forbidden':0.}
   sim.step(render=False);stepnum+=1
   if contact_errors:raise RuntimeError("contact measurement failed: "+str(contact_errors[-1]))
   transforms=seg.get_transforms().copy();xyz=transforms[:,:3];ring=rv.get_transforms()[0].copy();w=sensor.get_link_incoming_joint_force()[0,si].copy()
   joint_gap=float(np.linalg.norm(xyz[:-1]+Rotation.from_quat(transforms[:-1,3:]).apply(jp0)-xyz[1:]-Rotation.from_quat(transforms[1:,3:]).apply(jp1),axis=1).max())
   if not np.isfinite(xyz).all() or not np.isfinite(w).all():raise RuntimeError('nonfinite physics state')
   aa=xyz[1:-1]-xyz[:-2];bb=xyz[2:]-xyz[1:-1];cc=xyz[2:]-xyz[:-2];den=2*np.linalg.norm(np.cross(aa,bb),axis=1)
   radii=np.linalg.norm(aa,axis=1)*np.linalg.norm(bb,axis=1)*np.linalg.norm(cc,axis=1)/np.maximum(den,1e-12)
   radius=float(radii.min());force=float(np.linalg.norm(w[:3]));moment=float(np.linalg.norm(w[3:]));maxforbidden=max(maxforbidden,contact['forbidden'])
   rec={'time_s':stepnum*dt,'phase':phase,'attachment_force_N':force,'attachment_moment_Nm':moment,'min_bend_radius_m':radius,'min_contact_separation_m':contact['min_sep'],'finger_left_N':contact['left'],'finger_right_N':contact['right'],'seat_axial_N':contact['seat'],'forbidden_robot_contact_N':contact['forbidden'],'ring_x':float(ring[0]),'ring_y':float(ring[1]),'ring_z':float(ring[2]),'ring_tilt_deg':float(np.degrees(np.arccos(np.clip(Rotation.from_quat(ring[3:]).as_matrix()[2,2],-1,1)))),'control_box_N':float(np.linalg.norm(cv.get_net_contact_forces(dt)[0]))}
   sp,sf,sm,ssep,saxis=screw_wrench()
   rec.update(screw_x=float(sp[0]),screw_y=float(sp[1]),screw_z=float(sp[2]),screw_tilt_deg=float(np.degrees(np.arccos(np.clip(saxis[2],-1,1)))),screw_axial_N=float(abs(sf[2])),screw_lateral_N=float(np.linalg.norm(sf[:2])),screw_torque_Nm=float(abs(np.dot(sm,saxis))),screw_bending_Nm=float(np.linalg.norm(sm-np.dot(sm,saxis)*saxis)),screw_speed_m_s=float(np.linalg.norm(sv.get_velocities()[0,:3])))
   rec['attachment_axial_N']=abs(float(w[0]));rec['attachment_lateral_N']=float(np.linalg.norm(w[1:3]));rec['attachment_bending_Nm']=float(np.linalg.norm(w[4:]))
   rec['min_contact_separation_m']=min(rec['min_contact_separation_m'],ssep)
   rec['deepest_pair']=contact.get('deepest_pair','');rec['max_joint_anchor_gap_m']=joint_gap;rec['tightest_bend_segment']=int(np.argmin(radii)+1);rec['ring_speed_m_s']=float(np.linalg.norm(rv.get_velocities()[0,:3]))
   rec['safety_state']=monitor.update(rec,dt) if phase not in ['settle','sensor_calibration','reset'] else 'INITIALIZATION';records.append(rec)
   if force>100 or np.max(np.linalg.norm(xyz-xyz[0],axis=1))>.8:raise RuntimeError('numerical or structural instability')
   if args.record and stepnum%24==0:write_video_frame(rec)
   if stepnum%960==0:print('STATE',json.dumps(rec),flush=True)
 def setup_render():
  import omni.replicator.core as rep
  if not hasattr(setup_render,'annot'):
   cam=UsdGeom.Camera.Define(stage,'/World/TrainingCamera');cam.CreateFocalLengthAttr(22.);cam.CreateClippingRangeAttr(Gf.Vec2f(.01,10.))
   camera_pose=Gf.Matrix4d().SetLookAt(Gf.Vec3d(.13,-.55,1.24),Gf.Vec3d(-.16,.035,.785),Gf.Vec3d(0,0,1)).GetInverse()
   UsdGeom.Xformable(cam).AddTransformOp().Set(camera_pose)
   prod=rep.create.render_product(str(cam.GetPath()),(960,720));setup_render.annot=rep.AnnotatorRegistry.get_annotator('rgb');setup_render.annot.attach([prod])
  return setup_render.annot
 def write_video_frame(rec):
  global video_writer
  import cv2
  annot=setup_render()
  for _ in range(2):sim.render()
  a=annot.get_data()
  if a is None or not a.size:return
  frame=cv2.cvtColor(a[:,:,:3],cv2.COLOR_RGB2BGR)
  if video_writer is None:
   video_writer=cv2.VideoWriter(str(out/'rollout.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20.,(960,720))
   if not video_writer.isOpened():raise RuntimeError('MP4 encoder unavailable')
  violated=[k for k,v in monitor.stats.items() if v['duration_s']>0]
  color=(40,60,255) if violated else (60,220,80)
  cv2.rectangle(frame,(0,0),(960,120),(20,20,20),-1)
  lines=[f"{rec['time_s']:6.2f}s  {phase}  "+('EPISODE VIOLATED' if violated else 'WITHIN CANDIDATE LIMITS'),
   f"Cable F {rec['attachment_force_N']:.2f} N | Rmin {1000*rec['min_bend_radius_m']:.1f} mm | joint M {rec['attachment_bending_Nm']:.3f} Nm",
   f"Screw tilt {rec['screw_tilt_deg']:.1f} deg | axial {rec['screw_axial_N']:.2f} N | torque {rec['screw_torque_Nm']:.4f} Nm",
   'Limits: prototype operating bounds; material damage NOT calibrated']
  for i,line in enumerate(lines):cv2.putText(frame,line,(12,24+27*i),cv2.FONT_HERSHEY_SIMPLEX,.55,color if i==0 else (240,240,240),1,cv2.LINE_AA)
  if violated:cv2.putText(frame,', '.join(violated),(12,700),cv2.FONT_HERSHEY_SIMPLEX,.5,(40,60,255),1,cv2.LINE_AA)
  video_writer.write(frame);frame_log.append({'frame':len(frame_log),'time_s':rec['time_s'],'phase':phase})
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
   if stop_force is not None and records[-1]['attachment_force_N']>=stop_force:
    report['pull_stopped_force_N']=records[-1]['attachment_force_N'];return
  tick(120)
  hp,hr=handpose();err=np.linalg.norm(np.asarray(hp)-end)
  if err>.015:raise RuntimeError('hand tracking error '+str(err))
 def grip(width,seconds=1.5):
  global q
  start=q[fi].copy();steps=int(seconds*60)
  for k in range(steps):q[fi]=start+(width/2-start)*(k+1)/steps;robot.set_dof_position_targets(q[None].astype(np.float32),ix);tick(8)
 phase='settle';tick(960);capture('01_settled')
 report['settled_ring_pose']=rv.get_transforms()[0].tolist()
 # Known load on the fixed sensor link validates reaction-force scale independently of contact sensors.
 phase='sensor_calibration';tick(60);w0=sensor.get_link_incoming_joint_force()[0,si,:3].copy()
 calibration_force=np.zeros((1,sensor.max_links,3),np.float32);calibration_force[0,si,0]=.25
 for _ in range(120):sensor.apply_forces_and_torques_at_position(calibration_force,None,None,ix,True);tick()
 w1=sensor.get_link_incoming_joint_force()[0,si,:3].copy()
 report['attachment_sensor_known_load_N']=.25;report['attachment_sensor_increment_N']=float(np.linalg.norm(w1-w0))
 report['attachment_sensor_calibration_pass']=abs(report['attachment_sensor_increment_N']-.25)<.025
 assert report['attachment_sensor_calibration_pass'],'attachment sensor calibration failed'
 phase='settle';tick(120)
 reset_monitor=SafetyMonitor()
 for row in records[-120:]:reset_monitor.update(row,dt)
 report['reset_safety']=reset_monitor.summary()
 assert report['reset_safety']['safety_status']=='SAFE','settled reset is not safe'
 if not args.settle_only:
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
  report['release_hold_s']=3.
  if not args.wire_only:
   Rdown=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
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
    tilted=Rotation.from_euler('x',18,degrees=True).as_matrix()@Rdown
    phase='screw_slanted_loading';move(np.array([target[0],target[1],.765])-offset,4.,rotation=tilted);tick(480);capture('10_screw_slanted_loading')
    phase='screw_backoff';move(np.array([target[0],target[1],.81])-offset,3.)
   phase='screw_thread_entry';move(np.array([target[0],target[1],.7645])-offset,4.);capture('11_thread_entry')
   # Each half-turn is transmitted through actual finger/head and helical contact.
   # Regrasp above the seat keeps the robot wrist within its range.
   turns=[]
   for turn in range(14):
    hp,hr=handpose();tcp=np.asarray(hp)+np.asarray(hr)@np.array([0,0,.1034])
    phase='screw_turn_'+str(turn)
    goalrot=Rotation.from_euler('z',-150,degrees=True).as_matrix()@Rdown
    move(tcp+[0,0,-.002*150/360],3.,rotation=goalrot)
    pose=sv.get_transforms()[0].copy();turns.append({'turn':turn,'tip_z':float(pose[2]),'torque_Nm':records[-1]['screw_torque_Nm']})
    print('SCREW_TURN',turns[-1],flush=True)
    if pose[2]<.7536:break
    phase='screw_regrasp_open';grip(.035,.6)
    phase='screw_regrasp_return';move(tcp+[0,0,-.002*150/360],1.5,rotation=Rdown)
    phase='screw_regrasp_close';grip(0.,.6)
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
   report['task_success']=bool(report['screw_fastened'] and report['ring_retained'])

 report['physics_valid']=True
except Exception:
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
  report['safety_violations']=report['online_safety_evaluation']['violations']
  report['safety_status']=report['online_safety_evaluation']['safety_status'] if report['physics_valid'] else 'UNKNOWN'
  report['safe_success']=report['task_success'] and report['physics_valid'] and report['safety_status']=='SAFE'
 (out/'events.json').write_text(json.dumps(events,indent=2));(out/'result.json').write_text(json.dumps(report,indent=2));print('RESULT',json.dumps(report),flush=True)
 if app:app.close()
