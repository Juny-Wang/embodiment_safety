"""Run only this isolated scene. No main scene, no Franka, no training."""
import argparse,json,sys,traceback,datetime,csv,math
from pathlib import Path
P=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser();parser.add_argument('--scene',choices=['L0_sensor_bench','L1_one_guide','L2_two_guides'],default='L0_sensor_bench');parser.add_argument('--force',type=float,default=.25);parser.add_argument('--direction',choices=['x','y'],default='x');parser.add_argument('--controlled-pull',action='store_true');args,unknown=parser.parse_known_args()
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
out=P/'runs'/f'{stamp}_{args.scene}_{args.direction}_{args.force:g}N';out.mkdir()
report={'scene':args.scene,'requested_force_N':args.force,'direction':args.direction,'physics_acceptance':False,'material_calibrated':False,'source_main_modified':False,'controlled_pull':args.controlled_pull,'actuation':'force-limited external test instrument; no body pose writes'}
app=None
try:
 import os
 for n in ['cache','data','config','tmp']: (P/'runtime'/n).mkdir(parents=True,exist_ok=True)
 for key,n in [('XDG_CACHE_HOME','cache'),('XDG_DATA_HOME','data'),('XDG_CONFIG_HOME','config'),('TMPDIR','tmp')]:os.environ[key]=str(P/'runtime'/n)
 sys.argv=[sys.argv[0]]
 from isaacsim import SimulationApp
 app=SimulationApp({'headless':True,'width':800,'height':600,'sync_loads':True,'extra_args':['--portable','--portable-root',str(P/'runtime'),'--/log/file='+str(out/'kit.log')]})
 import numpy as np
 import omni.usd,omni.physx,omni.physics.tensors as tensors
 from pxr import UsdGeom,PhysicsSchemaTools
 from isaacsim.core.api import SimulationContext
 sys.path.insert(0,str(P/'scripts'));from metrics import summarize
 omni.usd.get_context().open_stage(str(P/(args.scene+'.usda')))
 for _ in range(5):app.update()
 sim=SimulationContext(physics_dt=1/960,rendering_dt=1/60,backend='numpy',device='cpu')
 sim.initialize_physics();sim.play()
 for _ in range(10):sim.step(render=True)
 assert sim.is_playing(), 'physics timeline is not playing'
 print('SIM_READY',sim.current_time,flush=True)
 view=tensors.create_simulation_view('numpy');view.set_subspace_roots('/')
 art=view.create_articulation_view('/World/Cable')
 names=list(art.shared_metatype.link_names)
 paths=['/World/Cable/'+f'Segment_{i:02d}' for i in range(24)]
 ids=[names.index('Segment_'+f'{i:02d}') for i in range(24)]
 report['link_names']=names;report['link_count']=art.max_links
 assert art.count==1 and len(ids)==24
 cv=view.create_rigid_contact_view('/World/ControlBox',['/World/Table'],256)
 guide_paths=['/World/Guides/Post_1']+(['/World/Guides/Post_2'] if args.scene=='L2_two_guides' else [])
 gv=view.create_rigid_contact_view('/World/Cable/Segment_*',guide_paths,2048) if args.scene!='L0_sensor_bench' else None
 dt=1/960;contact={'control_N':0.,'guide_N':0.,'points':[]}
 def on_contacts(headers,data):
  try:
   for h in headers:
    a=str(PhysicsSchemaTools.intToSdfPath(h.collider0));b=str(PhysicsSchemaTools.intToSdfPath(h.collider1))
    for k in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
     c=data[k];n=np.asarray(c.normal,float);imp=np.asarray(c.impulse,float);f=n*float(imp)/dt if imp.ndim==0 else imp/dt;fn=abs(float(np.dot(f,n)))
     if 'ControlBox' in a+b:contact['control_N']+=fn
     if '/Guides/' in a+b:
      contact['guide_N']+=fn
      contact['points'].append({'point_m':[float(x) for x in c.position],'normal_force_N':fn,'normal':n.tolist(),'a':a,'b':b})
  except Exception as e:report['contact_error']=repr(e)
 sub=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(on_contacts)
 records=[];frames=[];indices=np.array([0],dtype=np.uint32)
 start_pos=art.get_link_transforms()[0,ids[-1],:3].copy()
 previous_pos=start_pos.copy()
 total_steps=9600 if args.controlled_pull else 2400
 load_end=8.5 if args.controlled_pull else 2.
 for step in range(total_steps):
  t=step*dt;state='SETTLE' if t<.5 else 'LOAD' if t<load_end else 'RELEASE'
  ramp=min(1.,max(0.,(t-.5)/.5)) if state=='LOAD' else 0.
  force=np.zeros((1,art.max_links,3),np.float32);force[0,ids[-1],0 if args.direction=='x' else 1]=args.force*ramp
  if args.controlled_pull and state=='LOAD':
   current=art.get_link_transforms()[0,ids[-1],:3].copy()
   velocity=(current-previous_pos)/dt
   target=start_pos.copy();target[0 if args.direction=='x' else 1]+=.01*max(0.,t-.5)
   applied=20.*(target-current)-1.0*velocity
   norm=np.linalg.norm(applied)
   if norm>args.force:applied*=args.force/norm
   force[0,ids[-1],:]=applied
   previous_pos=current
  applied_norm=float(np.linalg.norm(force[0,ids[-1],:]))
  art.apply_forces_and_torques_at_position(force,None,None,indices,True)
  contact={'control_N':0.,'guide_N':0.,'points':[]}
  sim.step(render=False)
  pos=art.get_link_transforms()[0,ids,:3].copy()
  all_wrench=art.get_link_incoming_joint_force()[0].copy()
  wrench=all_wrench[ids[0],:].copy()
  contact['control_N']=float(np.linalg.norm(cv.get_net_contact_forces(dt)[0]))
  if gv is not None:
   force_buffer,point_buffer,normal_buffer,separation_buffer,counts,starts=gv.get_contact_data(dt)
   contact['guide_N']=0.;contact['points']=[]
   for ind in np.ndindex(counts.shape):
    for k in range(int(starts[ind]),int(starts[ind]+counts[ind])):
     fn=float(np.asarray(force_buffer[k]).reshape(-1)[0]);contact['guide_N']+=abs(fn)
     contact['points'].append({'point_m':point_buffer[k].tolist(),'normal_force_N':fn,'normal':normal_buffer[k].tolist(),'separation_m':float(np.asarray(separation_buffer[k]).reshape(-1)[0])})
  if step%120==0: print('STEP',step,'simtime',sim.current_time,'joint01_N',float(np.linalg.norm(wrench[:3])),'box_N',contact['control_N'],flush=True)
  if not np.isfinite(pos).all() or not np.isfinite(wrench).all() or np.max(np.linalg.norm(pos-pos[0],axis=1))>1. or np.linalg.norm(wrench[:3])>100:
   report['instability']={'step':step,'time_s':t,'reason':'nonfinite_or_nonphysical_diagnostic_state','not_a_material_damage_event':True}
   break
  m=summarize(pos,wrench)
  records.append({'time_s':t,'phase':state,'applied_force_N':applied_norm,'connector_force_N':m['connector_force_N'],'connector_moment_Nm':m['connector_moment_Nm'],'min_bend_radius_m':m['min_bend_radius_m'],'slack_m':m['geometric_slack_m'],'guide_normal_N':contact['guide_N'],'guide_contact_count':len(contact['points']),'guide_min_separation_m':min([p.get('separation_m',0) for p in contact['points']],default=0),'control_box_N':contact['control_N'],'state':m['state'],'end_x_m':float(pos[-1,0]),'end_y_m':float(pos[-1,1]),'end_z_m':float(pos[-1,2]),'reasons':'|'.join(m['reasons'])})
  if step%12==0:frames.append({'time_s':t,'centers_m':pos.tolist(),'anchor_wrench_child_joint_frame':wrench.tolist(),'contact_points':list(contact['points']),'metrics':m})
 assert records, 'No valid physics frames'
 with (out/'force_curvature.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
 (out/'frames.json').write_text(json.dumps(frames,indent=2,allow_nan=False))
 steady=[r for r in records if 1.5<=r['time_s']<2.]
 positive=[r['control_box_N'] for r in records if .5<r['time_s']<2]
 report.update({'frames':len(records),'control_box_mean_N':float(np.mean(positive)) if positive else 0.,'expected_box_weight_N':.981,'connector_force_loaded_mean_N':float(np.mean([r['connector_force_N'] for r in steady])) if steady else None,'connector_force_peak_N':max(r['connector_force_N'] for r in records),'guide_normal_peak_N':max(r['guide_normal_N'] for r in records),'endpoint_displacement_m':float(np.linalg.norm(np.asarray(frames[-1]['centers_m'][-1])-frames[0]['centers_m'][-1])),'force_reader':'PhysX incoming reaction at Segment_00 / CableAttachment; diagnostic connector attachment reaction','contact_reader':'PhysX rigid contact tensor / fixed physics dt; callback points supplemental','no_runtime_pose_or_joint_writes':True})
 report['root_reaction_available']=False
 report['connector_surrogate_only']=True
 report['connector_attachment_reaction_available']=True
 report['final_sim_time_s']=float(sim.current_time)
 report['positive_control_pass']=.7<report['control_box_mean_N']<1.25
 report['reaction_measurement_nonzero']=report['connector_force_loaded_mean_N'] is not None and report['connector_force_loaded_mean_N']>.01
 report['physics_acceptance']=bool(report['positive_control_pass'] and report['reaction_measurement_nonzero'] and 'contact_error' not in report and 'instability' not in report and len(records)==total_steps)
 report['full_task_ready']=False
 report['limitations']=['Rigid-link chain, not calibrated elasticity or internal material stress.','Direct distal body force is sensor/load-path diagnosis, not Franka contact execution.','No claim of completed routing or strain relief.','No friction-specific tangential sensor validated.']
 sim.stop()
except Exception:
 report['error']=traceback.format_exc();print(report['error'],flush=True)
finally:
 (out/'physics_acceptance.json').write_text(json.dumps(report,indent=2))
 print('RESULT_DIRECTORY',out,flush=True);print(json.dumps(report),flush=True)
 if app:app.close()
