from pathlib import Path
import json,csv,math,hashlib,ast,sys,zipfile
import numpy as np,trimesh as tm
from pxr import Usd,UsdGeom,UsdPhysics,Gf
from PIL import Image,ImageDraw,ImageFont
O=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/wire_safety_v001')
source=json.loads((O/'reports/source_audit.json').read_text())
assert hashlib.sha256(Path(source['source']).read_bytes()).hexdigest()==source['sha256'],'Main stage changed'
for p in (O/'scripts').glob('*.py'):ast.parse(p.read_text())
sys.path.insert(0,str(O/'scripts'))
from metrics import curvature,summarize
theta=np.linspace(0,1.5,50);arc=np.c_[.1*np.cos(theta),.1*np.sin(theta),np.zeros(50)]
assert abs(curvature(arc)['min_bend_radius_m']-.1)<1e-9
assert summarize(arc,None)['state']=='GRAY'
assert summarize(arc,[1.5,0,0,0,0,0])['state']=='RED'
assert summarize(arc,[.25,0,0,0,0,0])['state']=='GREEN'
static={}
for p in O.glob('*.usda'):
 st=Usd.Stage.Open(str(p));assert not st.GetCompositionErrors()
 xc=UsdGeom.XformCache();errors=[];qn=[]
 for prim in st.Traverse():
  if prim.IsA(UsdPhysics.Joint):
   j=UsdPhysics.Joint(prim);b0=j.GetBody0Rel().GetTargets();b1=j.GetBody1Rel().GetTargets()
   a=Gf.Vec3d(j.GetLocalPos0Attr().Get());b=Gf.Vec3d(j.GetLocalPos1Attr().Get())
   if b0:a=xc.GetLocalToWorldTransform(st.GetPrimAtPath(b0[0])).Transform(a)
   if b1:b=xc.GetLocalToWorldTransform(st.GetPrimAtPath(b1[0])).Transform(b)
   errors.append((a-b).GetLength()*1000);qn.extend([j.GetLocalRot0Attr().Get().GetLength(),j.GetLocalRot1Attr().Get().GetLength()])
 static[p.name]={'composition_errors':[],'joint_anchor_error_max_mm':max(errors,default=0),'quaternion_norm_error_max':max([abs(x-1) for x in qn],default=0)}
 assert max(errors,default=0)<.001 and max([abs(x-1) for x in qn],default=0)<1e-6
runs=[];latest={}
for p in sorted((O/'runs').glob('*/physics_acceptance.json')):
 d=json.loads(p.read_text());d['report_path']=str(p);runs.append(d)
 if d.get('frames',0)>=2400 and d.get('connector_attachment_reaction_available'):
  rows=list(csv.DictReader((p.parent/'force_curvature.csv').open()))
  d['red_frames']=sum(r['state']=='RED' for r in rows);d['yellow_frames']=sum(r['state']=='YELLOW' for r in rows)
  d['guide_min_separation_mm']=min([float(r.get('guide_min_separation_m',0))*1000 for r in rows],default=0)
  d['contact_point_frames']=sum(int(r.get('guide_contact_count',0))>0 for r in rows)
  d['minimum_bend_radius_mm']=min([float(r['min_bend_radius_m'])*1000 for r in rows if r['min_bend_radius_m']],default=None)
  latest[(d['scene'],d['requested_force_N'],d['direction'])]=d
fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=lambda n:ImageFont.truetype(fontpath,n)
normal=latest.get(('L0_sensor_bench',.25,'x'));high=latest.get(('L0_sensor_bench',1.5,'x'))
# Plot only measurements, never replace failed data with a desired curve.
selected=[d for d in [normal,high] if d and d['physics_acceptance']]
im=Image.new('RGB',(1300,820),'#f5f7fa');dr=ImageDraw.Draw(im)
dr.text((40,25),'MEASURED PHYSX RESPONSE / ISOLATED CABLE',font=font(27),fill='#16334c')
dr.text((40,68),'Low load vs demonstration over-limit | not a material strength / damage test',font=font(20),fill='#9c482e')
for i,d in enumerate(selected):
 rows=list(csv.DictReader((Path(d['report_path']).parent/'force_curvature.csv').open()))
 y0=140+i*310;left=100;right=1220;bottom=y0+230;top=y0+35;maximum=1.8
 dr.text((left,y0),f"Applied {d['requested_force_N']:.2f} N; measured mean {d['connector_force_loaded_mean_N']:.3f} N",font=font(21),fill='#16334c')
 def xy(t,v):return (left+(right-left)*t/2.5,bottom-(bottom-top)*v/maximum)
 for val in [0,.5,1.,1.5]:
  y=xy(0,val)[1];dr.line([(left,y),(right,y)],fill='#d9dfe5');dr.text((42,y-10),str(val),font=font(16),fill='black')
 dr.line([xy(0,1),xy(2.5,1)],fill='#d94836',width=2)
 for key,col in [('applied_force_N','#6b737b'),('connector_force_N','#207db0')]:
  pts=[xy(float(r['time_s']),float(r[key])) for r in rows if r[key]]
  dr.line(pts,fill=col,width=3)
 for tick in [0,.5,1.,1.5,2.,2.5]:
  xx=xy(tick,0)[0];dr.line([(xx,bottom),(xx,bottom+5)],fill='#45596c');dr.text((xx-10,bottom+8),str(tick),font=font(16),fill='#45596c')
 dr.text((right-80,bottom+32),'time (s)',font=font(16),fill='#45596c')
 dr.text((42,top-22),'N',font=font(16),fill='#45596c')
dr.text((70,775),'Blue: measured connector attachment force | Grey: applied distal load | Red: 1 N DEMO threshold',font=font(18),fill='#16334c')
im.save(O/'force_comparison.png')
# 2D measured pose replay, explicitly not rendered Kit video or damage animation.
if len(selected)==2:
 fs=[json.loads((Path(d['report_path']).parent/'frames.json').read_text()) for d in selected]
 pictures=[]
 for k in range(0,min(map(len,fs)),2):
  im=Image.new('RGB',(1100,620),'#f5f7fa');dr=ImageDraw.Draw(im)
  dr.text((25,20),'ACTUAL PHYSX SAMPLES - 2D POSE / LOAD REPLAY',font=font(23),fill='#16334c')
  for i,frames in enumerate(fs):
   frame=frames[k];m=frame['metrics'];y=155+i*250
   color={'GREEN':'#209b52','YELLOW':'#d9a320','RED':'#db403b','GRAY':'#888888'}[m['state']]
   dr.text((35,y-65),f"{selected[i]['requested_force_N']} N distal load | t={frame['time_s']:.2f}s | {m['state']}",font=font(21),fill=color)
   pts=[(60+v[0]*2100,y-v[1]*2100) for v in frame['centers_m']]
   dr.line(pts,fill=color,width=11)
   dr.rectangle((30,y-18,60,y+18),fill='#64788b')
   dr.text((35,y+36),f"Connector: {m['connector_force_N']:.3f} N | stress: unavailable | damage: not modeled",font=font(18),fill='#16334c')
  dr.text((35,580),'Identical 960 Hz solver settings. Color follows measured force/curvature, not animation time.',font=font(17),fill='#8b482f')
  pictures.append(im)
 pictures[0].save(O/'measured_comparison.gif',save_all=True,append_images=pictures[1:],duration=50,loop=0)
 pictures[len(pictures)//2].save(O/'measured_comparison.png')
# CAD geometry preview from printable STL, no generated illustration.
items=[]
def add(n,p,c):
 m=tm.load_mesh(O/'stl'/(n+'.stl'));m.apply_translation(p);items.append((m,np.asarray(c)))
add('mount_tile_160x120',[0,0,0],(85,103,124));add('guide_column_D36_H35',[0,0,6],(188,149,92))
add('relief_collar_lower_candidate',[125,0,0],(74,144,161));add('relief_collar_upper_candidate',[125,0,30],(87,169,178))
ang=math.radians(25);tilt=math.radians(-57);rz=np.array([[math.cos(ang),-math.sin(ang),0],[math.sin(ang),math.cos(ang),0],[0,0,1]]);rx=np.array([[1,0,0],[0,math.cos(tilt),-math.sin(tilt)],[0,math.sin(tilt),math.cos(tilt)]]);rot=rx@rz
vs=np.vstack([m.vertices@rot.T for m,c in items]);lo=vs[:,:2].min(0);hi=vs[:,:2].max(0);scale=min(1200/(hi[0]-lo[0]),600/(hi[1]-lo[1]));center=(lo+hi)/2
im=Image.new('RGB',(1400,850),'#f4f7fa');dr=ImageDraw.Draw(im);polys=[]
for m,c in items:
 v=m.vertices@rot.T;n=m.face_normals@rot.T
 for k in np.where(n[:,2]>0)[0]:
  pts=v[m.faces[k]];uv=(pts[:,:2]-center)*[scale,-scale]+[700,440];polys.append((float(pts[:,2].mean()),[tuple(x) for x in uv],tuple((c*(.55+.45*abs(n[k,2]))).astype(int))))
for depth,pts,c in sorted(polys,key=lambda p:p[0]):dr.polygon(pts,fill=c)
dr.text((35,25),'PRINTABLE FIXTURE CANDIDATES / ACTUAL STL',font=font(27),fill='#16334c')
dr.text((35,73),'160 x 120 mm mounting tile | D36 guide | split collar, 7.4 mm bore for 7 mm cable',font=font(19),fill='#16334c')
dr.text((35,787),'Collar is a locator candidate: preload, compliance and strain-relief performance are NOT validated.',font=font(18),fill='#a44530')
im.save(O/'printable_fixture.png')
guide_results=[d for d in latest.values() if d['scene']!='L0_sensor_bench']
report={'source_scene_unchanged':True,'static_checks':static,'metric_unit_tests_pass':True,
 'sensor_diagnostic_ready':bool(normal and high and normal['physics_acceptance'] and high['physics_acceptance']),
 'measured_force_color_comparison_ready':bool(normal and high and normal.get('red_frames',1)==0 and high.get('red_frames',0)>0 and high['physics_acceptance']),
 'guide_contact_diagnostic_ready':len(guide_results)==2 and all(d['physics_acceptance'] and d.get('contact_point_frames',0)>0 and d.get('guide_min_separation_mm',-999)>=-.5 for d in guide_results),
 'strain_relief_ready':False,'material_stress_ready':False,'source_wire_ready':False,'full_task_ready':False,'promoted_to_main':False,
 'gui_manual_smoke_verified':False,'latest_valid_record_sets':list(latest.values()),'all_run_reports':[d['report_path'] for d in runs],
 'blocked_items':['Original four wire joint-frame errors remain in main, unchanged.','Relief collar has 0.2 mm radial clearance, no validated preload/compliance or axial retention.','No material calibration, axial constitutive law, true stress or damage model.','No actual ring routing/Franka execution or four-wire collision regression.','Printed tile anchoring and screw/nut fit need physical assembly test.'],
 'thresholds':{'source':'DEMONSTRATION_ONLY','connector_force_N':1.,'minimum_bend_radius_mm':35.,'guide_contact_quality_penetration_limit_mm':.5}}
(O/'reports/acceptance.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in report.items() if k not in ['latest_valid_record_sets','all_run_reports']},indent=2))
print('LATEST_CASES',json.dumps([{k:d.get(k) for k in ['scene','requested_force_N','physics_acceptance','connector_force_loaded_mean_N','guide_normal_peak_N','guide_min_separation_mm','contact_point_frames','red_frames']} for d in latest.values()]))
