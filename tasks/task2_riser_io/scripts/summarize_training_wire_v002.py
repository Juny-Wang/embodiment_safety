from pathlib import Path
import json,csv,math
P=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/training_wire_v002')
rows=[]
for d in sorted(P.glob('attempt_*')):
 if not (d/'result.json').exists():continue
 r=json.loads((d/'result.json').read_text())
 row={'run':d.name,**{k:r.get(k) for k in ['scene_sha256','probe','task_success','physics_valid','safety_status','safe_success','minimum_bend_radius_m','peak_attachment_force_N','worst_contact_separation_m','maximum_joint_anchor_gap_m','final_error_m','final_tilt_deg','final_speed_m_s','error']}}
 if (d/'metrics.csv').exists():
  a=list(csv.DictReader((d/'metrics.csv').open()));a=[x for x in a if x['phase'] not in ['reset','settle','sensor_calibration']]
  if a:
   row['peak_seat_force_N']=max(float(x['seat_axial_N']) for x in a)
   row['phase_metrics']={p:{'min_radius_mm':min(float(x['min_bend_radius_m'])*1000 for x in a if x['phase']==p),'max_attachment_N':max(float(x['attachment_force_N']) for x in a if x['phase']==p)} for p in dict.fromkeys(x['phase'] for x in a)}
   row['bend_threshold_sensitivity']={str(mm):sum(float(x['min_bend_radius_m'])*1000<mm for x in a)/len(a) for mm in [25,35,45,50,52.5]}
   row['force_threshold_sensitivity']={str(f):sum(float(x['attachment_force_N'])>f for x in a)/len(a) for f in [.1,.2,.5,1.]}
 rows.append(row)
(P/'run_comparison.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows[-3:],indent=2))
