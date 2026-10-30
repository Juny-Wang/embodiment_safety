from pathlib import Path
import json,hashlib,difflib,zipfile,subprocess
import numpy as np
from pxr import Usd,UsdGeom
P=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/wire_guides_full_v001')
cfg=json.loads((P/'config.json').read_text());source=Path(cfg['source_scene']);copy=Path(cfg['scene'])
source_hash=hashlib.sha256(source.read_bytes()).hexdigest();assert source_hash==cfg['source_sha256']
a=Usd.Stage.Open(str(source));b=Usd.Stage.Open(str(copy))
def sig(s):
 out={}
 for p in Usd.PrimRange(s.GetPrimAtPath('/World/Robot')):
  out[str(p.GetPath())]={'type':p.GetTypeName(),'schemas':p.GetAppliedSchemas(),'attrs':{a.GetName():str(a.Get()) for a in p.GetAttributes()},'rels':{r.GetName():list(map(str,r.GetTargets())) for r in p.GetRelationships()}}
 return out
assert sig(a)==sig(b)
xa=UsdGeom.XformCache();xb=UsdGeom.XformCache();errors=[]
for w in range(1,5):
 for i in range(24):
  path=f'/wire{w}/Segment_{i:02d}';pa=a.GetPrimAtPath(path);pb=b.GetPrimAtPath(path)
  errors.append((xa.GetLocalToWorldTransform(pa).ExtractTranslation()-xb.GetLocalToWorldTransform(pb).ExtractTranslation()).GetLength()*1000)
assert max(errors)<1e-6
smoke=json.loads((P/'reports/full_smoke.json').read_text())
printed=json.loads((P/'reports/print_geometry_final.json').read_text())
report={'source_unchanged':True,'source_sha256':source_hash,'scene_copy':str(copy),'copy_sha256':hashlib.sha256(copy.read_bytes()).hexdigest(),'robot_composed_properties_unchanged':True,'wire_initial_center_max_error_mm':max(errors),'source_chain_retained':True,'rejected_articulation_changes_promoted':False,'static_scene_copy_ready':True,'native_passive_smoke_pass':smoke.get('full_scene_passive_smoke_pass',False),'contact_and_bend_monitor_ready':smoke.get('full_scene_passive_smoke_pass',False),'contact_force_positive_control_N':smoke.get('control_mean_last_120_N'),'internal_load_measurement_ready':False,'wire_material_stress_limit_ready':False,'manual_Franka_routing_test_passed':False,'full_task_ready':False,'all_print_STLs_watertight':all(x['watertight'] for x in printed.values()),'print_STL_types':len(printed),'physical_print_assembly_validated':False,'Franka_and_table':'External experiment platform, not printed robot or furniture','monitor_scope':'Measured guide normal contact force and discrete bending only; GREEN is not material safety certification'}
(P/'reports/acceptance.json').write_text(json.dumps(report,indent=2))
# Exact copy diff is meaningful even though the task directory is untracked in Git.
patch=''.join(difflib.unified_diff(source.read_text().splitlines(True),copy.read_text().splitlines(True),fromfile=str(source),tofile=str(copy),n=2))
(P/'reports/scene_changes.patch').write_text(patch)
(P/'reports/diff_stat.json').write_text(json.dumps({'source':str(source),'copy':str(copy),'added_lines':sum(l.startswith('+') and not l.startswith('+++') for l in patch.splitlines()),'removed_lines':sum(l.startswith('-') and not l.startswith('---') for l in patch.splitlines()),'git_note':'Task directory untracked; tracked git diff does not include newly created files.'},indent=2))
for name,args in [('git_status_after.txt',['git','status','--short']),('git_diff_stat.txt',['git','diff','--stat','--',str(copy),str(P)]),('git_diff.txt',['git','diff','--',str(copy),str(P)])]:
 (P/'reports'/name).write_text(subprocess.run(args,capture_output=True,text=True,check=True).stdout)
# Archive only source/report/printing artifacts; no caches and no data pretending to be a standalone complete USD dependency bundle.
files=[p for p in P.rglob('*') if p.is_file() and not any(k in p.relative_to(P).parts for k in ['runtime','runs','__pycache__']) and p.suffix not in ['.zip'] and p.name not in ['SHA256SUMS.txt','file_manifest.json','PACKAGE_SHA256.txt']]
manifest=[{'path':str(p),'relative':str(p.relative_to(P)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(files)]
manifest.append({'path':str(copy),'relative':'../'+copy.name,'sha256':report['copy_sha256'],'bytes':copy.stat().st_size})
(P/'file_manifest.json').write_text(json.dumps(manifest,indent=2))
(P/'SHA256SUMS.txt').write_text('\n'.join(x['sha256']+'  '+x['relative'] for x in manifest)+'\n')
z=P/'Task2_print_parts_v001.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as archive:
 for p in sorted((P/'stl').glob('*.stl')):archive.write(p,str(p.relative_to(P)))
 for n in ['README.md','BOM.json','assembly_layout.png','reports/print_geometry_final.json','reports/print_repairs.json','reports/static_audit.json','reports/acceptance.json']:archive.write(P/n,n)
(P/'PACKAGE_SHA256.txt').write_text(hashlib.sha256(z.read_bytes()).hexdigest()+'  '+z.name+'\n')
print(json.dumps(report,indent=2));print('PRINT_ZIP',str(z));print('DIFF_STAT',(P/'reports/diff_stat.json').read_text())
