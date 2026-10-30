from pathlib import Path
import json,hashlib,zipfile,os,difflib,ast
import numpy as np
from PIL import Image,ImageDraw,ImageFont
O=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/wire_safety_v001')
p=O/'reports/acceptance.json';r=json.loads(p.read_text());r['monitor_smoke']=json.loads((O/'reports/monitor_smoke.json').read_text());r['two_guide_S_route_ready']=False;r['two_guides_simultaneously_engaged_verified']=False
r['blocked_items'].append('L2 side-pull covers one guide, not an accepted two-guide S routing trajectory.')
r['numerical_stability_history']='Earlier 240 Hz and open-loop lateral force tests failed; retained raw reports. Accepted force comparison uses 960 Hz TGS 64/16; accepted guide diagnostics use feedback-limited endpoint actuation.'
s=json.loads((O/'reports/source_audit.json').read_text());actual=hashlib.sha256(Path(s['source']).read_bytes()).hexdigest();assert actual==s['sha256'];r['main_stage_sha256_before_after']=actual
p.write_text(json.dumps(r,indent=2))
# Extract a real guide-contact frame and mark its measured contact point.
cases=[d for d in r['latest_valid_record_sets'] if d['scene']=='L1_one_guide' and d['physics_acceptance']]
if cases:
 frames=json.loads((Path(cases[-1]['report_path']).parent/'frames.json').read_text())
 valid=[f for f in frames if f['contact_points']]
 if valid:
  f=max(valid,key=lambda x:sum(abs(c['normal_force_N']) for c in x['contact_points']))
  font=lambda n:ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
  im=Image.new('RGB',(1300,650),'#f5f7fa');d=ImageDraw.Draw(im)
  def uv(p):return (80+p[0]*2400,470-p[1]*2400)
  d.text((35,25),'MEASURED GUIDE CONTACT - XY projection / not a routing success claim',font=font(23),fill='#16334c')
  d.text((35,70),f"t = {f['time_s']:.3f} s | controlled pull, 10 mm/s target, 0.25 N force cap",font=font(19),fill='#16334c')
  x,y=uv([.2,.04,0]);rad=.018*2400;d.ellipse((x-rad,y-rad,x+rad,y+rad),fill='#bda17d',outline='#675240',width=2)
  d.line([uv(x) for x in f['centers_m']],fill='#1a8f65',width=10)
  for c in f['contact_points']:
   x,y=uv(c['point_m']);d.ellipse((x-7,y-7,x+7,y+7),fill='#d43e38')
   d.line([(x,y),(x+90,y-100)],fill='#d43e38',width=2)
   d.text((x+90,y-120),f"{c['normal_force_N']:.4f} N",font=font(18),fill='#b12d27')
  d.text((35,565),'Red marks: actual PhysX contact points. Force is normal component; material stress unavailable.',font=font(19),fill='#16334c')
  d.text((35,605),'Guide base/fasteners omitted in this XY data plot; printable solid geometry is in the CAD preview.',font=font(17),fill='#6b7785')
  im.save(O/'guide_contact.png')
# Variants listed honestly: built obstacle count, uncalibrated material/length variants disabled.
catalog={'version':1,'built':['L0_sensor_bench.usda','L1_one_guide.usda','L2_two_guides.usda'],
'wire_nominal':{'diameter_mm':7.,'length_mm':460.,'segments':24,'mass_g':24.},
'disabled_until_calibrated':{'length_variants':True,'diameter_variants':True,'bending_stiffness_variants':True,'friction_randomization':True,'four_wire_crossing':True},
'reason':'No calibrated elastic model or complete route feasibility/retention validation; these are not active RL variants.'}
(O/'variant_catalog.json').write_text(json.dumps(catalog,indent=2))
api={'knowledge_sources':['/mnt/data2/isaacsim-5.0.0/exts/isaacsim.core.prims/isaacsim/core/prims/impl/articulation.py','/mnt/data2/isaacsim-5.0.0/exts/isaacsim.core.prims/isaacsim/core/prims/impl/rigid_prim.py','/mnt/data2/isaacsim-5.0.0/extscache/omni.physics.tensors-107.3.18+107.3.1.lx64.r.cp311.u353/omni/physics/tensors/impl/api.py'],
'mcp_probe':{'tool':'get_scene_info','endpoint':'localhost:8766 on isaac-b','result':'connection refused; no GUI attached'},
'execution':'isolated headless Isaac 5.0 native PhysX; no PPO, no Franka execution'}
(O/'reports/api_provenance.json').write_text(json.dumps(api,indent=2))
# Reviewable new-source diff. Generated recordings are reviewed as artifacts with hashes.
codepaths=sorted([*O.glob('*.usda'),*O.glob('*.md'),O/'config.json',O/'variant_catalog.json',*(O/'scripts').glob('*.py')])
patch=[];stats=[]
for p in codepaths:
 lines=p.read_text().splitlines(True);patch.extend(difflib.unified_diff([],lines,fromfile='/dev/null',tofile=str(p)))
 stats.append({'path':str(p),'added_lines':len(lines),'removed_lines':0})
(O/'changes.patch').write_text(''.join(patch));(O/'diff_stat.json').write_text(json.dumps(stats,indent=2))
skip={'file_manifest.json','SHA256SUMS.txt','PACKAGE_SHA256.txt','Task2_wire_safety_v001.zip'}
files=[]
for root,dirs,names in os.walk(O):
 dirs[:]=[d for d in dirs if d not in ['runtime','__pycache__']]
 for n in names:
  if n not in skip:files.append(Path(root)/n)
manifest=[]
for p in sorted(files):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 manifest.append({'path':str(p),'relative_path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
(O/'file_manifest.json').write_text(json.dumps({'files':manifest,'excluded_cache_directories':[str(O/'runtime'),str(O/'scripts/__pycache__')],'main_stage_unchanged_sha256':actual},indent=2))
(O/'SHA256SUMS.txt').write_text(''.join(f"{x['sha256']}  {x['relative_path']}\n" for x in manifest))
pack=O/'Task2_wire_safety_v001.zip'
with zipfile.ZipFile(pack,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(files):
  if p.suffix not in ['.log']:z.write(p,str(p.relative_to(O)))
 for n in ['file_manifest.json','SHA256SUMS.txt']:z.write(O/n,n)
sha=hashlib.sha256(pack.read_bytes()).hexdigest();(O/'PACKAGE_SHA256.txt').write_text(sha+'  '+pack.name+'\n')
print('PACKAGE',pack,'BYTES',pack.stat().st_size,'SHA256',sha)
print('ARTIFACT_FILES',len(manifest),'SOURCE_DIFF_FILES',len(codepaths),'ADDED_LINES',sum(x['added_lines'] for x in stats))
print('GATES',json.dumps({k:v for k,v in r.items() if isinstance(v,bool)}))
