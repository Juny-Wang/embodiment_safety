from pathlib import Path
import json,hashlib,shutil,math
import numpy as np,trimesh as tm
from pxr import Usd,UsdGeom,UsdShade
from PIL import Image,ImageDraw,ImageFont
R=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io');S=R/'scene';O=S/'printable_v001'
report=json.loads((O/'geometry_audit.json').read_text());C=json.loads((O/'geometry_config.json').read_text())
st=Usd.Stage.Open(str(S/'riser_io_printable_v001.usda'))
xc=UsdGeom.XformCache()
checks={}
for i in range(1,5):
 seat=st.GetPrimAtPath(f'/PCB_4_Terminal/Terminal_{i}_Seat')
 nut=st.GetPrimAtPath(f'/NutPlate_Coarse/Nut_{i}')
 a=np.array(xc.GetLocalToWorldTransform(seat).ExtractTranslation())
 b=np.array(xc.GetLocalToWorldTransform(nut).ExtractTranslation())
 checks[f'coaxial_{i}_mm']=float(np.linalg.norm(a[:2]-b[:2])*1000)
checks['all_new_mesh_material_bindings_valid']=all(bool(UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial('physics')[0]) for p in st.Traverse() if p.IsA(UsdGeom.Mesh) and any(str(p.GetPath()).startswith(q) for q in ['/PCB_holder','/PCB_4_Terminal','/NutPlate_Coarse','/ScrewTray','/Screw_']) )
checks['ring_materials_valid']=all(bool(UsdShade.MaterialBindingAPI(st.GetPrimAtPath(f'/wire{i}/_0347300_ring_terminal/Mesh')).ComputeBoundMaterial('physics')[0]) for i in range(1,5))
assert checks['all_new_mesh_material_bindings_valid'] and checks['ring_materials_valid']
# Offline path sweep: coupled helix motion and fixed-angle pull interference.
m=lambda name:tm.load_mesh(O/'stl'/f'{name}.stl',process=True)
screw=m('screw_custom_D10_P2');nut=m('nut_custom_D10_P2_clearance_030')
nut.apply_translation([0,0,4])
vols=[]
for turn in np.linspace(0,1,9):
 a=screw.copy();a.apply_transform(tm.transformations.rotation_matrix(turn*2*math.pi,[0,0,1]));a.apply_translation([0,0,3+2*turn])
 v=tm.boolean.intersection([a,nut],engine='manifold')
 vols.append(abs(float(v.volume)) if len(v.faces) else 0.)
checks['helix_path_intersection_mm3']=vols;checks['helix_path_clear']=max(vols)<.01
a=screw.copy();a.apply_translation([0,0,3.5])
v=tm.boolean.intersection([a,nut],engine='manifold');checks['fixed_angle_0p5mm_pull_interference_mm3']=abs(float(v.volume))
# This is geometry evidence, not a friction/retention force result.
report['checks'].update(checks);assert checks['helix_path_clear']
(O/'geometry_audit.json').write_text(json.dumps(report,indent=2))

fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=lambda n:ImageFont.truetype(fontpath,n)
X=[-42,-14,14,42]
def render(path,exploded):
 items=[]
 def add(name,p,col):
  g=m(name);g.apply_translation(p);items.append((g,np.array(col)))
 add('fixture_base',[0,0,0],(72,91,115))
 add('terminal_board_with_open_ring_seats',[0,0,32 if exploded else 14],(55,147,105))
 for x in X:
  add('nut_custom_D10_P2_clearance_030',[x,0,17 if exploded else 4],(158,130,184))
  add('ring_terminal_surrogate',[x,0,54 if exploded else 18],(216,139,61))
  add('screw_custom_D10_P2',[x,0,69 if exploded else 3],(191,199,211))
 if not exploded:add('screw_tray_four_real_pockets',[120,0,0],(77,126,162))
 angle=math.radians(25);cx=math.cos(angle);sx=math.sin(angle)
 rz=np.array([[cx,-sx,0],[sx,cx,0],[0,0,1.]])
 tilt=math.radians(-57);rx=np.array([[1,0,0],[0,math.cos(tilt),-math.sin(tilt)],[0,math.sin(tilt),math.cos(tilt)]])
 transform=rx@rz
 pts=np.vstack([g.vertices@transform.T for g,c in items]);lo=pts[:,:2].min(0);hi=pts[:,:2].max(0)
 scale=min(1410/(hi[0]-lo[0]),650/(hi[1]-lo[1]))
 center=(lo+hi)/2
 im=Image.new('RGB',(1600,1000),(242,245,249));d=ImageDraw.Draw(im)
 polys=[]
 for g,col in items:
  verts=g.vertices@transform.T;norm=g.face_normals@transform.T
  faceids=np.where(norm[:,2]>0)[0]
  for k in faceids:
   v=verts[g.faces[k]]
   uv=(v[:,:2]-center)*[scale,-scale]+[800,520]
   light=.55+.45*abs(norm[k,2]);shade=tuple((col*light).astype(int))
   polys.append((float(v[:,2].mean()),[tuple(x) for x in uv],shade))
 for z,poly,col in sorted(polys,key=lambda x:x[0]):d.polygon(poly,fill=col)
 d.text((55,30),'TASK2  |  '+('EXPLODED PRINTABLE ASSEMBLY' if exploded else 'ASSEMBLED PRINTABLE WORKPIECES'),font=font(31),fill=(25,45,66))
 d.text((55,80),'Actual STL geometry | Custom D10 / pitch 2 mm | Not ISO M10 or M4',font=font(23),fill=(66,83,104))
 d.text((55,890),'Orange: ring  |  Green: seat board  |  Purple: printed nuts  |  Grey: screws / base',font=font(22),fill=(25,45,66))
 d.text((55,932),'STATIC DESIGN ONLY - printing, live threading, cable retention and robot task require validation.',font=font(20),fill=(150,55,25))
 im.save(path)
render(O/'assembly.png',False);render(O/'exploded.png',True)
print(json.dumps(checks,indent=2))
