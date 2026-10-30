from pathlib import Path
import json,math,shutil
import trimesh as tm,numpy as np
P=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/wire_guides_full_v001')
g=tm.load_mesh(P/'stl/connector_source_geometry.stl')
edges=g.edges_unique[np.bincount(g.edges_unique_inverse)==1]
adj={}
for a,b in edges:
 adj.setdefault(int(a),set()).add(int(b));adj.setdefault(int(b),set()).add(int(a))
seen=set();components=[]
for a in adj:
 if a in seen:continue
 stack=[a];component=set()
 while stack:
  v=stack.pop()
  if v in component:continue
  component.add(v);stack+=list(adj[v]-component)
 seen|=component;components.append(component)
repairs=[]
for comp in components:
 assert len(comp)==3
 ids=list(comp);pts=g.vertices[ids]
 axis=int(np.argmax(np.ptp(pts,axis=0)));ids.sort(key=lambda i:g.vertices[i,axis]);a,b,c=ids
 error=float(np.linalg.norm(np.cross(g.vertices[b]-g.vertices[a],g.vertices[c]-g.vertices[a]))/np.linalg.norm(g.vertices[c]-g.vertices[a]))
 assert error<1e-5,'Not a collinear T junction'
 fs=g.faces.tolist();matches=[i for i,f in enumerate(fs) if a in f and c in f];assert len(matches)==1
 index=matches[0];face=fs[index]
 for k in range(3):
  x,y,z=face[k],face[(k+1)%3],face[(k+2)%3]
  if {x,y}=={a,c}:break
 fs[index]=[x,b,z];fs.append([b,y,z])
 g=tm.Trimesh(g.vertices,fs,process=False)
 repairs.append({'long_edge':[a,c],'inserted_existing_vertex':b,'collinearity_error_mm':error,'surface_geometry_changed':False})
assert g.is_watertight and g.is_volume
g.export(P/'stl/connector_print_repaired.stl')
(P/'reference_only').mkdir(exist_ok=True)
shutil.move(P/'stl/connector_source_geometry.stl',P/'reference_only/connector_source_geometry_open_mesh.stl')
# Compact one-piece TPU print candidates. U layout fits a nominal 220 mm bed;
# no rigid couplers and no claims that TPU matches original physics.
report=json.loads((P/'reports/static_audit.json').read_text())
wire_candidates=[]
for spec in report['wire_print_candidates']:
 L=spec['length_mm'];R=40.;leg=(L-math.pi*R)/2
 xy=[]
 for y in np.linspace(0,leg,35):xy.append([-R,y])
 for a in np.linspace(math.pi,0,49)[1:]:xy.append([R*math.cos(a),leg+R*math.sin(a)])
 for y in np.linspace(leg,0,35)[1:]:xy.append([R,y])
 p=np.c_[np.array(xy),np.full(len(xy),3.5)]
 tang=np.gradient(p,axis=0);tang/=np.linalg.norm(tang,axis=1)[:,None]
 normal=np.c_[-tang[:,1],tang[:,0],np.zeros(len(p))]
 verts=[]
 for center,n in zip(p,normal):
  for theta in np.linspace(0,2*math.pi,48,endpoint=False):verts.append(center+3.5*(math.cos(theta)*n+math.sin(theta)*np.array([0,0,1])))
 faces=[];N=len(p);K=48
 for i in range(N-1):
  for k in range(K):
   a=i*K+k;b=i*K+(k+1)%K;c=(i+1)*K+(k+1)%K;d=(i+1)*K+k
   faces.extend([[a,b,c],[a,c,d]])
 for offset,reverse in [(0,True),((N-1)*K,False)]:
  for k in range(1,K-1):faces.append([offset,offset+k+1,offset+k] if reverse else [offset,offset+k,offset+k+1])
 cable=tm.Trimesh(verts,faces,process=True);cable.fix_normals()
 assert cable.is_watertight and cable.is_volume
 name=f'wire{spec["wire"]}_TPU_U_layout_candidate.stl';cable.export(P/'stl'/name)
 shutil.move(P/'stl'/spec['stl'],P/'reference_only'/spec['stl'])
 wire_candidates.append({'wire':spec['wire'],'stl':name,'dimensions_mm':cable.extents.tolist(),'centerline_nominal_mm':L,'polyline_chord_length_mm':float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()),'watertight':True,'material':'TPU, candidate only; bend/tension calibration required','matches_material_law':False})
(P/'reports/print_repairs.json').write_text(json.dumps({'connector_T_junction_repairs':repairs,'wire_compact_candidates':wire_candidates},indent=2))
geo={}
for f in (P/'stl').glob('*.stl'):
 mesh=tm.load_mesh(f);geo[f.name]={'watertight':bool(mesh.is_watertight),'volume_positive':bool(mesh.is_volume),'dimensions_mm':mesh.extents.tolist()}
assert all(v['watertight'] and v['volume_positive'] for v in geo.values())
(P/'reports/print_geometry_final.json').write_text(json.dumps(geo,indent=2))
print('PRINT_MESHES',len(geo),'all watertight; connector repair does not change surface')
