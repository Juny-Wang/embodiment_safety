"""Build a full Task2 scene COPY; no source edits, robot control or simulation."""
from pathlib import Path
import json,hashlib,shutil,math,ast,difflib
import numpy as np
import trimesh as tm
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Gf,Sdf
R=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io')
S=R/'scene';O=S/'wire_guides_full_v001'
source=S/'riser_io_preview.usda';target=S/'riser_io_wire_guides_v001.usda'
assert not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()==hashlib.sha256(source.read_bytes()).hexdigest(),'Refuse overwrite of an edited scene copy'
for d in ['stl','reports','runs','runtime','assets']: (O/d).mkdir(exist_ok=True)
before=hashlib.sha256(source.read_bytes()).hexdigest()
shutil.copy2(source,target)
stage=Usd.Stage.Open(str(target));original=Usd.Stage.Open(str(source))
assert UsdGeom.GetStageMetersPerUnit(stage)==1 and UsdGeom.GetStageUpAxis(stage)=='Z'
xc=UsdGeom.XformCache()
def mesh(path,g,color,mat=None):
 m=UsdGeom.Mesh.Define(stage,path);m.CreatePointsAttr([Gf.Vec3f(*v) for v in g.vertices])
 m.CreateFaceVertexCountsAttr([3]*len(g.faces));m.CreateFaceVertexIndicesAttr(g.faces.reshape(-1).tolist())
 m.CreateSubdivisionSchemeAttr('none');m.CreateDisplayColorAttr([Gf.Vec3f(*color)])
 m.CreateExtentAttr(UsdGeom.PointBased(m).ComputeExtent(m.GetPointsAttr().Get()))
 UsdPhysics.CollisionAPI.Apply(m.GetPrim()).CreateCollisionEnabledAttr(True)
 UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('none')
 m.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
 m.GetPrim().CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.0001)
 m.GetPrim().CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(0.)
 if mat:UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat,materialPurpose='physics')
 return m
mat=UsdShade.Material.Define(stage,'/Task2WireGuides/Materials/PrintedContact')
ma=UsdPhysics.MaterialAPI.Apply(mat.GetPrim());ma.CreateStaticFrictionAttr(.4);ma.CreateDynamicFrictionAttr(.3);ma.CreateRestitutionAttr(0)
mat.GetPrim().SetCustomDataByKey('calibration','DEMONSTRATION_ONLY')
parts={}
old=S/'wire_safety_v001/stl'
parts['guide_column_D36_H35']=tm.load_mesh(old/'guide_column_D36_H35.stl')
parts['nut_custom_D10_P2_clearance_030']=tm.load_mesh(old/'nut_custom_D10_P2_clearance_030.stl')
ns={'tm':tm,'np':np,'math':math};tree=ast.parse((R/'scripts/build_printable_v001.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['radius','thread_volume']],type_ignores=[]),'<thread_functions>','exec'),ns)
head=tm.creation.cylinder(8.5,5,sections=12);head.apply_translation([0,0,68.5])
parts['guide_mount_screw_D10_P2_shaft66']=tm.boolean.union([ns['thread_volume'](66,taper=True),head],engine='manifold')
for p in (S/'printable_v001/stl').glob('*.stl'):
 if p.name.startswith('fit_'):continue
 parts.setdefault(p.stem,tm.load_mesh(p))
# Export original mechanical geometry without turning a visualization into an invented replacement.
legacy={}
for root,name in [('/Electronics_140mm_19mm_cage','cage_source_geometry'),('/TE_L26620011_4_connector_v2','connector_source_geometry')]:
 pr=stage.GetPrimAtPath(root+'/Mesh');u=UsdGeom.Mesh(pr)
 pts=np.array([list(xc.GetLocalToWorldTransform(pr).Transform(Gf.Vec3d(v))) for v in u.GetPointsAttr().Get()])*1000
 counts=u.GetFaceVertexCountsAttr().Get();inds=u.GetFaceVertexIndicesAttr().Get();faces=[];offset=0
 for n in counts:
  ids=list(inds[offset:offset+n]);offset+=n
  faces.extend([[ids[0],ids[k],ids[k+1]] for k in range(1,n-1)])
 g=tm.Trimesh(pts,faces,process=True);g.fix_normals()
 origin=g.bounds[0].copy();g.apply_translation(-origin)
 parts[name]=g;legacy[name]={'source_prim':root,'print_origin_world_mm':origin.tolist(),'unchanged_shape':True}
# TPU specimens: same nominal wire diameter / approximate source centerline length,
# straight specimens print flat. They are not calibrated matches to the rigid-link simulation.
wire_specs=[]
for w in range(1,5):
 prims=[stage.GetPrimAtPath(f'/wire{w}/Segment_{i:02d}') for i in range(24)]
 centers=np.array([xc.GetLocalToWorldTransform(p).ExtractTranslation() for p in prims])
 lengths=[float(p.GetAttribute('height').Get())*.001 for p in prims]
 total=float(np.linalg.norm(np.diff(centers,axis=0),axis=1).sum()+(lengths[0]+lengths[-1])/2)*1000
 # Tile print as four mechanically joined segments is not equivalent to a soft cable;
 # therefore export one continuous strip only, with explicit bed-size requirement.
 g=tm.creation.cylinder(3.5,total,sections=48)
 g.apply_transform(tm.transformations.rotation_matrix(math.pi/2,[0,1,0]));g.apply_translation([total/2,0,3.5])
 name=f'wire{w}_TPU_diameter7_length{total:.1f}_candidate';parts[name]=g
 wire_specs.append({'wire':w,'stl':name+'.stl','diameter_mm':7,'length_mm':total,'material':'TPU candidate, calibration required','print_bed_length_mm_min':total+10,'electrical':False,'matches_simulated_stiffness':False})
# Mount guides on the actual 50 mm tabletop. Four real clearance holes,
# four printed long screws and under-table nuts; no hidden floating mount.
positions=[(-.18,-.03,.75),(.13,.065,.75)]
holes=[];guides=[]
for i,pos in enumerate(positions,1):
 g=parts['guide_column_D36_H35'].copy();g.apply_scale(.001);g.apply_translation(pos)
 path=f'/Task2WireGuides/Post_{i}';mesh(path,g,(.82,.52,.16),mat);guides.append(path)
 for j,dx in enumerate([-.027,.027],1):
  x,y,z=pos[0]+dx,pos[1],.689
  bolt=parts['guide_mount_screw_D10_P2_shaft66'].copy();bolt.apply_scale(.001);bolt.apply_translation([x,y,z])
  mesh(f'/Task2WireGuides/Mount_{i}/Screw_{j}',bolt,(.3,.42,.56),mat)
  nut=parts['nut_custom_D10_P2_clearance_030'].copy();nut.apply_scale(.001);nut.apply_translation([x,y,.690])
  mesh(f'/Task2WireGuides/Mount_{i}/Nut_{j}',nut,(.3,.42,.56),mat)
  h=tm.creation.cylinder(.0057,.07,sections=64);h.apply_translation([x,y,.725]);holes.append(h)
top=stage.GetPrimAtPath('/World/Table/Top')
bound=UsdGeom.BBoxCache(0,['default']).ComputeWorldBound(top).ComputeAlignedRange()
table=tm.creation.box(np.array(bound.GetMax())-np.array(bound.GetMin()))
table.apply_translation((np.array(bound.GetMax())+np.array(bound.GetMin()))/2)
table=tm.boolean.difference([table]+holes,engine='manifold')
parent_inv=xc.GetLocalToWorldTransform(top.GetParent()).GetInverse()
table.vertices=np.array([parent_inv.Transform(Gf.Vec3d(*v)) for v in table.vertices])
color=UsdGeom.Gprim(top).GetDisplayColorAttr().Get() or [Gf.Vec3f(.5,.5,.5)]
UsdGeom.Xformable(top).ClearXformOpOrder()
top.SetTypeName('Mesh')
mesh('/World/Table/Top',table,tuple(color[0]))
# Only invalid rotation normalization, no change to wire shape or anchor positions.
normalized=[]
for w in range(1,5):
 for p in Usd.PrimRange(stage.GetPrimAtPath(f'/wire{w}')):
  if p.IsA(UsdPhysics.Joint):
   j=UsdPhysics.Joint(p)
   for a in [j.GetLocalRot0Attr(),j.GetLocalRot1Attr()]:
    q=a.Get()
    if q and abs(q.GetLength()-1)>1e-5:
     normalized.append({'prim':str(p.GetPath()),'attribute':a.GetName(),'previous_norm':q.GetLength()})
     a.Set(q.GetNormalized())
  if p.IsA(UsdGeom.Capsule):
   UsdShade.MaterialBindingAPI.Apply(p).Bind(mat,materialPurpose='physics')
# Do not change robot, drive targets, cable anchor positions or their articulation model.
stage.GetRootLayer().customLayerData={**stage.GetRootLayer().customLayerData,'wireGuideCopy':True,'sourceSHA256':before,'materialStressCalibrated':False}
stage.GetRootLayer().Save()
print_report={}
for name,g in parts.items():
 g.export(O/'stl'/(name+'.stl'))
 print_report[name]={'dimensions_mm':g.extents.tolist(),'watertight':bool(g.is_watertight),'positive_volume':bool(g.is_volume),'connected_bodies':int(g.body_count)}
(O/'reports/print_geometry.json').write_text(json.dumps(print_report,indent=2))
# Conservative initial clearance: guide complete AABB vs all original task meshes.
bc=UsdGeom.BBoxCache(0,['default','render','proxy'])
clearances=[]
for path in guides:
 r=bc.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange();lo=np.array(r.GetMin());hi=np.array(r.GetMax());ds=[]
 for p in original.Traverse():
  if not p.IsA(UsdGeom.Gprim) or str(p.GetPath()).startswith('/World'):continue
  b=bc.ComputeWorldBound(p).ComputeAlignedRange()
  d=np.linalg.norm(np.maximum(0,np.maximum(np.array(b.GetMin())-hi,lo-np.array(b.GetMax()))))
  ds.append({'prim':str(p.GetPath()),'aabb_clearance_mm':float(d*1000)})
 clearances.append({'guide':path,'nearest':sorted(ds,key=lambda v:v['aabb_clearance_mm'])[:5]})
def signature(s,root):
 out={}
 for p in Usd.PrimRange(s.GetPrimAtPath(root)):
  out[str(p.GetPath())]={'type':p.GetTypeName(),'schemas':p.GetAppliedSchemas(),'attrs':{a.GetName():str(a.Get()) for a in p.GetAttributes()},'rels':{r.GetName():list(map(str,r.GetTargets())) for r in p.GetRelationships()}}
 return out
robot_same=signature(stage,'/World/Robot')==signature(original,'/World/Robot')
assert robot_same
assert all(c['nearest'][0]['aabb_clearance_mm']>1 for c in clearances)
after=hashlib.sha256(source.read_bytes()).hexdigest();assert after==before
cfg={'version':1,'scene':str(target),'source_scene':str(source),'source_sha256':before,'units':{'length':'m','force':'N','moment':'N*m','STL':'mm'},'guide_paths':guides,'wire_roots':[f'/wire{i}' for i in range(1,5)],'thresholds':{'source':'DEMONSTRATION_ONLY_NOT_MATERIAL_LIMIT','joint_reaction_N':1.,'guide_normal_N':1.,'min_bend_radius_m':.035,'warning_ratio':.7,'mode':'warning_only'},'force_location':'Joint_00 child Segment_01 incoming wrench; NOT total connector root load','material_stress_available':False,'strain_relief_verified':False}
(O/'config.json').write_text(json.dumps(cfg,indent=2))
report={'source_unchanged':before==after,'source_sha256':before,'scene_copy':str(target),'robot_composed_properties_unchanged':robot_same,'source_prim_paths_preserved':all(stage.GetPrimAtPath(p.GetPath()) for p in original.Traverse()),'guide_initial_aabb_clearance':clearances,'normalized_copy_only_quaternions':normalized,'table_mount':{'table_thickness_mm':50,'clearance_hole_mm':11.4,'holes':[[p[0]+dx,p[1]] for p in positions for dx in [-.027,.027]],'screw_shaft_mm':66,'nut_bottom_world_z_m':.690,'mount_structural_load_test_passed':False},'legacy_print_exports':legacy,'wire_print_candidates':wire_specs,'all_task_parts_print_verified':False,'material_stress_ready':False,'full_task_ready':False,'source_joint_anchor_defects':'Retained; see source_audit.json. Full scene dynamics must be assessed separately.'}
(O/'reports/static_audit.json').write_text(json.dumps(report,indent=2))
shutil.copy2(S/'wire_safety_v001/reports/source_audit.json',O/'reports/source_audit.json')
patch=''.join(difflib.unified_diff(source.read_text().splitlines(True),target.read_text().splitlines(True),fromfile=str(source),tofile=str(target),n=2))
(O/'reports/scene_changes.patch').write_text(patch)
print(json.dumps({'created':str(target),'static':report,'print_geometry':print_report},indent=2))
