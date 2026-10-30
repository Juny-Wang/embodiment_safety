from pathlib import Path
import json, math, hashlib, shutil, re, sys
import numpy as np
import trimesh as tm
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Sdf,Gf,Vt

ROOT=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io')
SC=ROOT/'scene';OUT=SC/'printable_v001';STL=OUT/'stl';STL.mkdir(exist_ok=True)
AS=SC/'assets/task2_wire/usd/printable_v001';AS.mkdir(exist_ok=True)
REPORT={'version':'printable_v001','units_stl':'mm','units_usd':'m','thread':'CUSTOM_D10_P2_NOT_ISO_M10','physics_ready':False,'print_fit_verified':False,'parts':{},'checks':{}}
X=[-42.,-14.,14.,42.]
ORIGIN=np.array([-.05,-.071,.75])
meshes={}

def box(size,center):
 m=tm.creation.box(extents=size);m.apply_translation(center);return m
def cyl(r,h,z=0,xy=(0,0),n=96):
 m=tm.creation.cylinder(radius=r,height=h,sections=n);m.apply_translation([*xy,z+h/2]);return m
def union(*m):return tm.boolean.union(list(m),engine='manifold')
def diff(a,*b):return tm.boolean.difference([a,*b],engine='manifold')
def shifted(m,p):
 c=m.copy();c.apply_translation(p);return c
def radius(z,theta,clearance=0,offset=0,taper=False):
 phase=((z+offset)/2.-theta/(2*math.pi))%1.
 # Trapezoidal crest/root: 2 mm lead, 1.2 mm radial depth.
 t=np.minimum(phase,1-phase)*2
 ridge=np.clip((t-.18)/.64,0,1)
 r=3.8+1.2*ridge+clearance
 if taper:r=np.minimum(r,3.4+np.maximum(z,0)*1.0)
 return r
def thread_volume(length,clearance=0,offset=0,taper=False):
 n=128;nz=int(math.ceil(length/.10))+1
 th=np.arange(n)*2*math.pi/n
 vertices=[]
 for z in np.linspace(0,length,nz):
  rr=radius(z,th,clearance,offset,taper)
  vertices.extend(np.c_[rr*np.cos(th),rr*np.sin(th),np.full(n,z)].tolist())
 vertices.extend([[0,0,0],[0,0,length]])
 faces=[]
 for j in range(nz-1):
  for k in range(n):
   a=j*n+k;b=j*n+(k+1)%n;c=b+n;d=a+n
   faces.extend([[a,b,c],[a,c,d]])
 bottom=len(vertices)-2;top=bottom+1
 for k in range(n):
  faces.extend([[bottom,(k+1)%n,k],[top,(nz-1)*n+k,(nz-1)*n+(k+1)%n]])
 m=tm.Trimesh(vertices=np.array(vertices),faces=np.array(faces),process=True)
 m.fix_normals();assert m.is_volume
 return m
def save(name,m):
 assert m.is_watertight and m.is_volume,(name,m.is_watertight,m.volume)
 meshes[name]=m
 file=STL/(name+'.stl');m.export(file)
 reload=tm.load_mesh(file,process=True)
 assert reload.is_watertight and reload.is_volume,name
 REPORT['parts'][name]={'stl':str(file),'bbox_mm':m.bounds.tolist(),'dimensions_mm':m.extents.tolist(),'volume_mm3':float(m.volume),'watertight':True,'faces':len(m.faces),'connected_bodies':int(m.body_count)}
 return m

# Female cavity is the exact same helix with radial clearance, phase aligned at final insertion.
male=thread_volume(18,taper=True)
head=cyl(8.5,5,18,n=12)
screw=save('screw_custom_D10_P2',union(male,head))
nut_cutter=shifted(thread_volume(12,clearance=.30,offset=0),[0,0,-1])
# final male tip at z=3; nut bottom at z=4: cutter local z corresponds male z+1.
nut=diff(cyl(11.5,10,0,n=6),nut_cutter,cyl(5.9,1,9.2))
save('nut_custom_D10_P2_clearance_030',nut)
for gap in [.2,.4]:
 cut=shifted(thread_volume(12,clearance=gap,offset=0),[0,0,-1])
 save('fit_nut_clearance_'+str(int(gap*100)).zfill(3),diff(cyl(11.5,10,n=6),cut,cyl(5.9,1,9.2)))

# Ring + grasp tab + cable insertion barrel; functional annular hole remains open.
annulus=diff(cyl(11,3),cyl(5.7,5,-1))
tab=box([10,16,3],[0,15,1.5])
boss=box([11,10,9],[0,23,4.5])
bore=tm.creation.cylinder(radius=3.7,height=14,sections=64)
bore.apply_transform(tm.transformations.rotation_matrix(math.pi/2,[1,0,0]))
bore.apply_translation([0,23,3.5])
# 7 mm cable snaps into the 7.4 mm open-bottom snap barrel; fit is a prototype, no rated retention.
ring=save('ring_terminal_surrogate',diff(union(annulus,tab,boss),bore))

# Base is supported on tabletop. Nut seats and shank pockets are actual voids.
base=box([154,70,6],[0,0,3])
cuts=[]
for x in X:
 cuts += [cyl(11.85,4,4,(x,0),n=6),cyl(6.1,4,2,(x,0))]
base=diff(base,*cuts)
for x in [-65,65]:
 for y in [-20,20]:
  base=union(base,cyl(4,8,6,(x,y)),cyl(1.9,4.5,14,(x,y)))
# Four bench mounting holes. Physical fixture attachment is an explicit boundary condition.
base=diff(base,*[cyl(3.3,8,-1,(x,y)) for x in [-68,68] for y in [-29,29]])
save('fixture_base',base)

board=diff(box([140,50,4],[0,0,2]),*[cyl(5.7,6,-1,(x,0)) for x in X],
 *[cyl(2.15,6,-1,(x,y)) for x in [-65,65] for y in [-20,20]])
for x in X:
 fence=diff(cyl(13.4,1.6,3.9,(x,0)),cyl(11.4,3,3.4,(x,0)),box([12,24,5],[x,16,5]))
 board=union(board,fence)
save('terminal_board_with_open_ring_seats',board)

# Four open-top screw nests: true pocket, head remains accessible above the tray.
TX=[-20,20];TY=[-20,20]
tray=box([100,100,20],[0,0,10])
tray=diff(tray,*[cyl(6.3,20,2,(x,y)) for x in TX for y in TY])
save('screw_tray_four_real_pockets',tray)

# Separate thread coupon fastener is identical geometry; BOM references the screw.
# Unit-correct watertight STL for all task workpieces; no material properties are claimed.
MAT_PATH='/Asset/Materials/PrintedPolymer'
def mesh_prim(stage,path,m,collision=True,dynamic=False,color=(.5,.5,.5)):
 g=UsdGeom.Mesh.Define(stage,path)
 g.CreatePointsAttr(Vt.Vec3fArray.FromNumpy((m.vertices*.001).astype(np.float32)))
 g.CreateFaceVertexCountsAttr([3]*len(m.faces))
 g.CreateFaceVertexIndicesAttr(m.faces.reshape(-1).tolist())
 g.CreateSubdivisionSchemeAttr('none');g.CreateDoubleSidedAttr(False)
 g.CreateDisplayColorAttr([Gf.Vec3f(*color)])
 g.CreateExtentAttr([Gf.Vec3f(*(m.bounds[0]*.001)),Gf.Vec3f(*(m.bounds[1]*.001))])
 if collision:
  UsdPhysics.CollisionAPI.Apply(g.GetPrim()).CreateCollisionEnabledAttr(True)
  UsdPhysics.MeshCollisionAPI.Apply(g.GetPrim()).CreateApproximationAttr('sdf' if dynamic else 'none')
  if dynamic:
   g.GetPrim().AddAppliedSchema('PhysxSDFMeshCollisionAPI')
   g.GetPrim().CreateAttribute('physxSDFMeshCollision:sdfResolution',Sdf.ValueTypeNames.Int).Set(256)
  g.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
  g.GetPrim().CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.0001)
  g.GetPrim().CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(0.)
  UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath(MAT_PATH)),materialPurpose='physics')
 return g.GetPrim()
def asset(name,color,dynamic=False):
 f=AS/(name+'.usda')
 if f.exists():f.unlink()
 st=Usd.Stage.CreateNew(str(f));UsdGeom.SetStageMetersPerUnit(st,1);UsdGeom.SetStageUpAxis(st,'Z')
 root=UsdGeom.Xform.Define(st,'/Asset');st.SetDefaultPrim(root.GetPrim())
 mat=UsdShade.Material.Define(st,MAT_PATH);mp=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
 mp.CreateStaticFrictionAttr(.4);mp.CreateDynamicFrictionAttr(.3);mp.CreateRestitutionAttr(0.)
 mesh_prim(st,'/Asset/Mesh',meshes[name],True,dynamic,color)
 if dynamic:
  rb=UsdPhysics.RigidBodyAPI.Apply(root.GetPrim());rb.CreateRigidBodyEnabledAttr(True);rb.CreateKinematicEnabledAttr(False)
  mass=UsdPhysics.MassAPI.Apply(root.GetPrim());mass.CreateMassAttr(float(meshes[name].volume*1.27e-6))
  root.GetPrim().AddAppliedSchema('PhysxRigidBodyAPI')
  root.GetPrim().CreateAttribute('physxRigidBody:enableCCD',Sdf.ValueTypeNames.Bool).Set(True)
  root.GetPrim().AddAppliedSchema('PhysxContactReportAPI')
  root.GetPrim().CreateAttribute('physxContactReport:threshold',Sdf.ValueTypeNames.Float).Set(0.)
 root.GetPrim().SetCustomDataByKey('design_status','PRINT_PROTOTYPE_STATIC_AUDIT_ONLY')
 st.GetRootLayer().customLayerData={'units':'m','thread':'CUSTOM_D10_P2_NOT_ISO','friction':'demo_not_calibrated'}
 st.GetRootLayer().Save()
 return './assets/task2_wire/usd/printable_v001/'+name+'.usda'
refs={}
for name,col,dyn in [('fixture_base',(.25,.32,.42),False),('terminal_board_with_open_ring_seats',(.18,.58,.42),False),('nut_custom_D10_P2_clearance_030',(.55,.48,.68),False),('screw_tray_four_real_pockets',(.3,.48,.65),False),('screw_custom_D10_P2',(.8,.65,.26),True),('ring_terminal_surrogate',(.8,.42,.2),True)]:
 refs[name]=asset(name,col,dyn)

# Work on repaired sibling candidate, never compose malformed original phase2 assets.
candidate=SC/'riser_io_printable_v001.usda'
stage=Usd.Stage.Open(str(candidate));layer=stage.GetRootLayer()
layer.subLayerPaths=[p for p in layer.subLayerPaths if not p.startswith('metrics:')]
def place(path,name,pos):
 stage.RemovePrim(path)
 x=UsdGeom.Xform.Define(stage,path);x.GetPrim().GetReferences().AddReference(refs[name])
 x.AddTranslateOp().Set(Gf.Vec3d(*map(float,pos)))
 return x.GetPrim()
for path in ['/PCB_4_Terminal','/Task1PCB','/PCB_holder','/TBY21_M3_1','/TBY21_M3_02']:
 stage.RemovePrim(path)
place('/PCB_holder','fixture_base',ORIGIN)
place('/PCB_4_Terminal','terminal_board_with_open_ring_seats',ORIGIN+[0,0,.014])
# Keep a non-geometric legacy marker; board role now has a single authoritative prim.
marker=UsdGeom.Xform.Define(stage,'/Task1PCB').GetPrim();marker.SetCustomDataByKey('replaced_by','/PCB_4_Terminal')
UsdGeom.Xform.Define(stage,'/NutPlate_Coarse')
for i,x in enumerate(X,1):
 place('/NutPlate_Coarse/Nut_'+str(i),'nut_custom_D10_P2_clearance_030',ORIGIN+[x/1000,0,.004])
 frame=UsdGeom.Xform.Define(stage,f'/PCB_4_Terminal/Terminal_{i}_Seat')
 frame.AddTranslateOp().Set(Gf.Vec3d(x/1000,0,.004))
 frame.GetPrim().SetCustomDataByKey('wire_index',i)
 frame.GetPrim().SetCustomDataByKey('role','ring_bottom_seat_and_coaxial_screw_axis')
tray_origin=np.array([.22,-.21,.75])
place('/ScrewTray','screw_tray_four_real_pockets',tray_origin)
for i,(x,y) in enumerate([(x,y) for x in TX for y in TY],1):
 # Head underside rests on top of tray; shaft tip remains 0.05 mm above pocket floor.
 place('/Screw_'+str(i),'screw_custom_D10_P2',tray_origin+[x/1000,y/1000,.00205])
 xf=UsdGeom.Xform.Define(stage,f'/ScrewTray/Nest_{i}')
 xf.AddTranslateOp().Set(Gf.Vec3d(x/1000,y/1000,.020))

# Use existing local robot cache; preserve original robot placement, no runtime transforms.
robot_path='/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/franka_latch_open_test_v1/robot_cache/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd'
assert Path(robot_path).is_file()
stage.RemovePrim('/World/Robot')
robot=UsdGeom.Xform.Define(stage,'/World/Robot')
robot.GetPrim().GetReferences().AddReference(robot_path);robot.AddTranslateOp().Set(Gf.Vec3d(-.43,0,.75))
home=[0,-45,0,-135,0,90,45]
for p in Usd.PrimRange(robot.GetPrim()):
 if p.GetName()=='visuals':p.SetInstanceable(False)
 if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName().startswith('panda_joint'):
  k=int(p.GetName().replace('panda_joint',''))-1
  if 0<=k<7:
   UsdPhysics.DriveAPI.Apply(p,'angular').CreateTargetPositionAttr(home[k])
   p.AddAppliedSchema('PhysicsJointStateAPI:angular')
   p.CreateAttribute('state:angular:physics:position',Sdf.ValueTypeNames.Float).Set(home[k])
   p.CreateAttribute('state:angular:physics:velocity',Sdf.ValueTypeNames.Float).Set(0.)

# Replace visual and collision on each actual wire-end body; retain cable topology.
ring_audits=[]
for i in range(1,5):
 parent=stage.GetPrimAtPath(f'/wire{i}')
 end=stage.GetPrimAtPath(f'/wire{i}/Segment_23')
 xc=UsdGeom.XformCache()
 parentM=xc.GetLocalToWorldTransform(parent);endM=xc.GetLocalToWorldTransform(end)
 endpoint=np.array(endM.ExtractTranslation())
 # Barrel is centered +Y at 23 mm, 4 mm above ring underside. Rotate in XY outward.
 end_axis=np.array(endM.TransformDir(Gf.Vec3d(1,0,0)))
 end_axis=end_axis/np.linalg.norm(end_axis)
 feed_xy=-end_axis[:2]/np.linalg.norm(end_axis[:2])
 theta=math.atan2(feed_xy[1],feed_xy[0])-math.pi/2
 R=Gf.Rotation(Gf.Vec3d(0,0,1),math.degrees(theta))
 tail=np.array(R.TransformDir(Gf.Vec3d(0,.023,.0035)))
 center=endpoint-tail
 # Do not alter wire endpoint. Any existing table overlap is reported separately.
 path=f'/wire{i}/_0347300_ring_terminal'
 stage.RemovePrim(path)
 root=UsdGeom.Xform.Define(stage,path);root.GetPrim().GetReferences().AddReference(refs['ring_terminal_surrogate'])
 world=Gf.Matrix4d(1);world.SetRotate(R);world.SetTranslateOnly(Gf.Vec3d(*center))
 local=world*parentM.GetInverse()
 root.AddTransformOp().Set(local)
 root.GetPrim().SetCustomDataByKey('printable_part','ring_terminal_surrogate.stl')
 frame=UsdGeom.Xform.Define(stage,path+'/HoleFrame');frame.AddTranslateOp().Set(Gf.Vec3d(0,0,.0015))
 grasp=UsdGeom.Xform.Define(stage,path+'/GraspFrame');grasp.AddTranslateOp().Set(Gf.Vec3d(0,-.008,.0015))
 j=UsdPhysics.FixedJoint(stage.GetPrimAtPath(f'/wire{i}/FixedJoint'))
 if not j:j=UsdPhysics.FixedJoint.Define(stage,f'/wire{i}/FixedJoint')
 j.CreateBody0Rel().SetTargets([Sdf.Path(path)]);j.CreateBody1Rel().SetTargets([end.GetPath()])
 j.CreateLocalPos0Attr().Set(Gf.Vec3f(0,.023,.0035));j.CreateLocalPos1Attr().Set(Gf.Vec3f(0,0,0))
 # Body frame conventions: use normalized world orientations, no scale in quaternion.
 qend=endM.ExtractRotationQuat();qring=world.ExtractRotationQuat()
 q0=qring.GetInverse()*qend
 j.CreateLocalRot0Attr().Set(Gf.Quatf(q0));j.CreateLocalRot1Attr().Set(Gf.Quatf(1))
 j.CreateCollisionEnabledAttr().Set(False)
 stage.GetPrimAtPath(path).CreateAttribute('physics:velocity',Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(0))
 stage.GetPrimAtPath(path).CreateAttribute('physics:angularVelocity',Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(0))
 ring_audits.append({'path':path,'endpoint_world_m':endpoint.tolist(),'ring_world_m':center.tolist(),'anchor_error_mm':float(np.linalg.norm(center+tail-endpoint)*1000),'bottom_world_m':float(center[2]),'cable_diameter_mm':7.,'snap_bore_mm':7.4,'source_cable_tilt_deg':float(math.degrees(math.asin(abs(end_axis[2]))))})

# Erase baked initial velocities only; do not bake new rigid body poses.
for p in stage.Traverse():
 for key in ['physics:velocity','physics:angularVelocity']:
  a=p.GetAttribute(key)
  if a and a.HasAuthoredValueOpinion():a.Set(Gf.Vec3f(0))
scene=UsdPhysics.Scene(stage.GetPrimAtPath('/World/physicsScene'))
scene.CreateGravityMagnitudeAttr(9.81);scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1))
meta=dict(layer.customLayerData);meta.update({'printable_version':'v001','status':'STATIC_AUDIT_ONLY_PHYSICS_NOT_ACCEPTED','thread_spec':'CUSTOM_D10_P2_NOT_M4','bench_fixing':'fixture mounting holes; external bench attachment required'})
layer.customLayerData=meta
layer.Save()

# Geometry tests: closed screw, ring, board and nut assembly. All in mm.
closed_screw=shifted(screw,[0,0,3])
closed_ring=shifted(ring,[0,0,18])
closed_nut=shifted(nut,[0,0,4])
board_center=shifted(board,[0,0,14])
pairs={}
for name,a,b in [('screw_ring',closed_screw,closed_ring),('screw_nut',closed_screw,closed_nut),
 ('screw_base',shifted(closed_screw,[X[0],0,0]),base),('ring_board',shifted(closed_ring,[X[0],0,0]),board_center),
 ('nut_base',shifted(closed_nut,[X[0],0,0]),base)]:
 hit=tm.boolean.intersection([a,b],engine='manifold')
 pairs[name]=float(abs(hit.volume)) if hit is not None else 0.
REPORT['checks']['closed_design_intersection_mm3']=pairs
REPORT['checks']['closed_design_no_volume_overlap']=all(x<.01 for x in pairs.values())
REPORT['checks']['ring_anchors']=ring_audits
REPORT['checks']['composition_errors']=[str(x) for x in Usd.Stage.Open(str(candidate)).GetCompositionErrors()]
REPORT['checks']['printable_body_count_pass']=all(v['connected_bodies']==1 for v in REPORT['parts'].values())
REPORT['checks']['female_thread_radial_clearance_mm']=.30
REPORT['checks']['ring_shank_radial_clearance_mm']=.70
REPORT['checks']['head_to_fence_radial_clearance_mm']=2.9
REPORT['checks']['screw_tip_to_base_floor_mm']=1.
REPORT['limitations']=['No Kit or PhysX run in this build.','Full helical contact is authored; no kinematic screw-lock shortcut.','FDM thread clearance requires coupon test.','PETG mass and friction are prototype assumptions.','Fixture static boundary requires actual bench attachment through mounting holes.','Original cable body poses retained; inherited cable/table contacts not dynamically accepted.','Printed ring barrel/TPU cable retention is untested.']
REPORT['candidate_scene']=str(candidate)
(OUT/'geometry_audit.json').write_text(json.dumps(REPORT,indent=2))
config={'schema_version':1,'units':'m','stl_units':'mm','thread':{'name':'CUSTOM_D10_P2','major_diameter_m':.010,'pitch_m':.002,'radial_clearance_m':.0003,'true_helical_mesh':True,'validated':False},
 'fixture_origin_world_m':ORIGIN.tolist(),'seat_world_m':[(ORIGIN+[x/1000,0,.018]).tolist() for x in X],
 'seat_paths':[f'/PCB_4_Terminal/Terminal_{i}_Seat' for i in range(1,5)],'ring_paths':[f'/wire{i}/_0347300_ring_terminal' for i in range(1,5)],
 'screw_paths':[f'/Screw_{i}' for i in range(1,5)],'nut_paths':[f'/NutPlate_Coarse/Nut_{i}' for i in range(1,5)],
 'collision_mode':'static triangle surfaces; dynamic SDF256; not Kit-validated','physics_ready':False}
(OUT/'geometry_config.json').write_text(json.dumps(config,indent=2))
print(json.dumps(REPORT['checks'],indent=2),flush=True)
assert REPORT['checks']['closed_design_no_volume_overlap'],pairs
assert not REPORT['checks']['composition_errors']
assert REPORT['checks']['printable_body_count_pass']
print('BUILD_STATIC_PASS',flush=True)
