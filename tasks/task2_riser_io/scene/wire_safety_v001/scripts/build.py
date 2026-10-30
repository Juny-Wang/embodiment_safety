"""Offline build. All outputs isolated; does not change Task2 main stage."""
from pathlib import Path
import json, math, hashlib, shutil, ast
import numpy as np
import trimesh as tm
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf
from PIL import Image, ImageDraw, ImageFont
R=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io')
O=R/'scene/wire_safety_v001'
for d in ['stl','assets','reports','runs','scripts']: (O/d).mkdir(exist_ok=True)
source=R/'scene/riser_io_preview.usda'
s=Usd.Stage.Open(str(source)); xc=UsdGeom.XformCache()
audit={'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'wires':{},'source_unchanged':True}
for w in range(1,5):
 root=s.GetPrimAtPath(f'/wire{w}');segs=[];joints=[];badq=[]
 for p in Usd.PrimRange(root):
  if p.IsA(UsdGeom.Capsule):
   m=xc.GetLocalToWorldTransform(p);radius=float(p.GetAttribute('radius').Get())*m.TransformDir(Gf.Vec3d(0,1,0)).GetLength()
   segs.append({'path':str(p.GetPath()),'world_center_m':list(m.ExtractTranslation()),'radius_m':radius,'kinematic':p.GetAttribute('physics:kinematicEnabled').Get(),'collision':p.HasAPI(UsdPhysics.CollisionAPI),'schemas':p.GetAppliedSchemas(),'material':str(UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial('physics')[0].GetPath())})
  if p.IsA(UsdPhysics.Joint):
   j=UsdPhysics.Joint(p);p0=j.GetBody0Rel().GetTargets();p1=j.GetBody1Rel().GetTargets()
   pos0=j.GetLocalPos0Attr().Get();pos1=j.GetLocalPos1Attr().Get()
   a=xc.GetLocalToWorldTransform(s.GetPrimAtPath(p0[0])).Transform(Gf.Vec3d(pos0)) if p0 else Gf.Vec3d(pos0)
   b=xc.GetLocalToWorldTransform(s.GetPrimAtPath(p1[0])).Transform(Gf.Vec3d(pos1)) if p1 else Gf.Vec3d(pos1)
   qn=[float(j.GetLocalRot0Attr().Get().GetLength()),float(j.GetLocalRot1Attr().Get().GetLength())]
   if any(abs(n-1)>.001 for n in qn):badq.append({'path':str(p.GetPath()),'norms':qn})
   joints.append({'path':str(p.GetPath()),'type':p.GetTypeName(),'axis':p.GetAttribute('physics:axis').Get(),'body0':[str(x) for x in p0],'body1':[str(x) for x in p1],'anchor_error_mm':float((a-b).GetLength()*1000),'quaternion_norms':qn,'drive':{k:p.GetAttribute('drive:angular:physics:'+k).Get() for k in ['stiffness','damping','maxForce']}})
 audit['wires'][str(w)]={'segments':segs,'joints':joints,'non_unit_joint_quaternions':badq,'anchor_error_max_mm':max(j['anchor_error_mm'] for j in joints)}
audit['findings']=['Revolute chain is a proxy, not a calibrated 3-D elastic cable.','Kinematic proximal segment is not a measured connector strain sensor.','Drive maxForce is not measured tension.','Nonunit joint quaternions and joint anchor mismatches require source repair before promotion.']
(O/'reports/source_audit.json').write_text(json.dumps(audit,indent=2))

def cyl(r,h,p=(0,0,0),sections=96):
 g=tm.creation.cylinder(r,h,sections=sections);g.apply_translation(p);return g
def box(ext,p):
 g=tm.creation.box(ext);g.apply_translation(p);return g
def union(gs): return tm.boolean.union(gs,engine='manifold')
def subtract(a,bs):return tm.boolean.difference([a]+bs,engine='manifold')
parts={}
# All STL coordinates in mm. Printable fixture prototypes; bench attachment boundary explicit.
parts['mount_tile_160x120']=subtract(box([160,120,6],[0,0,3]),[cyl(5.7,12,[x,y,3]) for x in [-54,-27,0,27,54] for y in [-40,0,40]])
foot=subtract(box([76,46,5],[0,0,2.5]),[cyl(5.7,12,[x,0,2.5]) for x in [-27,27]])
# rounded top; vertical contact wall; same triangles used for collision.
parts['guide_column_D36_H35']=union([foot,cyl(18,32,[0,0,21]),tm.creation.icosphere(subdivisions=3,radius=18).apply_scale([1,1,.17]).apply_translation([0,0,37])])
# Split collar: actual 7.4 mm passage for 7 mm wire, no false claim of preload.
# Pair is a locator until compliance/preload is measured.
body=box([64,26,12],[0,0,6])
bore=cyl(3.7,70);bore.apply_transform(tm.transformations.rotation_matrix(math.pi/2,[0,1,0]));bore.apply_translation([0,0,12])
holes=[cyl(5.7,35,[0,y,12]) for y in [-10,10]]
# mount screws are along X ends instead, so bores and screws cannot intersect.
body=box([76,44,12],[0,0,6])
holes=[cyl(5.7,35,[x,0,12]) for x in [-27,27]]
# cable axis Y, screw holes X; preserve side walls
bore=cyl(3.7,60);bore.apply_transform(tm.transformations.rotation_matrix(math.pi/2,[1,0,0]));bore.apply_translation([0,0,12])
parts['relief_collar_lower_candidate']=subtract(body,[bore]+holes)
parts['relief_collar_upper_candidate']=subtract(box([76,44,8],[0,0,16]),[bore]+holes)
# Reuse actual full-print threaded hardware as separate files, not metal hardware.
previous=R/'scene/printable_v001/stl'
for n in ['screw_custom_D10_P2','nut_custom_D10_P2_clearance_030']:
 parts[n]=tm.load_mesh(previous/(n+'.stl'))
# Reuse only the audited thread-generating functions; never execute the old builder.
tree=ast.parse((R/'scripts/build_printable_v001.py').read_text())
ns={'np':np,'tm':tm,'math':math}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['radius','thread_volume']],type_ignores=[]),'<thread_geometry>', 'exec'),ns)
parts['screw_custom_D10_P2_shaft36']=union([ns['thread_volume'](36,taper=True),cyl(8.5,5,[0,0,38.5],sections=12)])
report={}
for name,g in parts.items():
 assert g.is_watertight and g.is_volume,(name,'not closed')
 g.export(O/'stl'/(name+'.stl'))
 report[name]={'dimensions_mm':g.extents.tolist(),'bounds_mm':g.bounds.tolist(),'watertight':bool(g.is_watertight),'connected_bodies':int(g.body_count),'volume_mm3':float(g.volume)}
(O/'reports/print_geometry.json').write_text(json.dumps(report,indent=2))

def attr(p,n,t,v):p.CreateAttribute(n,t).Set(v)
def material(stage):
 m=UsdShade.Material.Define(stage,'/World/Materials/CableContact')
 a=UsdPhysics.MaterialAPI.Apply(m.GetPrim());a.CreateStaticFrictionAttr(.4);a.CreateDynamicFrictionAttr(.3);a.CreateRestitutionAttr(0)
 return m
def collide(p,mat):
 UsdPhysics.CollisionAPI.Apply(p).CreateCollisionEnabledAttr(True)
 p.AddAppliedSchema('PhysxCollisionAPI');attr(p,'physxCollision:contactOffset',Sdf.ValueTypeNames.Float,.0001);attr(p,'physxCollision:restOffset',Sdf.ValueTypeNames.Float,0.)
 UsdShade.MaterialBindingAPI.Apply(p).Bind(mat,materialPurpose='physics')
def mesh(stage,path,g,mat,pos=(0,0,0)):
 m=UsdGeom.Mesh.Define(stage,path);m.CreatePointsAttr([Gf.Vec3f(*(v*.001)) for v in g.vertices]);m.CreateFaceVertexCountsAttr([3]*len(g.faces));m.CreateFaceVertexIndicesAttr(g.faces.flatten().tolist());m.CreateSubdivisionSchemeAttr('none');m.CreateDisplayColorAttr([Gf.Vec3f(.23,.40,.57)])
 UsdGeom.Xformable(m).AddTranslateOp().Set(Gf.Vec3d(*pos));collide(m.GetPrim(),mat);UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('none')
 return m
def make_stage(name,n_guides=0):
 out=O/(name+'.usda');stage=Usd.Stage.CreateNew(str(out))
 UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.Xform.Define(stage,'/World');stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
 phys=UsdPhysics.Scene.Define(stage,'/World/PhysicsScene');phys.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));phys.CreateGravityMagnitudeAttr(9.81)
 phys.GetPrim().AddAppliedSchema('PhysxSceneAPI');attr(phys.GetPrim(),'physxScene:enableGPUDynamics',Sdf.ValueTypeNames.Bool,False);attr(phys.GetPrim(),'physxScene:timeStepsPerSecond',Sdf.ValueTypeNames.UInt,960)
 attr(phys.GetPrim(),'physxScene:solverType',Sdf.ValueTypeNames.Token,'TGS')
 mat=material(stage)
 floor=UsdGeom.Cube.Define(stage,'/World/Table');floor.CreateSizeAttr(1)
 UsdGeom.Xformable(floor).AddTranslateOp().Set(Gf.Vec3d(.23,0,-.01));UsdGeom.Xformable(floor).AddScaleOp().Set(Gf.Vec3f(.7,.35,.02));collide(floor.GetPrim(),mat)
 floor.CreateDisplayColorAttr([Gf.Vec3f(.5,.52,.55)])
 # Actual unit-metre diagnostic cable: 24 spherical-linked capsules, no frame writes during simulation.
 cable=UsdGeom.Xform.Define(stage,'/World/Cable');UsdPhysics.ArticulationRootAPI.Apply(cable.GetPrim())
 cable.GetPrim().AddAppliedSchema('PhysxArticulationAPI');attr(cable.GetPrim(),'physxArticulation:enabledSelfCollisions',Sdf.ValueTypeNames.Bool,True)
 attr(cable.GetPrim(),'physxArticulation:solverPositionIterationCount',Sdf.ValueTypeNames.Int,64);attr(cable.GetPrim(),'physxArticulation:solverVelocityIterationCount',Sdf.ValueTypeNames.Int,16)
 points=np.array([[i*.46/24,0,.010] for i in range(25)])
 segpaths=[]
 for i,(a,b) in enumerate(zip(points[:-1],points[1:])):
  path=f'/World/Cable/Segment_{i:02d}';segpaths.append(path)
  cap=UsdGeom.Capsule.Define(stage,path);cap.CreateAxisAttr('X');cap.CreateRadiusAttr(.0035);cap.CreateHeightAttr(float(np.linalg.norm(b-a)))
  UsdGeom.Xformable(cap).AddTranslateOp().Set(Gf.Vec3d(*((a+b)/2)))
  cap.CreateDisplayColorAttr([Gf.Vec3f(.08,.45,.7)])
  p=cap.GetPrim();UsdPhysics.RigidBodyAPI.Apply(p).CreateRigidBodyEnabledAttr(True);mass=UsdPhysics.MassAPI.Apply(p);mass.CreateMassAttr(.001)
  p.AddAppliedSchema('PhysxRigidBodyAPI');attr(p,'physxRigidBody:enableCCD',Sdf.ValueTypeNames.Bool,True);attr(p,'physxRigidBody:linearDamping',Sdf.ValueTypeNames.Float,.02);attr(p,'physxRigidBody:angularDamping',Sdf.ValueTypeNames.Float,.02)
  p.AddAppliedSchema('PhysxContactReportAPI');attr(p,'physxContactReport:threshold',Sdf.ValueTypeNames.Float,0.)
  collide(p,mat)
  if i:
   joint=UsdPhysics.SphericalJoint.Define(stage,f'/World/Cable/Joint_{i-1:02d}')
   joint.CreateBody0Rel().SetTargets([segpaths[i-1]]);joint.CreateBody1Rel().SetTargets([path]);joint.CreateLocalPos0Attr(Gf.Vec3f(.46/48,0,0));joint.CreateLocalPos1Attr(Gf.Vec3f(-.46/48,0,0))
   joint.CreateAxisAttr('X');joint.CreateConeAngle0LimitAttr(70);joint.CreateConeAngle1LimitAttr(70);joint.CreateCollisionEnabledAttr(False);UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(segpaths[i-1])).CreateFilteredPairsRel().AddTarget(path)
 stub=UsdGeom.Cube.Define(stage,'/World/Cable/ConnectorStub');stub.CreateSizeAttr(.02);UsdGeom.Xformable(stub).AddTranslateOp().Set(Gf.Vec3d(-.014,0,.01));UsdPhysics.RigidBodyAPI.Apply(stub.GetPrim());UsdPhysics.MassAPI.Apply(stub.GetPrim()).CreateMassAttr(.05);collide(stub.GetPrim(),mat)
 anchor=UsdPhysics.FixedJoint.Define(stage,'/World/Cable/AnchorJoint');anchor.CreateBody1Rel().SetTargets([stub.GetPath()]);anchor.CreateLocalPos0Attr(Gf.Vec3f(-.014,0,.01));anchor.CreateLocalPos1Attr(Gf.Vec3f(0,0,0))
 attachment=UsdPhysics.FixedJoint.Define(stage,'/World/Cable/CableAttachment');attachment.CreateBody0Rel().SetTargets([stub.GetPath()]);attachment.CreateBody1Rel().SetTargets([segpaths[0]]);attachment.CreateLocalPos0Attr(Gf.Vec3f(.014,0,0));attachment.CreateLocalPos1Attr(Gf.Vec3f(-.46/48,0,0));attachment.CreateCollisionEnabledAttr(False);UsdPhysics.FilteredPairsAPI.Apply(stub.GetPrim()).CreateFilteredPairsRel().AddTarget(segpaths[0])
 guidecenters=[(.20,.040,0),(.32,-.040,0)]
 for i,p in enumerate(guidecenters[:n_guides]):mesh(stage,f'/World/Guides/Post_{i+1}',parts['guide_column_D36_H35'],mat,p)
 # Force sensor positive control, separate from cable chain
 c=UsdGeom.Cube.Define(stage,'/World/ControlBox');c.CreateSizeAttr(.02);UsdGeom.Xformable(c).AddTranslateOp().Set(Gf.Vec3d(.50,.10,.02));UsdPhysics.RigidBodyAPI.Apply(c.GetPrim());UsdPhysics.MassAPI.Apply(c.GetPrim()).CreateMassAttr(.1);collide(c.GetPrim(),mat)
 c.GetPrim().AddAppliedSchema('PhysxRigidBodyAPI');attr(c.GetPrim(),'physxRigidBody:sleepThreshold',Sdf.ValueTypeNames.Float,0.);c.GetPrim().AddAppliedSchema('PhysxContactReportAPI');attr(c.GetPrim(),'physxContactReport:threshold',Sdf.ValueTypeNames.Float,0.)
 stage.GetRootLayer().customLayerData={'diagnosticOnly':True,'notMaterialCalibrated':True,'sourceSceneUntouched':True,'strainReliefClampValidated':False}
 stage.GetRootLayer().Save()
 return str(out)
scenes={key:make_stage(key,n) for key,n in [('L0_sensor_bench',0),('L1_one_guide',1),('L2_two_guides',2)]}
# Fixture assembly for inspection/printing; collar not silently inserted into an already laid cable.
st=Usd.Stage.CreateNew(str(O/'fixture_assembly.usda'));UsdGeom.SetStageMetersPerUnit(st,1);UsdGeom.SetStageUpAxis(st,'Z');UsdGeom.Xform.Define(st,'/World');st.SetDefaultPrim(st.GetPrimAtPath('/World'));mat=material(st)
mesh(st,'/World/Tile',parts['mount_tile_160x120'],mat)
mesh(st,'/World/Guide',parts['guide_column_D36_H35'],mat,(0,0,.006))
mesh(st,'/World/CollarLower',parts['relief_collar_lower_candidate'],mat,(.12,0,0))
mesh(st,'/World/CollarUpper',parts['relief_collar_upper_candidate'],mat,(.12,0,0))
st.GetRootLayer().Save()
cfg={'version':1,'status':'ISOLATED_DIAGNOSTIC_NOT_TRAINING','units':{'length':'m','force':'N','moment':'N*m'},'scenes':scenes,'source_stage':str(source),'source_sha256':audit['sha256'],'wire':{'diameter_m':.007,'centerline_length_m':.46,'segments':24,'mass_kg':.024,'model':'spherical rigid-link chain; no calibrated elastic or axial constitutive law','material_stress_available':False},'thresholds':{'source':'DEMONSTRATION_ONLY_NOT_DAMAGE_LIMIT','connector_force_N':1.,'minimum_bend_radius_m':.035,'guide_normal_force_N':1.,'warning_ratio':.7,'mode':'warning_only'},'variants':{'L0':{'guides':0},'L1':{'guides':1},'L2':{'guides':2},'length_and_stiffness_randomization':{'enabled':False,'reason':'requires calibrated material and feasible route checks'}},'gates':{'promote_to_main':False,'strain_relief_effect_verified':False,'full_task_ready':False},'collar':{'bore_m':.0074,'wire_m':.007,'radial_clearance_m':.0002,'preload_N':None,'claim':'locator candidate only; no axial load relief claimed without measured preload/compliance'}}
(O/'config.json').write_text(json.dumps(cfg,indent=2))
im=Image.new('RGB',(1400,780),'#f4f7fa');d=ImageDraw.Draw(im);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
d.text((35,25),'TASK2 / WIRE SAFETY v001 - isolated diagnostic, not accepted routing',font=font,fill='#18364b')
for row,n in enumerate([0,1,2]):
 y=180+row*190;d.text((35,y-60),f'L{n}: {n} guide(s); real force application at distal end',font=font,fill='#18364b')
 d.line([(120,y),(1150,y)],fill='#2778aa',width=12);d.rectangle((85,y-25,120,y+25),fill='#64778a')
 for i,(x,side) in enumerate([(570,-1),(840,1)][:n]):
  cy=y+side*58;d.ellipse((x-42,cy-42,x+42,cy+42),fill='#bba188',outline='#5a4730',width=2)
 d.line((1150,y,1230,y),fill='#cf5246',width=5);d.polygon([(1230,y),(1215,y-9),(1215,y+9)],fill='#cf5246')
d.text((35,725),'Straight reset -> measured response. Final S-route / real strain relief NOT yet validated.',font=font,fill='#ad412b');im.save(O/'layout.png')
print('BUILT',O)
print('SOURCE_FINDINGS',json.dumps({k:{'max_anchor_error_mm':v['anchor_error_max_mm'],'bad_joint_quaternions':v['non_unit_joint_quaternions']} for k,v in audit['wires'].items()}))
