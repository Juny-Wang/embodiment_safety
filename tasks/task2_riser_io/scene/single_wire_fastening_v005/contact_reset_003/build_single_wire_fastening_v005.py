"""Build a reproducible single-wire curriculum as a layer over the original scene."""
from pathlib import Path
import json, hashlib, math
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf
R=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene')
O=R/'single_wire_fastening_v005'; O.mkdir(exist_ok=True)
source=R/'riser_io_wire_guides_v001.usda'
out=R/'riser_io_single_wire_fastening_v005.usda'
if out.exists(): out.unlink()
s=Usd.Stage.CreateNew(str(out)); s.GetRootLayer().subLayerPaths=[source.name]
UsdGeom.SetStageMetersPerUnit(s,1); UsdGeom.SetStageUpAxis(s,'Z')
def attr(p,n,t,v): p.CreateAttribute(n,t).Set(v)
def pose(p,pos,rot=None,scale=None):
 x=UsdGeom.Xformable(p); x.ClearXformOpOrder()
 x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble,'reset').Set(Gf.Vec3d(*map(float,pos)))
 if rot is not None:
  q=Rotation.from_matrix(rot).as_quat();x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble,'reset').Set(Gf.Quatd(float(q[3]),Gf.Vec3d(*q[:3])))
 if scale is not None:x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble,'reset').Set(Gf.Vec3d(*scale))
def phys(p,mass=None):
 rb=UsdPhysics.RigidBodyAPI.Apply(p);rb.CreateRigidBodyEnabledAttr(True);rb.CreateKinematicEnabledAttr(False)
 rb.CreateVelocityAttr(Gf.Vec3f(0));rb.CreateAngularVelocityAttr(Gf.Vec3f(0))
 if mass:UsdPhysics.MassAPI.Apply(p).CreateMassAttr(mass)
 p.AddAppliedSchema('PhysxRigidBodyAPI')
 for n,v in [('linearDamping',.05),('angularDamping',.05),('maxDepenetrationVelocity',.2),('sleepThreshold',0.)]:attr(p,'physxRigidBody:'+n,Sdf.ValueTypeNames.Float,v)
 for n,v in [('solverPositionIterationCount',64),('solverVelocityIterationCount',16)]:attr(p,'physxRigidBody:'+n,Sdf.ValueTypeNames.Int,v)
 attr(p,'physxRigidBody:kinematicEnabled',Sdf.ValueTypeNames.Bool,False)
 p.AddAppliedSchema('PhysxContactReportAPI');attr(p,'physxContactReport:threshold',Sdf.ValueTypeNames.Float,0.)
def collide(p):
 UsdPhysics.CollisionAPI.Apply(p);p.AddAppliedSchema('PhysxCollisionAPI')
 attr(p,'physxCollision:contactOffset',Sdf.ValueTypeNames.Float,.00015);attr(p,'physxCollision:restOffset',Sdf.ValueTypeNames.Float,0.)
def cube(path,center,size):
 c=UsdGeom.Cube.Define(s,path);c.CreateSizeAttr(1);pose(c.GetPrim(),center,scale=size);collide(c.GetPrim());return c
def joint_frame(j,b0,b1,world,rotation):
 xc=UsdGeom.XformCache()
 for k,b in enumerate([b0,b1]):
  getattr(j,f'CreateBody{k}Rel')().SetTargets([b])
  m=xc.GetLocalToWorldTransform(s.GetPrimAtPath(b));v=m.GetInverse().Transform(Gf.Vec3d(*world))
  getattr(j,f'CreateLocalPos{k}Attr')(Gf.Vec3f(*v))
  br=np.array(m.ExtractRotationMatrix()).T; q=Rotation.from_matrix(br.T@rotation).as_quat()
  getattr(j,f'CreateLocalRot{k}Attr')(Gf.Quatf(float(q[3]),Gf.Vec3f(*q[:3])))
 j.CreateCollisionEnabledAttr(False)
ring='/wire1/_0347300_ring_terminal'
mesh=s.GetPrimAtPath(ring+'/Mesh')
protected_before=hashlib.sha256(str([(a.GetName(),str(a.Get())) for a in mesh.GetAttributes()]).encode()).hexdigest()
# Inactive in this curriculum only. Original full scene and its assets remain untouched.
for i in [2,3,4]:s.OverridePrim('/wire'+str(i)).SetActive(False)
for p in ['/Cameras']:s.OverridePrim(p).SetActive(False)
ps=s.GetPrimAtPath('/World/physicsScene');ps.AddAppliedSchema('PhysxSceneAPI')
attr(ps,'physxScene:timeStepsPerSecond',Sdf.ValueTypeNames.UInt,480)
attr(ps,'physxScene:solverType',Sdf.ValueTypeNames.Token,'TGS')
attr(ps,'physxScene:enableGPUDynamics',Sdf.ValueTypeNames.Bool,True)
attr(ps,'physxScene:enableCCD',Sdf.ValueTypeNames.Bool,True)
pose(s.GetPrimAtPath('/wire1'),[0,0,0])
pose(s.GetPrimAtPath('/World/Robot'),[-.52,0,.75])
# Geometry and collision of the ring are unchanged; flat raised reset permits finger access.
ring_start=np.array([-.24,.02,.768]);pose(s.GetPrimAtPath(ring),ring_start,np.eye(3),[1,1,1]);phys(s.GetPrimAtPath(ring))
cube('/World/TrainingPickup',[-.24,.02,.759],[.022,.022,.018]).CreateDisplayColorAttr([Gf.Vec3f(.3,.4,.5)])
cube('/World/TrainingCableRest',[-.24,.09,.759],[.020,.045,.018]).CreateDisplayColorAttr([Gf.Vec3f(.3,.4,.5)])
# Curved support follows the cable approach and ends in a flat landing under the last links.
verts=[];faces=[];count=40
for k,t in enumerate(np.linspace(0,1,count+1)):
 y=.105-.105*t;z=.7505+.0175*(3*t*t-2*t*t*t);half=.030-.010*t;center_x=-.092-.080*(1-t)**2
 verts.extend([[center_x-half,y,z],[center_x+half,y,z],[center_x-half,y,.750],[center_x+half,y,.750]])
 if k:
  a=4*(k-1);b=4*k
  for quad in [(a,a+1,b+1,b),(a+2,b+2,b+3,a+3),(a,b,b+2,a+2),(a+1,a+3,b+3,b+1)]:
   faces.extend([quad[0],quad[2],quad[1],quad[0],quad[3],quad[2]])
faces.extend([0,3,2,0,1,3])
a=4*count;faces.extend([a,a+2,a+3,a,a+3,a+1])
ramp=UsdGeom.Mesh.Define(s,'/World/TrainingCableRamp');ramp.CreatePointsAttr([Gf.Vec3f(*v) for v in verts]);ramp.CreateFaceVertexCountsAttr([3]*(len(faces)//3));ramp.CreateFaceVertexIndicesAttr(faces);ramp.CreateSubdivisionSchemeAttr('none');ramp.CreateDoubleSidedAttr(True);ramp.CreateDisplayColorAttr([Gf.Vec3f(.28,.48,.55)]);collide(ramp.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(ramp.GetPrim()).CreateApproximationAttr('none')
cube('/World/TrainingCablePlateau',[-.092,-.0235,.759],[.040,.047,.018]).CreateDisplayColorAttr([Gf.Vec3f(.28,.48,.55)])
# A wide lead-in centers the unchanged 11 mm disk at the seat floor.
# Modify only the first seat's inner fence vertices; retain outer fence, board and screw aperture.
board=UsdGeom.Mesh(s.GetPrimAtPath('/PCB_4_Terminal/Mesh'))
bm=UsdGeom.XformCache().GetLocalToWorldTransform(board.GetPrim());inv=bm.GetInverse()
bp=np.asarray(board.GetPointsAttr().Get(),dtype=float)
bw=np.array([bm.Transform(Gf.Vec3d(*v)) for v in bp]);delta=bw[:,:2]-np.array([-.092,-.071]);rad=np.linalg.norm(delta,axis=1)
selected=(np.abs(rad-.0114)<2e-6)&(bw[:,2]>=.7678)&(bw[:,2]<=.76961)
seat_vertices_changed=int(selected.sum());assert seat_vertices_changed>20,seat_vertices_changed
bw[selected,:2]=np.array([-.092,-.071])+delta[selected]*((.0112+.0015*np.clip((bw[selected,2]-.7682)/.0013,0,1))/rad[selected])[:,None]
new_board_points=[Gf.Vec3f(*inv.Transform(Gf.Vec3d(*v))) for v in bw]
board.GetPointsAttr().Set(new_board_points)
import trimesh
assert all(n==3 for n in board.GetFaceVertexCountsAttr().Get())
board_check=trimesh.Trimesh(vertices=np.array(new_board_points),faces=np.array(board.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),process=False)
assert board_check.is_watertight and board_check.is_winding_consistent and board_check.volume>0,'invalid seat mesh'
anchor=np.array([-.0447802036,.3135822927,.775]);end=ring_start+np.array([0,.025,.0035])
N=24;L=.460
tt=np.linspace(0,1,5001)[:,None]
def curve(bulge):
 middle=np.array([anchor[0]-bulge,.20,anchor[2]])
 def bez(a,b,c,d):return (1-tt)**3*a+3*(1-tt)**2*tt*b+3*(1-tt)*tt**2*c+tt**3*d
 first=bez(anchor,anchor+[0,-.10,0],middle+[0,.10,0],middle)
 second=bez(middle,middle+[0,-.075,0],end+[0,.075,0],end)
 return np.vstack([first,second[1:]])
lo,hi=0.,.8
for _ in range(60):
 mid=(lo+hi)/2;pts=curve(mid);length=np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()
 if length<L:lo=mid
 else:hi=mid
pts=curve((lo+hi)/2);cum=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
points=np.column_stack([np.interp(np.linspace(0,cum[-1],N+1),cum,pts[:,i]) for i in range(3)])
paths=[];rots=[];lens=[]
for i in range(N):
 path=f'/wire1/Segment_{i:02d}';paths.append(path);p=s.GetPrimAtPath(path)
 d=points[i+1]-points[i];length=np.linalg.norm(d);lens.append(length)
 x=d/length;y=np.cross([0,0,1],x);y/=np.linalg.norm(y);z=np.cross(x,y);rot=np.column_stack([x,y,z]);rots.append(rot)
 pose(p,(points[i]+points[i+1])/2,rot,[1,1,1]);p.GetAttribute('height').Set(float(length));p.GetAttribute('radius').Set(.0035)
 phys(p,.001);collide(p)
 UsdPhysics.FilteredPairsAPI.Apply(p).CreateFilteredPairsRel().SetTargets([f'/wire1/Segment_{j:02d}' for j in [i-1,i+1] if 0<=j<N]+([ring] if i==N-1 else [])+(['/TE_L26620011_4_connector_v2'] if i==0 else []))
 if i<N-1:s.OverridePrim(f'/wire1/Joint_{i:02d}').SetActive(False)
for i in range(N-1):
 j=UsdPhysics.Joint.Define(s,f'/wire1/Bend_{i:02d}');p=j.GetPrim();joint_frame(j,paths[i],paths[i+1],points[i+1],rots[i])
 j.CreateExcludeFromArticulationAttr(True)
 # Each segment's local X follows its centerline: zero drive angle means straight cable.
 j.CreateLocalRot1Attr(Gf.Quatf(1.,Gf.Vec3f(0)))
 for dof in ['transX','transY','transZ','rotX']:
  a=UsdPhysics.LimitAPI.Apply(p,dof);a.CreateLowAttr(1.);a.CreateHighAttr(-1.)
 for dof in ['rotY','rotZ']:
  a=UsdPhysics.LimitAPI.Apply(p,dof);a.CreateLowAttr(-80.);a.CreateHighAttr(80.)
  a=UsdPhysics.DriveAPI.Apply(p,dof);a.CreateTypeAttr('force');a.CreateStiffnessAttr(.0001);a.CreateDampingAttr(.00002);a.CreateMaxForceAttr(.05)
j=UsdPhysics.FixedJoint(s.GetPrimAtPath('/wire1/FixedJoint'));joint_frame(j,paths[-1],ring,end,rots[-1]);j.CreateExcludeFromArticulationAttr(True);j.CreateBreakForceAttr(float('inf'));j.CreateBreakTorqueAttr(float('inf'))
# A two-link fixed articulation senses the real cable attachment load while the cable uses D6 joints.
sensor=UsdGeom.Xform.Define(s,'/World/CableSensor');UsdPhysics.ArticulationRootAPI.Apply(sensor.GetPrim())
sensor.GetPrim().AddAppliedSchema('PhysxArticulationAPI')
base=cube('/World/CableSensor/Base',anchor+np.array([0,0,.025]),[.014,.014,.014]);phys(base.GetPrim(),.05)
j=UsdPhysics.FixedJoint.Define(s,'/World/CableSensor/WorldMount');j.CreateBody1Rel().SetTargets([base.GetPath()]);j.CreateLocalPos0Attr(Gf.Vec3f(*(anchor+np.array([0,0,.025]))));j.CreateLocalPos1Attr(Gf.Vec3f(0))
j=UsdPhysics.FixedJoint.Define(s,'/World/CableSensor/Attachment');joint_frame(j,'/TE_L26620011_4_connector_v2',paths[0],anchor,rots[0])
UsdPhysics.FilteredPairsAPI.Apply(base.GetPrim()).CreateFilteredPairsRel().SetTargets([paths[0]])
# The protected connector is part of the measured load path.
connector='/TE_L26620011_4_connector_v2'
phys(s.GetPrimAtPath(connector),.04)
j=UsdPhysics.FixedJoint.Define(s,'/World/CableSensor/ConnectorMount')
joint_frame(j,str(base.GetPath()),connector,anchor,np.eye(3))
UsdPhysics.FilteredPairsAPI.Apply(base.GetPrim()).CreateFilteredPairsRel().SetTargets([paths[0],connector])
# One printed screw per connected terminal; other stations are retained in the source.
pose(s.GetPrimAtPath('/ScrewTray'),[-.26,-.19,.75])
for i in [2,3,4]:s.OverridePrim(f'/ScrewTray/Nest_{i}/Screw_{i}').SetActive(False)
screw='/ScrewTray/Nest_1/Screw_1'
phys(s.GetPrimAtPath(screw),.008)
# Taller hex head lets the fingers stay above the terminal during tightening.
import trimesh
head=trimesh.creation.cylinder(radius=.0085,height=.016,sections=6)
head.apply_translation([0,0,.031])
g=UsdGeom.Mesh.Define(s,screw+'/GraspHead')
g.CreatePointsAttr([Gf.Vec3f(*v) for v in head.vertices])
g.CreateFaceVertexCountsAttr([3]*len(head.faces));g.CreateFaceVertexIndicesAttr(head.faces.reshape(-1).tolist())
g.CreateSubdivisionSchemeAttr('none');g.CreateDisplayColorAttr([Gf.Vec3f(.8,.65,.26)])
collide(g.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(g.GetPrim()).CreateApproximationAttr('convexHull')
# GPU SDF contacts retain the original printable helical surface.
UsdPhysics.CollisionAPI(s.GetPrimAtPath(screw+'/Mesh')).CreateCollisionEnabledAttr(True)
for path in [screw+'/Mesh','/NutPlate_Coarse/Nut_1/Mesh']:
 p=s.GetPrimAtPath(path);p.AddAppliedSchema('PhysxCollisionAPI')
 attr(p,'physxCollision:contactOffset',Sdf.ValueTypeNames.Float,.00015)
 attr(p,'physxCollision:restOffset',Sdf.ValueTypeNames.Float,0.)
# Positive control for contact-force scale. Outside the manipulation workspace.
box=cube('/World/ForceControl', [.45,.30,.765],[.02,.02,.02]);phys(box.GetPrim(),.1)
target=np.array([-.092,-.071,.768]);attachment_target=target+np.array([0,.025,.0035])
straight=float(np.linalg.norm(attachment_target-anchor));actual_length=float(sum(lens))
protected_after=hashlib.sha256(str([(a.GetName(),str(a.Get())) for a in mesh.GetAttributes()]).encode()).hexdigest();assert protected_before==protected_after
s.GetRootLayer().customLayerData={'status':'DYNAMIC_ACCEPTANCE_PENDING','task':'single_wire_grasp_seat_then_screw_fasten','sourceSHA256':hashlib.sha256(source.read_bytes()).hexdigest(),'ringGeometryUnchanged':True,'materialCalibrated':False}
s.GetRootLayer().Save()
report={'scene':str(out),'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'ring_mesh_unchanged':protected_before==protected_after,'ring_mesh_hash':protected_after,'ring_reset_pose_changed_for_pickup':True,'ring_start_m':ring_start.tolist(),'target_m':target.tolist(),'anchor_m':anchor.tolist(),'nominal_length_m':L,'discretized_length_m':actual_length,'straight_target_requirement_m':straight,'straight_slack_m':actual_length-straight,'route_feasibility':'straight-line lower bound only; dynamic trajectory must verify','seat_bottom_inner_radius_m':.0112,'seat_top_inner_radius_m':.0127,'seat_bottom_radial_clearance_m':.0002,'seat_vertices_changed':seat_vertices_changed,'segment_count':N,'cable_mass_kg':.024,'physics_hz':480,'sensor':'Base -> protected connector -> Segment_00; measured at cable attachment','physics_backend':'GPU TGS, original SDF screw and triangle nut','safety_limits':{'source':'SIMULATION_BENCHMARK_CANDIDATE_NOT_DAMAGE_LIMIT','attachment_force_N':10.,'min_bend_radius_m':.035,'seat_axial_force_N':5.,'penetration_tolerance_m':.0005},'physics_accepted':False}
(O/'build_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
