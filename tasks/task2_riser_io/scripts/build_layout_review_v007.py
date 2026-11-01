"""Single-wire assembly geometry for layout review."""
from pathlib import Path
import json, math
import numpy as np
import trimesh as tm
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
R=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io')
O=R/'scene/layout_review_v007';O.mkdir(exist_ok=True)
out=R/'scene/riser_io_layout_review_v007.usda'
if out.exists():out.unlink()
s=Usd.Stage.CreateNew(str(out));s.GetRootLayer().subLayerPaths=['riser_io_single_wire_fastening_v005.usda']
s.GetRootLayer().customLayerData={'status':'LAYOUT_REVIEW_ONLY_PHYSICS_PENDING','physics_accepted':False,'full_task_demonstrated':False}
def pose(p,pos,rot=None,scale=None):
 x=UsdGeom.Xformable(p);x.ClearXformOpOrder();x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble,'layout').Set(Gf.Vec3d(*map(float,pos)))
 if rot is not None:
  q=Rotation.from_matrix(rot).as_quat();x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble,'layout').Set(Gf.Quatd(float(q[3]),Gf.Vec3d(*q[:3])))
 if scale is not None:x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble,'layout').Set(Gf.Vec3d(*scale))
def box(size,center):
 m=tm.creation.box(extents=size);m.apply_translation(center);return m
def cyl(r,h,center,axis='Z',sections=64):
 m=tm.creation.cylinder(radius=r,height=h,sections=sections)
 if axis=='Y':m.apply_transform(tm.transformations.rotation_matrix(math.pi/2,[1,0,0]))
 m.apply_translation(center);return m
def diff(a,*b):return tm.boolean.difference([a,*b],engine='manifold')
def union(*m):return tm.boolean.union(list(m),engine='manifold')
colors={'housing':(.20,.25,.30),'base':(.30,.34,.38),'blue':(.13,.40,.58),'green':(.13,.45,.30),'brass':(.64,.45,.16),'copper':(.65,.27,.09),'steel':(.42,.48,.54),'wire':(.82,.47,.09),'table':(.45,.39,.29),'measurement':(.42,.23,.62)}
for name,col in colors.items():
 path='/World/LayoutMaterials/'+name
 mat=UsdShade.Material.Define(s,path);sh=UsdShade.Shader.Define(s,path+'/Shader');sh.CreateIdAttr('UsdPreviewSurface')
 sh.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*col));sh.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.52)
 sh.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(.55 if name in ['steel','copper','brass'] else 0.)
 mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),'surface')
def bind(p,color):UsdShade.MaterialBindingAPI.Apply(p).Bind(UsdShade.Material(s.GetPrimAtPath('/World/LayoutMaterials/'+color)))
parts={}
def mesh(path,g,color):
 assert g.is_watertight and g.is_volume,path
 p=UsdGeom.Mesh.Define(s,path)
 # Geometry is authored in meters relative to its existing parent.
 parent=p.GetPrim().GetParent();inv=UsdGeom.XformCache().GetLocalToWorldTransform(parent).GetInverse()
 v=np.array([inv.Transform(Gf.Vec3d(*map(float,a))) for a in g.vertices])
 p.CreatePointsAttr([Gf.Vec3f(*map(float,a)) for a in v]);p.CreateFaceVertexCountsAttr([3]*len(g.faces));p.CreateFaceVertexIndicesAttr(g.faces.reshape(-1).tolist())
 p.CreateSubdivisionSchemeAttr('none');p.CreateDoubleSidedAttr(False);p.CreateExtentAttr([Gf.Vec3f(*v.min(0)),Gf.Vec3f(*v.max(0))]);bind(p.GetPrim(),color)
 parts[path]={'bounds_world_m':g.bounds.tolist(),'watertight':True}
 return p
for path in ['/Task2WireGuides','/Electronics_140mm_19mm_cage','/Task1PCB','/World/ForceControl','/World/TrainingPickup','/World/TrainingCableRest','/World/TrainingCableRamp','/World/TrainingCablePlateau','/PCB_holder/Mesh','/PCB_4_Terminal/Mesh','/ScrewTray/Mesh']:
 if s.GetPrimAtPath(path):s.OverridePrim(path).SetActive(False)
for i in [2,3,4]:
 for path in [f'/wire{i}',f'/NutPlate_Coarse/Nut_{i}',f'/ScrewTray/Nest_{i}']:
  if s.GetPrimAtPath(path):s.OverridePrim(path).SetActive(False)
old_mesh=s.GetPrimAtPath('/TE_L26620011_4_connector_v2/Mesh')
mat=UsdGeom.XformCache().GetLocalToWorldTransform(old_mesh)
port=np.array(mat.Transform(Gf.Vec3d(-11.43,3.8,8.25)))
s.OverridePrim(old_mesh.GetPath()).SetActive(False)
x,y,z=port
# Insulated wire stops at the entrance; a separate exposed core enters the terminal.
housing=box([.025,.026,.025],[x,y+.011,.7665])
housing=diff(housing,cyl(.0042,.009,[x,y+.001,z],'Y'),cyl(.0016,.030,[x,y+.014,z],'Y'),box([.012,.013,.023],[x,y+.013,z+.0075]))
mesh('/TE_L26620011_4_connector_v2/SingleHousing',housing,'housing')
terminal=diff(box([.011,.012,.012],[x,y+.012,z+.002]),cyl(.0016,.016,[x,y+.012,z],'Y'),cyl(.0017,.010,[x,y+.012,z+.008]))
mesh('/TE_L26620011_4_connector_v2/MetalTerminal',terminal,'brass')
mesh('/TE_L26620011_4_connector_v2/ClampScrew',union(cyl(.0015,.006,[x,y+.012,z+.004]),cyl(.003,.002,[x,y+.012,z+.008],sections=6)),'steel')
# The rear mount supports the protected terminal and provides its future sensing interface.
base=s.GetPrimAtPath('/World/CableSensor/Base')
pose(base,[x,y+.010,.752],scale=[.045,.044,.004]);bind(base,'base')
for dx in [-.018,.018]:
 mesh('/World/TerminalMountBolt'+('L' if dx<0 else 'R'),cyl(.003,.002,[x+dx,y+.010,.755],sections=6),'steel')
# Open support permits lift-off and axial sliding without a threading operation.
gy=y-.037
cutprofile=np.array([[0,-.014],[.008,-.014],[.008,-.012],[.0062,-.010],[.0048,-.007],[.0042,-.004],[.0042,.004],[.0048,.007],[.0062,.010],[.008,.012],[.008,.014],[0,.014]])
cut=tm.creation.revolve(cutprofile,sections=96);cut.apply_transform(tm.transformations.rotation_matrix(math.pi/2,[1,0,0]));cut.apply_translation([x,gy,z])
body=box([.032,.024,.017],[x,gy,.7585])
holecuts=[cyl(.0018,.030,[x+dx,gy,.763]) for dx in [-.012,.012]]
opening=box([.0084,.030,.024],[x,gy,z+.012])
body=diff(body,cut,opening,*holecuts)
mesh('/World/SlidingGuide/OpenSaddle',body,'blue')
for k,dx in enumerate([-.012,.012]):
 mesh('/World/SlidingGuide/MountBolt_'+str(k),union(cyl(.0015,.012,[x+dx,gy,.760]),cyl(.003,.002,[x+dx,gy,.768],sections=6)),'steel')
# Broad service loop with two 50 mm corners and a straight lead through the guide.
ring='/wire1/_0347300_ring_terminal';anchor=port+[0,-.001,0];radius=.050;yturn=.225;L=.460
def layout_path(pickup_x):
 end=np.array([pickup_x,.045,.7715])
 a=np.linspace(anchor[:2],[x,yturn],300)
 th=np.linspace(0,-math.pi/2,240);b=np.column_stack([x-radius+radius*np.cos(th),yturn+radius*np.sin(th)])
 c=np.linspace(b[-1],[pickup_x+radius,yturn-radius],400)
 th=np.linspace(math.pi/2,math.pi,240);d=np.column_stack([pickup_x+radius+radius*np.cos(th),yturn-2*radius+radius*np.sin(th)])
 e=np.linspace(d[-1],end[:2],300)
 xy=np.vstack([a,b[1:],c[1:],d[1:],e[1:]])
 arclength=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
 t=np.clip((arclength-.067)/.078,0,1);zz=z+(.7535-z)*(3*t*t-2*t*t*t)
 last_length=float(np.linalg.norm(end[:2]-d[-1]))
 t=np.clip((arclength-(arclength[-1]-last_length))/last_length,0,1)
 zz+=(end[2]-.7535)*(3*t*t-2*t*t*t)
 return np.column_stack([xy,zz])
lo,hi=-.34,-.25
for _ in range(50):
 mid=(lo+hi)/2;v=layout_path(mid);ell=np.linalg.norm(np.diff(v,axis=0),axis=1).sum()
 if ell>L:lo=mid
 else:hi=mid
pickup_x=(lo+hi)/2;start=np.array([pickup_x,.020,.768]);end=start+[0,.025,.0035]
v=layout_path(pickup_x);cum=np.r_[0,np.cumsum(np.linalg.norm(np.diff(v,axis=0),axis=1))]
pts=np.column_stack([np.interp(np.linspace(0,cum[-1],25),cum,v[:,k]) for k in range(3)])
pose(s.GetPrimAtPath(ring),start,np.eye(3),[1,1,1])
mesh('/World/PickupNest',union(box([.038,.067,.003],[pickup_x,.0395,.7515]),box([.021,.021,.015],[pickup_x,.02,.7605]),box([.009,.042,.015],[pickup_x,.052,.7605])),'blue')
for i in range(24):
 a,b=pts[i:i+2];d=b-a;dx=d/np.linalg.norm(d);dy=np.cross([0,0,1],dx);dy/=np.linalg.norm(dy);dz=np.cross(dx,dy)
 p=s.GetPrimAtPath(f'/wire1/Segment_{i:02d}');pose(p,(a+b)/2,np.column_stack([dx,dy,dz]),[1,1,1]);p.GetAttribute('height').Set(float(np.linalg.norm(d)));bind(p,'wire')
mesh('/wire1/Segment_00/ExposedCore',cyl(.001,.010,[x,y+.007,z],'Y'),'copper')
# One target station with a captive nut and open ring seat.
tx,ty,tz=-.092,-.071,.768
base=diff(box([.046,.048,.012],[tx,ty,.756]),cyl(.01185,.010,[tx,ty,.758],sections=6),cyl(.0061,.018,[tx,ty,.759]))
# Seat load passes through the annular sensing-support location.
# The nut has a separate reaction path into the base.
mesh('/PCB_holder/LoadSupportEnvelope',diff(cyl(.019,.002,[tx,ty,.763]),cyl(.013,.004,[tx,ty,.763])),'measurement')
nut_mesh=UsdGeom.Mesh(s.GetPrimAtPath('/NutPlate_Coarse/Nut_1/Mesh'))
nm=UsdGeom.XformCache().GetLocalToWorldTransform(nut_mesh.GetPrim())
nv=np.array([nm.Transform(Gf.Vec3d(*map(float,a))) for a in nut_mesh.GetPointsAttr().Get()])
nf=np.array(nut_mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
nut=tm.Trimesh(nv+[0,0,-.001],nf,process=True)
s.OverridePrim(nut_mesh.GetPath()).SetActive(False)
mesh('/NutPlate_Coarse/Nut_1/IndependentNut',nut,'steel')
mesh('/PCB_holder/SingleBase',base,'base')
plate=diff(box([.046,.048,.004],[tx,ty,.766]),cyl(.0057,.008,[tx,ty,.766]))
outer=cyl(.0134,.0016,[tx,ty,.7687])
# Tapered lead-in surrounds the annulus; a rear opening clears the cable barrel.
profile=np.array([[0,0],[.0112,0],[.0127,.0016],[0,.0016]])
funnel=tm.creation.revolve(profile,sections=96);funnel.apply_translation([tx,ty,.7679])
fence=diff(outer,funnel,box([.012,.026,.006],[tx,ty+.017,.769]))
mesh('/PCB_4_Terminal/SingleSeat',union(plate,fence),'green')
# Integrated cable landing provides support after release.
yc=np.linspace(ty+.025,ty+.120,33);vv=[]
for yy in yc:
 t=np.clip((yy-(ty+.056))/.064,0,1);top=.768-.0175*(3*t*t-2*t*t*t)
 vv.extend([[tx-.012,yy,.750],[tx+.012,yy,.750],[tx-.012,yy,top],[tx+.012,yy,top]])
ff=[]
for k in range(len(yc)-1):
 a=4*k;b=a+4
 for q in [(a,b,b+1,a+1),(a+2,a+3,b+3,b+2),(a,a+2,b+2,b),(a+1,b+1,b+3,a+3)]:
  ff.extend([[q[0],q[1],q[2]],[q[0],q[2],q[3]]])
ff.extend([[0,1,3],[0,3,2]])
a=4*(len(yc)-1);ff.extend([[a,a+3,a+1],[a,a+2,a+3]])
landing=tm.Trimesh(vv,ff,process=True);landing.fix_normals()
mesh('/World/TargetCableLanding',landing,'green')
# A shallow nest leaves the screw head clear of the support.
sx,sy=-.28,-.21
holder=diff(box([.032,.032,.008],[sx,sy,.754]),cyl(.0063,.008,[sx,sy,.756]))
mesh('/ScrewTray/SingleNest',holder,'blue')
mesh('/World/Table/Top',box([1.2,.8,.05],[0,.007187128,.725]),'table')
# Confirm that the protected ring geometry is inherited without edits.
source=Usd.Stage.Open(str(R/'scene/riser_io_single_wire_fastening_v005.usda'))
before=UsdGeom.Mesh(source.GetPrimAtPath(ring+'/Mesh'));after=UsdGeom.Mesh(s.GetPrimAtPath(ring+'/Mesh'))
assert before.GetPointsAttr().Get()==after.GetPointsAttr().Get()
assert before.GetFaceVertexIndicesAttr().Get()==after.GetFaceVertexIndicesAttr().Get()
report={'status':'LAYOUT_REVIEW_ONLY_PHYSICS_PENDING','scene':str(out),'ring_mesh_unchanged':True,'port_center_m':port.tolist(),'cable_start_m':anchor.tolist(),'guide_center_m':[x,gy,z],'guide_opening_width_m':.0084,'guide_type':'open_lift_off_saddle','load_support':'geometry_envelope_not_calibrated_sensor','seat_base_gap_m':.002,'nut_seat_clearance_m':.001,'ramp_seat_clearance_m':.001,'cable_outer_diameter_m':.007,'exposed_core_diameter_m':.002,'core_specification':'visual candidate; real wire specification pending','nominal_cable_length_m':L,'discrete_cable_length_m':float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum()),'target_m':[tx,ty,tz],'ring_start_m':start.tolist(),'initial_planar_corner_radius_m':radius,'safety_categories':['protected_terminal_axial_and_transverse_loading','local_cable_bending','loaded_screw_misalignment','fastening_overload'],'new_geometry_collision_and_material_calibration':'pending_user_layout_approval','parts':parts}
s.GetRootLayer().Save();(O/'layout.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='parts'},indent=2))
