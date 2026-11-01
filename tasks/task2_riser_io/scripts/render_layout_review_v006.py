from pathlib import Path
from isaacsim import SimulationApp
R=Path("/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io")
O=R/"scene/layout_review_v006"
app=SimulationApp({"headless":True,"width":1440,"height":1080,"sync_loads":True,"extra_args":["--/log/file="+str(O/"render.log")]})
import omni.usd,omni.replicator.core as rep
from pxr import UsdGeom,UsdLux,UsdPhysics,Gf
from PIL import Image
import json,numpy as np
try:
 omni.usd.get_context().open_stage(str(R/"scene/riser_io_layout_review_v006.usda"))
 for _ in range(12):app.update()
 stage=omni.usd.get_context().get_stage();stage.SetEditTarget(stage.GetSessionLayer())
 # Freeze geometry for review; session overrides are not saved to the scene.
 for prim in stage.Traverse():
  if prim.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
  if prim.IsA(UsdPhysics.Joint):UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
 UsdLux.DomeLight.Define(stage,"/World/ReviewLight").CreateIntensityAttr(1500.)
 cam=UsdGeom.Camera.Define(stage,"/World/ReviewCamera");cam.CreateFocalLengthAttr(36.);cam.CreateClippingRangeAttr(Gf.Vec2f(.001,10.))
 op=UsdGeom.Xformable(cam).AddTransformOp()
 prod=rep.create.render_product(str(cam.GetPath()),(1440,1080));ann=rep.AnnotatorRegistry.get_annotator("rgb");ann.attach([prod])
 configs={}
 for name,eye,target in [
  ("overview",(-.10,-.54,1.62),(-.22,.045,.76)),
  ("connection",(-.145,.16,.875),(-.045,.287,.765)),
  ("target",(-.005,-.21,.92),(-.092,-.045,.767))]:
  mat=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse();op.Set(mat)
  cam.GetFocalLengthAttr().Set(32. if name=="overview" else 36.)
  rep.orchestrator.step(rt_subframes=16,delta_time=0.0,pause_timeline=True)
  a=ann.get_data();assert a is not None and a.size
  Image.fromarray(a).save(O/(name+".png"))
  configs[name]={"matrix":np.array(mat).tolist(),"focal_length":float(cam.GetFocalLengthAttr().Get()),"horizontal_aperture":float(cam.GetHorizontalApertureAttr().Get()),"vertical_aperture":float(cam.GetVerticalApertureAttr().Get()),"resolution":[1440,1080]}
  print("SAVED",name,flush=True)
 (O/"cameras.json").write_text(json.dumps(configs,indent=2))
finally:
 app.close()
