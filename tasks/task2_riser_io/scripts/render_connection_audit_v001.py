from pathlib import Path
from isaacsim import SimulationApp
R=Path("/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io")
O=R/"scene/connection_audit_v001";O.mkdir(exist_ok=True)
app=SimulationApp({"headless":True,"width":1200,"height":900,"sync_loads":True,"extra_args":["--/log/file="+str(O/"kit.log")]})
import omni.usd,omni.replicator.core as rep
from pxr import UsdGeom,UsdLux,Gf
from PIL import Image
omni.usd.get_context().open_stage(str(R/"scene/riser_io_single_wire_fastening_v005.usda"))
for _ in range(12):app.update()
stage=omni.usd.get_context().get_stage();stage.SetEditTarget(stage.GetSessionLayer())
UsdLux.DomeLight.Define(stage,"/World/AuditLight").CreateIntensityAttr(1800.)
cam=UsdGeom.Camera.Define(stage,"/World/AuditCamera");cam.CreateFocalLengthAttr(40.);cam.CreateClippingRangeAttr(Gf.Vec2f(.001,10.))
op=UsdGeom.Xformable(cam).AddTransformOp()
prod=rep.create.render_product(str(cam.GetPath()),(1200,900))
ann=rep.AnnotatorRegistry.get_annotator("rgb");ann.attach([prod])
for name,eye,target in [
 ("fixed_port_front",(-.045,.13,.85),(-.025,.325,.779)),
 ("fixed_port_side",(-.19,.23,.89),(-.02,.325,.779)),
 ("layout_top",(-.15,.06,1.45),(-.15,.065,.76))]:
 op.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,1,0) if name=="layout_top" else Gf.Vec3d(0,0,1)).GetInverse())
 rep.orchestrator.step(rt_subframes=8, pause_timeline=True)
 a=ann.get_data()
 assert a is not None and a.size
 Image.fromarray(a).save(O/(name+".png"))
 print("SAVED",name,flush=True)
app.close()
