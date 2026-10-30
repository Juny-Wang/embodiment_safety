"""Short complete-candidate sensor/UI validation. No robot task commands."""
from pathlib import Path
import os,sys,json,hashlib,signal
ROOT=Path("/mnt/data2/wjy_SERL/franka_sim");OUT=ROOT/"outputs/T1_rl/franka_manual_safety_v002"
for k,n in [("XDG_CACHE_HOME","cache"),("XDG_CONFIG_HOME","config"),("XDG_DATA_HOME","data"),("TMPDIR","tmp")]:os.environ[k]=str(OUT/"runtime"/n)
sys.argv=sys.argv[:1]+["--portable-root",str(OUT/"runtime")]
from isaacsim import SimulationApp
app=SimulationApp({"headless":True,"create_new_stage":False,"enable_crashreporter":False,"limit_cpu_threads":8,
 "extra_args":["--/app/settings/persistent=false","--/app/settings/loadUserConfig=false",f"--/log/file={OUT}/keyboard_native.log"]})
from pxr import Usd,UsdUtils,UsdGeom
import omni.usd,omni.physx
sys.path.insert(0,str(OUT))
import keyboard_panel as launch_panel
main=ROOT/"outputs/T1_rl/t1_dimm_insert_train.usda"
asset=ROOT/"outputs/T1_rl/socket_franka_install_v008/socket_print_v007_installed.usda"
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [main,asset]}
omni.usd.get_context().open_stage(str(OUT/"manual_franka_scene.usda"))
st=omni.usd.get_context().get_stage()
panel=launch_panel.KeyboardPanel(show_ui=True)
panel.physics_sub=None;panel.update_sub=None
from keyboard_input import vectors
import math,types,numpy as np,carb.input
from scipy.spatial.transform import Rotation
assert vectors({'UP','DOWN'})==[[0.,0.,0.],[0.,0.,0.]]
assert abs(np.linalg.norm(vectors({'UP','LEFT','PAGE_UP'})[0])-1)<1e-12
assert vectors({'UP','LEFT_SHIFT'})[1]==[1.,0.,0.]
saved=(panel.controller,panel.appwindow,panel.timeline)
class FakeController:
 ready=True
 commands=0
 armed=False
 cfg=launch_panel.monitor.CONFIG
 goal=(np.zeros(3),Rotation.identity())
 width_goal=.04
 def arm(self):self.armed=True
 def stop(self,*a):self.armed=False
fake=FakeController();panel.controller=fake
panel.appwindow=types.SimpleNamespace(is_focused=lambda:True)
panel.timeline=types.SimpleNamespace(is_playing=lambda:True)
panel.keyboard_enabled=True
event=lambda key,kind:types.SimpleNamespace(input=getattr(carb.input.KeyboardInput,key),type=kind)
press=carb.input.KeyboardEventType.KEY_PRESS
release=carb.input.KeyboardEventType.KEY_RELEASE
panel.on_key(event('UP',press));panel.keyboard_tick(.01)
assert abs(fake.goal[0][0]-.0001)<1e-12 and fake.armed
panel.on_key(event('UP',release));panel.keyboard_tick(.01)
assert not fake.armed and not panel.held
panel.on_key(event('LEFT_SHIFT',press));panel.on_key(event('UP',press));panel.keyboard_tick(.01)
assert np.linalg.norm(fake.goal[1].as_rotvec())>0
panel.appwindow.is_focused=lambda:False;panel.keyboard_tick(.01)
assert not panel.keyboard_enabled and not panel.held and not fake.armed
panel.controller,panel.appwindow,panel.timeline=saved
panel.safety={"rows":[],"stop":False}
print('KEYBOARD_MAPPING_RELEASE_FOCUS_TESTS_PASS',flush=True)

sim=omni.physx.get_physx_simulation_interface();omni.physx.get_physx_interface().set_thread_count(0)
sid=UsdUtils.StageCache.Get().Insert(st).ToLongInt();panel.simulation_stage_id=sid
sim.attach_stage(sid)
for i in range(200):
 sim.simulate(.001,i*.001);sim.fetch_results();panel.physics_step(.001)
panel.update(None)
result={"keyboard_mapping_release_focus_tests":True,"hardware_key_delivery_tested":False,"physics_steps":panel.stats["physics_steps"],"measurement_errors":panel.errors,
 "metrics":panel.metrics,"colors":{r["key"]:r["color"] for r in panel.safety["rows"]},
 "latched_reasons":panel.safety["latched_reasons"],"contact_count":panel.stats["contact_count"],
 "colored_labels":len(panel.metric_labels),"debug_draw_available":panel.draw_iface is not None,
 "robot_auto_armed":panel.controller.armed,"robot_commands":panel.controller.commands,
 "source_files_unchanged":all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items()),
 "log_directory":str(panel.logdir),"full_mechanical_task_accepted":False}
result["feedback_pipeline_ready"]=not result["measurement_errors"] and result["physics_steps"]==200 and len(panel.safety["rows"])>0
panel.close();sim.detach_stage()
(OUT/"keyboard_acceptance.json").write_text(json.dumps(result,indent=2))
print("FEEDBACK_RESULT",json.dumps(result),flush=True)
signal.signal(signal.SIGALRM,lambda *a:os._exit(0));signal.alarm(5);app.close()
