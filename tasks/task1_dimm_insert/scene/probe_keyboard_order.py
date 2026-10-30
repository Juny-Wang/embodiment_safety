"""Isolated actual articulated Franka drive test. No runtime joint-state teleport."""
from pathlib import Path
import os,sys,json,csv,hashlib,signal,math
ROOT=Path("/mnt/data2/wjy_SERL/franka_sim");OUT=ROOT/"outputs/T1_rl/franka_manual_safety_v002"
for n in ["runtime/cache","runtime/config","runtime/data","runtime/tmp"]:(OUT/n).mkdir(parents=True,exist_ok=True)
for k,n in [("XDG_CACHE_HOME","cache"),("XDG_CONFIG_HOME","config"),("XDG_DATA_HOME","data"),("TMPDIR","tmp")]:os.environ[k]=str(OUT/"runtime"/n)
sys.argv=sys.argv[:1]+["--portable-root",str(OUT/"runtime")]
from isaacsim import SimulationApp
app=SimulationApp({"headless":True,"create_new_stage":False,"enable_crashreporter":False,"limit_cpu_threads":8,
 "extra_args":["--/app/settings/persistent=false","--/app/settings/loadUserConfig=false",f"--/log/file={OUT}/keyboard_motion_native.log"]})
import carb.input
inp=carb.input.acquire_input_interface();provider=inp.get_input_provider()
key=provider.create_keyboard("Franka capture ordering probe")
results=[]
for order in [-100000,100000]:
 for consume in [False,True]:
  trace=[]
  def first(e):trace.append("capture");return consume
  def second(e):trace.append("downstream");return True
  b=inp.subscribe_to_input_events(second,order=0)
  a=inp.subscribe_to_input_events(first,order=order)
  app.update()
  provider.buffer_keyboard_key_event(key,carb.input.KeyboardEventType.KEY_PRESS,carb.input.KeyboardInput.UP,0)
  inp.distribute_buffered_events()
  results.append({"order":order,"return":consume,"trace":trace})
  inp.unsubscribe_to_input_events(a);inp.unsubscribe_to_input_events(b)
print("INPUT_ORDER_PROBE",json.dumps(results),flush=True)
(OUT/"keyboard_order_probe.json").write_text(json.dumps(results,indent=2))
provider.destroy_keyboard(key)
signal.signal(signal.SIGALRM,lambda *a:os._exit(0));signal.alarm(5);app.close()
