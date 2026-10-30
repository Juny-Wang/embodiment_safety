from pathlib import Path
import os,sys,json,traceback,runpy
P=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io/scene/wire_safety_v001')
for key,n in [('XDG_CACHE_HOME','cache'),('XDG_DATA_HOME','data'),('XDG_CONFIG_HOME','config'),('TMPDIR','tmp')]:os.environ[key]=str(P/'runtime'/n)
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'width':800,'height':600,'extra_args':['--portable','--portable-root',str(P/'runtime'),'--/log/file='+str(P/'runs/monitor_kit.log')]})
report={'headless_monitor_smoke_pass':False,'manual_gui_verified':False}
try:
 import omni.usd
 from isaacsim.core.api import SimulationContext
 omni.usd.get_context().open_stage(str(P/'L0_sensor_bench.usda'))
 for _ in range(5):app.update()
 sim=SimulationContext(physics_dt=1/960,rendering_dt=1/60,backend='numpy',device='cpu')
 sim.initialize_physics();sim.play()
 for _ in range(10):sim.step(render=True)
 runpy.run_path(str(P/'scripts/launch_monitor.py'))
 monitor=sys.modules['_task2_wire_safety_monitor_v001'].monitor
 for _ in range(100):sim.step(render=True)
 report['labels']={k:v.text for k,v in monitor.labels.items()}
 report['sample_log']=str(monitor.out/'metrics.jsonl')
 report['headless_monitor_smoke_pass']=bool(monitor.control_pass and monitor.labels['status'].text.startswith('GREEN'))
 report['arrows_api_available']=monitor.dd is not None
 monitor.close();sim.stop()
except Exception:report['error']=traceback.format_exc()
finally:
 (P/'reports/monitor_smoke.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report),flush=True);app.close()
