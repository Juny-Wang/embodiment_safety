from pathlib import Path
import sys,gc,contextlib
R=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io')
cases=[('reference','reference_003','fixture_contact_impl_v012.py'),('seat_press','seat_press_002','fixture_contact_impl_v012.py')]

sys.argv=[sys.argv[0]]
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'limit_cpu_threads':4,'width':960,'height':720,'sync_loads':True})
import omni.usd
from isaacsim.core.api import SimulationContext
for case,name,script in cases:
 source=(R/'scripts'/script).read_text()
 line=next(x for x in source.splitlines() if "app=SimulationApp(" in x)
 source=source.replace(line," app=shared_app").replace(" if app is not None:app.close()","")
 out=R/'scene/unsafe_suite_v011'/name
 if (out/'result.json').exists():continue
 print('BATCH_BEGIN',case,flush=True)
 sys.argv=['run_unsafe_suite_v011.py','--name',name,'--case',case,'--record','--resume']
 ns={'__name__':'__main__','shared_app':app}
 with (R/'scene/unsafe_suite_v011'/(name+'.execution.log')).open('w') as log,contextlib.redirect_stdout(log):
  exec(compile(source,'run_unsafe_suite_v011.py','exec'),ns)
 if 'sim' in ns:ns['sim'].stop()
 omni.usd.get_context().close_stage()
 SimulationContext.clear_instance()
 del ns
 gc.collect()
 for _ in range(10):app.update()
 print('BATCH_END',case,flush=True)
app.close()
