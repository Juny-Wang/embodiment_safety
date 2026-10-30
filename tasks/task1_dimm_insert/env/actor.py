"""Native-clock policy rollout; source scene read-only, fresh process per reset."""
import os,sys,json,time,traceback,subprocess,hashlib,inspect,textwrap,asyncio
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(sys.argv[1]).resolve()
assert OUT.is_relative_to(ROOT/"outputs")
OUT.mkdir(parents=True,exist_ok=False);os.chdir(ROOT);sys.path.insert(0,str(ROOT))
from runners.common import config,resolve
CFG=config(os.environ.get("TASK1_CONFIG","configs/experiments/task1_serl.json"))
MAX_STEPS=int(os.environ.get("TASK1_MAX_STEPS",round(CFG["timeout_s"]*10)))
SMOKE=os.environ.get("TASK1_SMOKE","0")=="1"
terminal_label=None

INJECT=int(sys.argv[sys.argv.index("--inject-stop-after")+1]) if "--inject-stop-after" in sys.argv else 0
assert INJECT in (0,7)
def write(n,x):
 p=OUT/n;q=p.with_suffix(p.suffix+".tmp");q.write_text(json.dumps(x,indent=2));q.replace(p)
app=None;p=None;child=None;log=None
try:
 from isaacsim import SimulationApp
 app=SimulationApp({"headless":True,"renderer":"RayTracedLighting","width":640,"height":480,"sync_loads":True})
 import numpy as np,omni.usd,omni.kit.app
 from pxr import UsdLux,UsdGeom
 from tasks.task1_dimm_insert.env.episode import Episode as EpisodeCoordinator
 from tasks.task1_dimm_insert.env.evidence import Evidence
 from tasks.task1_dimm_insert.env.policy import attach_progress
 from tasks.task1_dimm_insert.teleop import runtime
 from tasks.task1_dimm_insert.env.data import Episode,clean
 from tasks.task1_dimm_insert.env import observations as vision
 from tasks.task1_dimm_insert.env.boundary import Boundary
 from tasks.task1_dimm_insert.env.execution import install
 from tasks.task1_dimm_insert.env.trace import traced_apply_velocity
 from tasks.task1_dimm_insert.env.transition import labels,validate_action,completed_partial
 prerequisite=resolve(CFG["dataset"])/"manifest.json"
 assert json.loads(prerequisite.read_text())["transitions"]>0
 scene=ROOT/"tasks/task1_dimm_insert/scene/collection.usda"
 protected=[scene,Path(runtime.__file__),ROOT/"tasks/task1_dimm_insert/env/actions.py"]
 protected_before={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in protected}
 ctx=omni.usd.get_context();assert ctx.open_stage(str(scene))
 for _ in range(40):app.update()
 stage=ctx.get_stage();stage.SetEditTarget(stage.GetSessionLayer())
 if not any(x.HasAPI(UsdLux.LightAPI) for x in stage.Traverse()):
  UsdLux.DomeLight.Define(stage,"/World/DiagnosticDome").CreateIntensityAttr(1000.)
 callback=textwrap.dedent(inspect.getsource(runtime.CollectionPanel.physics_step))
 start=callback.index("        if self.active:\n            if self.pending_obs is None:")
 end=callback.index("        if not self.enabled:return",start)
 callback=callback[:start]+"""        if self.active:
            self.native_boundary.step()
        else:self.action=np.zeros(7)
"""+callback[end:]
 scope={};exec(compile(callback,"<native actor callback>","exec"),runtime.__dict__,scope)
 runtime.CollectionPanel.physics_step=scope["physics_step"]
 p=runtime.CollectionPanel(show_ui=False);p.cfg["control_hz"]=10
 p.cfg["teleop_assist"]["enabled"]=False
 p.cfg["max_image_age_s"]=.1
 p.cfg["demonstration_limits"]=dict(left_press_N=3.,right_press_N=3.,dimm_environment_N=5.,finger_left_N=5.,finger_right_N=5.)
 runtime.Episode=lambda ignored,meta:Episode(OUT/"episodes",meta)
 stopped=False;failure=None;executed_steps=0;calls=0;initial=None;final=None
 frozen=None;check=None
 live_frames=[]; evidence_monitor=None; evaluator=None; last_obs=None
 def capture(n,raw):
  global evidence_monitor,evaluator,last_obs,terminal_label
  if evaluator is None:
   q=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath("/World/MatchedSocket/Asset")).ExtractRotationQuat().GetNormalized()
   evaluator=EpisodeCoordinator([*q.GetImaginary(),q.GetReal()],timeout_s=MAX_STEPS*.1);evaluator.reset(raw);evidence_monitor=Evidence(raw)
  elif n>0:
   frames=list(live_frames)
   measured=evidence_monitor.update(raw,last_obs["sim_time"],frames)
   if not measured["valid"]:raise RuntimeError("Invalid live reward measurements")
   peaks={k:max(f[1]["metrics"][k] for f in frames) for k in ("left_press_N","right_press_N")}
   if failure:
    label=evaluator.evaluator.reward.update(raw["sim_time"],dict(valid=measured["valid"],sim_time=raw["sim_time"],source="measured_short_terminal"),terminated=True,failure=failure)
    label.update(dt_s=raw["sim_time"]-last_obs["sim_time"],physics_steps=len(frames),injected=bool(INJECT))
   else:
    label=evaluator.step(raw,measured,peaks)

   terminal_label=label
   write(f"stage_transition_{n-1:03d}.json",dict(index=n-1,**label))
  live_frames.clear();last_obs=clean(raw)
  ob=vision.load_observation(raw,p.active.path if p.active else Path(p.last_saved_path),max_age_s=.1)
  ob=attach_progress(ob,evaluator.evaluator.reward.progress())
  temp=OUT/f"obs_{n:03d}.tmp"
  with temp.open("wb") as f:np.savez_compressed(f,**ob)
  temp.replace(OUT/f"obs_{n:03d}.npz")
  write(f"raw_{n:03d}.json",clean(raw))
  return ob
 def partial(row):
  write("partial_transition.json",clean(row))
 def decide():
  global calls,initial,final,stopped
  check()
  n=p.native_boundary.total;raw=p.pending_obs
  capture(n,raw)
  if initial is None:initial=clean(raw)
  if terminal_label and (terminal_label['terminated'] or terminal_label['truncated']):
   final=clean(raw);stopped=True
   write("time_limit_event.json",dict(time=p.t,full_transitions=n))
   p.finish("success_request" if terminal_label["success"] else "interrupted",terminal_label.get("failure") or "episode_end")
   p.disable()
   write("policy_stop.json",dict(final_observation_index=n,reason="episode_end"))
   return np.zeros(7)
  write(f"request_{n:03d}.json",dict(index=n,sim_time=p.t))
  deadline=time.monotonic()+360
  path=OUT/f"action_{n:03d}.json"
  while not path.exists():
   if child.poll() is not None or (OUT/"worker_error.json").exists():raise RuntimeError("Inference worker failed")
   if time.monotonic()>deadline:raise TimeoutError("Policy action")
   time.sleep(.05)
  reply=json.loads(path.read_text());a=validate_action(reply["executed"])
  if reply["index"]!=n:raise ValueError("Action index mismatch")
  if SMOKE and (np.linalg.norm(a[:3])>.002/.32+1e-9 or np.any(a[3:]!=0)):raise ValueError("Smoke action bound")
  calls+=1
  return a
 p.native_boundary=Boundary(p,decide,partial)
 finish=p.finish
 def finish_record(outcome,reason=""):
  global failure,stopped
  if reason.startswith("controller_disarmed:"):
   failure=reason;stopped=True
   write("controller_diagnostics.json",clean(dict(time=p.t,error=p.controller.error,
     ik=p.controller.ik_diagnostics,control=p.controller.cfg["control"])))
  if p.active:
   p.native_boundary.flush(outcome,reason)
   write("aligned_decisions.json",clean(p.native_boundary.decisions))
  return finish(outcome,reason)
 p.finish=finish_record
 measured=p.measure_contacts
 def measure(dt):
  global failure,stopped
  contacts=measured(dt)
  if p.active:
   live_frames.append((executed_steps,dict(sim_time=p.t,contacts=contacts,metrics=p.contact_metrics,controller_error=p.controller.error)))
  for k,limit in p.cfg["demonstration_limits"].items():
   if float(p.contact_metrics.get(k,0) or 0)>limit:failure="force_limit:"+k
  if INJECT and executed_steps>=INJECT and not failure:failure="diagnostic_stop_injected"
  if failure and p.enabled:
   p.enabled=False;p.controller.stop(failure);stopped=True
  return contacts
 p.measure_contacts=measure
 apply=traced_apply_velocity(runtime.apply_velocity)
 def counted(c,a,dt,cfg):
  global executed_steps
  x=apply(c,a,dt,cfg);executed_steps+=1;return x
 runtime.apply_velocity=counted
 async def launch():
  global frozen,check,child,log
  p.timeline.play()
  for _ in range(180):await omni.kit.app.get_app().next_update_async()
  p.initialize()
  for _ in range(60):await omni.kit.app.get_app().next_update_async()
  limits=p.controller.view.get_dof_max_forces().copy();limits[0,p.controller.fingers]=5.
  p.controller.view.set_dof_max_forces(limits,np.array([0],dtype=np.uint32))
  frozen,check=install(p,lambda row:write("unexpected_startup_release.json",row))
  write("execution_configuration.json",clean(frozen))
  poses={k:v.get_transforms()[0].copy() for k,v in p.bodies.items()}
  metrics=p.task_metrics(poses)
  if max(abs(metrics[k+"_angle_deg"]) for k in ("left","right"))>1:raise RuntimeError("Reset latches open")
  if np.linalg.norm(metrics["dimm_position_in_socket_m"][:2])<.05:raise RuntimeError("Not free-space smoke")
  write("reset.json",clean(dict(metrics=metrics,poses=poses,ready=True)))
  env={k:v for k,v in os.environ.items() if k not in ("PYTHONHOME","PYTHONPATH","LD_LIBRARY_PATH","LD_PRELOAD","PYTHONSTARTUP")}
  env["PYTHONDONTWRITEBYTECODE"]="1"
  env["SERL_REPLAY_DIR"]=str(OUT/"diagnostic_replay" if SMOKE or os.environ.get("TASK1_EVAL")=="1" else resolve(CFG["replay"]))
  env["TASK1_MAX_STEPS"]=str(MAX_STEPS)
  env["TASK1_SMOKE"]="1" if SMOKE else "0"
  log=(OUT/"worker.log").open("w")
  child=subprocess.Popen(["/home/ufdwaa/anaconda3/envs/env_isaaclab/bin/python","-B",
    str(ROOT/"algos/serl/actor_worker.py"),str(OUT)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  p.start();p.enable()
 launch_task=asyncio.ensure_future(launch());wall=time.monotonic();last=wall
 while app.is_running():
  app.update()
  if launch_task.done() and launch_task.exception():raise launch_task.exception()
  if time.monotonic()-last>15:
   write("heartbeat.json",dict(wall_s=time.monotonic()-wall,sim_time=p.t,commands=executed_steps,policy_actions=calls,failure=failure))
   last=time.monotonic()
  if p.sensor_errors:raise RuntimeError(str(p.sensor_errors))
  if stopped and not p.active:break
  if p.controller.error and calls>0:raise RuntimeError("controller_fault:"+str(p.controller.error))
  if time.monotonic()-wall>86400:raise TimeoutError("Native actor diagnostic")
 p.timeline.pause()
 episode=Path(p.last_saved_path)
 if failure:
  tail=completed_partial(json.loads((OUT/"partial_transition.json").read_text()))
  final=tail["next_observations"];capture(p.native_boundary.total+1,final)
  write("policy_stop.json",dict(final_observation_index=p.native_boundary.total+1,reason=failure))
 child.wait(timeout=90)
 if child.returncode:raise RuntimeError("Worker failed")
 rows=[json.loads(s) for s in (episode/"transitions.jsonl").read_text().splitlines()]
 replay=[]
 for i,row in enumerate(rows):
  label=json.loads((OUT/f"stage_transition_{i:03d}.json").read_text())
  replay.append(dict(label,index=i))
 if failure:
  tail=completed_partial(json.loads((OUT/"partial_transition.json").read_text()))
  label=labels(tail["dt_s"],len(tail["controller_commands"]),terminated=True,reason=failure)
  write("terminal_transition.json",dict(tail,**label,diagnostic_only=bool(INJECT),training_eligible=not bool(INJECT)))
 write("replay_labels.json",replay)
 unchanged={str(x):hashlib.sha256(x.read_bytes()).hexdigest()==protected_before[str(x)] for x in protected}
 write("source_integrity.json",dict(all_unchanged=all(unchanged.values()),files=unchanged,hashes=protected_before))
 write("result.json",dict(execution_completed=True,full_transitions=len(rows),physics_commands=executed_steps,
  observed_physics_steps=len(rows)*24+(tail["physics_steps"] if failure else 0),policy_actions=calls,network_updates=json.loads((OUT/"worker_result.json").read_text())["network_updates"],injected_stop=bool(INJECT),failure_reason=failure,
  ee_displacement_m=float(np.linalg.norm(np.array(final["ee_position_m"])-initial["ee_position_m"])),
  episode=str(episode),source_unchanged=all(unchanged.values()),acceptance_pending=True,
  full_task_success=bool(terminal_label and terminal_label["success"]),training_ready=False,native_app_update=True))
except Exception:
 write("actor_error.json",dict(error=traceback.format_exc()));traceback.print_exc()
finally:
 if p:
  try:
   if p.active:p.finish("interrupted","diagnostic_shutdown")
   p.disable();p.close()
  except Exception:traceback.print_exc()
 if child and child.poll() is None:
  child.terminate()
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:child.kill()
 if log:log.close()
 if app:app.close()
