"""100 ms recorder boundary, independent of Isaac imports."""
import copy
class Boundary:
 def __init__(self,panel,decide,emit_partial):
  self.p=panel;self.decide=decide;self.emit_partial=emit_partial;self.steps=0
  self.total=0;self.decisions=[];self.config=None
 def decide_next(self):
  p=self.p
  before=float(p.controller.width_goal)
  p.action=self.decide()
  self.current_action=p.action.copy()
  self.config=copy.deepcopy(dict(control=p.controller.cfg["control"],action=p.cfg["action"],
    width_goal_before=before,width_goal_after=float(p.controller.width_goal)))
  self.decisions.append(dict(time=p.t,action=p.action.tolist(),configuration=self.config))
 def step(self):
  p=self.p
  if p.pending_obs is None:
   p.pending_obs=p.observe(require_images=True)
   p.clock=0.;self.steps=0;p.subcommands=[];p.interval_peaks={}
   self.decide_next()
   return
  self.steps+=1;p.clock=self.steps/240.
  if not p.enabled:
   p.finish("interrupted","disabled_within_action_interval")
   return
  if self.steps==24:
   nxt=p.observe(require_images=True)
   if abs(nxt["sim_time"]-p.pending_obs["sim_time"]-.1)>1e-6:
    raise RuntimeError("Non-100ms interval")
   if len(p.subcommands)!=24:raise RuntimeError("Expected exactly 24 physics commands")
   p.active.append(p.pending_obs,p.action,nxt,dict(contact_peaks=p.interval_peaks,
     controller_commands=p.subcommands,controller_error=p.controller.error,
     execution_configuration=self.config,physics_steps=24,operator="aligned_10hz_expert"))
   p.pending_obs=nxt;p.clock=0.;self.steps=0;p.subcommands=[];p.interval_peaks={}
   self.total+=1
   self.decide_next()
  elif self.steps>24:raise RuntimeError("Missed control boundary")
 def flush(self,outcome,reason):
  p=self.p
  if p.active and p.pending_obs is not None:
   elapsed=p.t-p.pending_obs["sim_time"]
   if elapsed>1e-9:
    self.emit_partial(dict(observations=p.pending_obs,actions=self.current_action.tolist(),
     next_observations=p.observe(),dt_s=elapsed,controller_commands=p.subcommands,
     outcome=outcome,reason=reason,valid_for_fixed_100ms_replay=False))
