"""Replay terminal semantics. No Isaac imports; duration is measured, never padded."""
import math
import numpy as np

def labels(dt_s,physics_steps,*,terminated=False,truncated=False,success=False,reason=""):
 if terminated and truncated:raise ValueError("Termination and truncation are exclusive")
 if physics_steps<1 or physics_steps>24:raise ValueError("Expected 1..24 executed physics steps")
 if not math.isfinite(dt_s) or abs(dt_s-physics_steps/240)>1e-6:raise ValueError("Duration/physics mismatch")
 full=physics_steps==24
 if not full and not terminated:raise ValueError("Short transition must be a true terminal, never a truncation")
 if success and (not terminated or not full):raise ValueError("Success requires full independently accepted terminal boundary")
 if terminated and not success and not reason:raise ValueError("Failure reason required")
 return dict(reward=float(success),terminated=bool(terminated),truncated=bool(truncated),
  done=bool(terminated or truncated),mask=0. if terminated else 1.,
  dt_s=float(dt_s),physics_steps=int(physics_steps),partial_terminal=not full,
  nominal_control_dt_s=.1,reason=reason,
  discount_rule="gamma*mask; true terminal mask=0 so no duration-dependent bootstrap")

def validate_action(a):
 a=np.asarray(a,dtype=np.float64)
 if a.shape!=(7,) or not np.isfinite(a).all() or np.max(np.abs(a))>1.00001:
  raise ValueError("Invalid action")
 return a

def completed_partial(tail):
 """Keep only issued commands with a fully observed physics interval."""
 start=float(tail["observations"]["sim_time"]);end=float(tail["next_observations"]["sim_time"])
 dt=end-start;n=round(dt*240)
 if n<1 or n>24 or abs(dt-n/240)>1e-6:raise ValueError("Partial duration")
 commands=tail["controller_commands"]
 done=[c for c in commands if float(c["time"])<end-1e-7]
 pending=[c for c in commands if float(c["time"])>=end-1e-7]
 if len(done)!=n or len(pending)>1:raise ValueError("Command/observed step mismatch")
 for i,c in enumerate(done):
  if abs(float(c["time"])-(start+i/240))>1e-6:raise ValueError("Noncontiguous command time")
 if pending and abs(float(pending[0]["time"])-end)>1e-6:raise ValueError("Unexpected pending time")
 return dict(tail,controller_commands=done,physics_steps=n,issued_command_count=len(commands),
  unobserved_commands=pending,dt_s=dt)
