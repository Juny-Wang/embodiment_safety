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
