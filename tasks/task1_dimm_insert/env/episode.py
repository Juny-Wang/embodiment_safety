"""Shared causal episode/reward state. reset creates fresh history; no simulator writes."""
from .termination import EpisodeEnd
from .milestones import Evaluator
class Episode:
 def __init__(self,socket_quat,timeout_s=240.):
  self.socket_quat=socket_quat;self.timeout_s=timeout_s;self.end=None
 def reset(self,initial):
  self.end=EpisodeEnd(initial,self.timeout_s);self.evaluator=Evaluator(self.socket_quat)
  self.previous_time=initial["sim_time"]
  return self.evaluator.reward.progress()
 def step(self,o,e,peaks):
  if self.end is None:raise RuntimeError("reset required")
  end=self.end.update(o,e)
  if end["failure"]:
   evidence=dict(valid=e.get("valid",False),sim_time=o["sim_time"],source="episode_failure")
   label=self.evaluator.reward.update(o["sim_time"],evidence,terminated=True,failure=end["failure"])
   label["evidence"]=evidence
  else:
   label=self.evaluator.update(o,self.previous_time,e,peaks,
     terminal_acceptance=end if end["success"] else None,truncated=end["truncated"])
  self.previous_time=o["sim_time"]
  return dict(label,end=end,valid_for_training=end["valid_for_training"],
    lifecycle_version=end["version"])
