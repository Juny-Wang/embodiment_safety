"""Causal operational milestones for the user-accepted fixed scene."""
import numpy as np
from scipy.spatial.transform import Rotation
from .reward import StageReward,STAGES
class Evaluator:
 def __init__(self,socket_quat):
  self.socket=Rotation.from_quat(socket_quat);self.reward=StageReward();self.opening_since=None
 def update(self,o,start,measured,peaks,*,terminal_acceptance=None,truncated=False):
  t=o["sim_time"];m=o["task_truth"]["metrics"]
  open_now=m["left_angle_deg"]<=-15 and m["right_angle_deg"]>=15
  clear=max(peaks["left_press_N"],peaks["right_press_N"])<.05
  if open_now and clear:
   if self.opening_since is None:self.opening_since=start
  else:self.opening_since=None
  e={k:False for _,_,ks in STAGES for k in ks}
  e.update(valid=measured["valid"],sim_time=t,source="accepted_scene_milestones_v001",
    sensors_valid=measured["sensors_valid"],no_safety_violation=measured["no_safety_violation"],
    no_unintended_penetration=measured["recorded_dimm_contacts_under_penetration_limit"],
    insertion_path_clear=bool(open_now),hands_clear=bool(clear),
    opening_survives_withdrawal=bool(self.opening_since is not None and t-self.opening_since>=.3))
  for k in ("grasp_in_allowed_band","bilateral_grasp_stable","dimm_tracks_gripper","rack_clear"):e[k]=measured[k]
  angle=float(np.degrees((self.socket.inv()*Rotation.from_quat(o["task_truth"]["poses"]["dimm"][3:])).magnitude()))
  pos=m["dimm_position_in_socket_m"]
  e.update(key_orientation_correct=bool(angle<=2),orientation_ok=bool(angle<=2),
    lateral_alignment_ok=bool(np.linalg.norm(pos[:2])<=.0025 and .0068<=pos[2]<=.0368))
  if terminal_acceptance is not None:
   if terminal_acceptance.get("accepted") is not True:raise ValueError("Not accepted")
   for k in ("fixed_base_support","left_hook_engaged","right_hook_engaged","contact_driven_closure",
     "released","retracted","retention_stable","independent_acceptance_passed"):e[k]=True
  result=self.reward.update(t,e,terminated=terminal_acceptance is not None,truncated=truncated)
  return dict(result,evidence=e)
