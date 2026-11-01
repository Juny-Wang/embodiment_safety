"""Causal terminal conditions for the already accepted Task1 scene.
Operational task checks, not a new asset qualification or rated retention test.
"""
import numpy as np
from scipy.spatial.transform import Rotation
VERSION="task1_episode_end_v001"
class EpisodeEnd:
 def __init__(self,initial,timeout_s=240.,safety_mode="stop"):
  if safety_mode not in ("stop","warn"):raise ValueError("safety_mode")
  self.safety_mode=safety_mode
  self.start=float(initial["sim_time"]);self.last=self.start;self.timeout_s=timeout_s
  self.seen_open=[False,False];self.closure_seen=False;self.release_pose=None
  self.release_hand=None;self.hold_start=None;self.done=False
 def update(self,o,e):
  if self.done:raise RuntimeError("reset required")
  t=float(o["sim_time"])
  if t<=self.last:raise ValueError("nonmonotonic observation")
  self.last=t
  reason=None;warnings=[]
  if not e.get("valid") or not e.get("sensors_valid"):reason="sensor_invalid"
  else:
   if not e.get("no_safety_violation"):warnings.append("safety_violation")
   if not e.get("recorded_dimm_contacts_under_penetration_limit"):warnings.append("contact_penetration_limit")
   if warnings and self.safety_mode=="stop":reason=warnings[0]
  m=o["task_truth"]["metrics"];angles=[m["left_angle_deg"],m["right_angle_deg"]]
  self.seen_open[0]|=angles[0]<=-15;self.seen_open[1]|=angles[1]>=15
  pos=np.asarray(m["dimm_position_in_socket_m"]);d=np.asarray(o["task_truth"]["poses"]["dimm"])
  assistance=o["task_truth"].get("assistance",{}).get("sides",{})
  drive_off=len(assistance)==2 and all(v.get("stiffness")==0 for v in assistance.values())
  seated=bool(max(map(abs,angles))<=3.5 and np.linalg.norm(pos[:2])<=.0025
    and abs(pos[2]-.0003963)<=.0015 and e.get("fixed_base_contact_force_N",0)>.05 and drive_off)
  if all(self.seen_open) and seated:self.closure_seen=True
  forces=e.get("measurements",{}).get("force_peaks_N",{})
  free=bool(o["gripper_width_m"]>=.019 and max(forces.get("finger_left_N",1e9),forces.get("finger_right_N",1e9))<.1)
  stable=False;retracted=False
  if seated and self.closure_seen and free and reason is None and not warnings:
   if self.release_pose is None:
    self.release_pose=d.copy();self.release_hand=np.asarray(o["ee_position_m"]).copy()
   shift=float(np.linalg.norm(d[:3]-self.release_pose[:3]))
   tilt=float(np.degrees((Rotation.from_quat(d[3:])*Rotation.from_quat(self.release_pose[3:]).inv()).magnitude()))
   stable=shift<=.0005 and tilt<=.5
   retracted=bool(o["ee_position_m"][2]-self.release_hand[2]>=.029)
   if not stable:reason="release_instability"
   if stable and retracted:
    if self.hold_start is None:self.hold_start=t
   else:self.hold_start=None
  else:
   self.hold_start=None
   self.release_pose=None;self.release_hand=None
  success=bool(reason is None and self.hold_start is not None and t-self.hold_start>=2.-1e-6)
  terminated=bool(success or reason)
  truncated=bool(not terminated and self.timeout_s is not None and t-self.start>=self.timeout_s)
  self.done=terminated or truncated
  return dict(version=VERSION,success=success,failure=reason,terminated=terminated,truncated=truncated,
    mask=0. if terminated else 1.,valid_for_training=reason!="sensor_invalid" and self.safety_mode=="stop",
     warnings=warnings,safety_mode=self.safety_mode,
    accepted=success,seated_operational=seated,closure_history=self.closure_seen,
    released=free,stable=stable,retracted=retracted,
    hold_s=0. if self.hold_start is None else t-self.hold_start,
    scope="accepted-scene operational end conditions; hook relationship inherited from accepted geometry")
