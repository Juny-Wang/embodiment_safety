"""Policy IK control with unchanged pose, joint-limit and drive-rate constraints."""
import numpy as np
from scipy.spatial.transform import Rotation

def tick(self,dt):
 if not self.ready:return
 self.elapsed+=dt
 if not self.armed or self.elapsed<self.cfg["control"]["ik_period_s"]:return
 h=min(self.elapsed,.05);self.elapsed=0
 p,Rm=self.pose();R=Rotation.from_matrix(Rm);gp,gR=self.goal
 d=gp-p;dist=np.linalg.norm(d)
 if dist>self.cfg["control"]["tracking_error_stop_m"]:self.stop("EE TRACKING ERROR");return
 cp,cR=self.cart_command;cd=gp-cp;cdist=np.linalg.norm(cd)
 p_cmd=cp+cd*min(1.,self.cfg["control"]["translation_speed_m_s"]*h/max(cdist,1e-12))
 rv=(gR*cR.inv()).as_rotvec();a=np.linalg.norm(rv)
 R_cmd=Rotation.from_rotvec(rv*min(1.,self.cfg["control"]["rotation_speed_rad_s"]*h/max(a,1e-12)))*cR
 qq=R_cmd.as_quat();wxyz=np.r_[qq[3],qq[:3]]
 def errors(x):
  pp,rr=self.solver.compute_forward_kinematics(self.cfg["control"]["ee_frame"],np.asarray(x,dtype=float))
  return np.asarray(pp)-p_cmd,(Rotation.from_matrix(rr)*R_cmd.inv()).as_rotvec()
 pos_tol=self.cfg["control"].get("ik_position_tolerance_m",1e-6)
 rot_tol=self.cfg["control"].get("ik_rotation_tolerance_rad",1e-4)
 candidate=self.last_solution
 pe,re=errors(candidate) if candidate is not None else (np.full(3,np.inf),np.full(3,np.inf))
 ok=candidate is not None and np.linalg.norm(pe)<=pos_tol and np.linalg.norm(re)<=rot_tol
 used_cache=bool(ok)
 solve_info=dict(attempts=[],accepted=bool(ok),solve_wall_s=0.)
 if not ok:
  from tasks.task1_dimm_insert.env.ik import solve
  seed=candidate if candidate is not None else self.get_joint_positions()[self.arm_indices].astype(float)
  candidate,solve_info=solve(errors,seed,self.lower,self.upper,pos_tol,rot_tol,
    self.cfg["control"].get("ik_max_evaluations",40),self.cfg["control"].get("ik_solver","trf"))
  self.fallback_count+=1;pe,re=errors(candidate)
  ok=solve_info["local_solver_success"]
 self.ik_diagnostics={"local_solver_success":bool(ok),"used_verified_cache":used_cache,"position_error_m":float(np.linalg.norm(pe)),
   "orientation_error_rad":float(np.linalg.norm(re)),"fallback_count":self.fallback_count}
 self.ik_diagnostics.update(solve_info)
 self.ik_diagnostics.update(
   candidate_joint_positions=np.asarray(candidate).tolist(),
   measured_joint_positions=self.get_joint_positions()[self.arm_indices].tolist(),
   lower_limits_rad=self.lower.tolist(),upper_limits_rad=self.upper.tolist(),
   joint_margin_rad=np.minimum(candidate-self.lower,self.upper-candidate).tolist(),
   position_tolerance_m=pos_tol,orientation_tolerance_rad=rot_tol,
   requested_goal_position_m=np.asarray(gp).tolist(),requested_goal_quaternion_xyzw=gR.as_quat().tolist(),
   commanded_position_m=np.asarray(p_cmd).tolist(),commanded_quaternion_xyzw=R_cmd.as_quat().tolist(),
   measured_ee_position_m=np.asarray(p).tolist(),measured_ee_quaternion_xyzw=R.as_quat().tolist())
 if not np.isfinite(candidate).all() or np.any(candidate<self.lower) or np.any(candidate>self.upper) or np.linalg.norm(pe)>pos_tol or np.linalg.norm(re)>rot_tol:
  self.stop("IK RESIDUAL/LIMIT FAILURE - no new motion");return
 self.last_solution=candidate.copy()
 self.cart_command=(p_cmd,R_cmd);desired=np.clip(candidate+self.hold_bias[self.arm_indices],self.lower,self.upper)
 q=np.asarray(self.last_action).copy()
 desired=np.asarray(desired,dtype=float)
 indices=self.arm_indices
 delta=np.clip(desired-q[indices],-self.cfg["control"]["arm_joint_speed_rad_s"]*h,self.cfg["control"]["arm_joint_speed_rad_s"]*h)
 q[indices]+=delta
 self.ik_diagnostics.update(command_interval_s=float(h),
   arm_target_step_rad=delta.tolist(),arm_target_rad=q[indices].tolist(),
   arm_speed_limit_rad_s=self.cfg["control"]["arm_joint_speed_rad_s"])
 measured=self.robot.get_joint_positions()
 each=self.width_goal/2.
 finger_step=self.cfg["control"]["gripper_speed_m_s"]*h/2.
 q[self.fingers]=self.last_action[self.fingers]+np.clip(each-self.last_action[self.fingers],-finger_step,finger_step)
 self.apply_targets(q)
 self.last_action=q.copy();self.last_q=np.array(measured).copy();self.commands+=1
