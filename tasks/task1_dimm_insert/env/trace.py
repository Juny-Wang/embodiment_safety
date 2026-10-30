import copy
import numpy as np

def traced_apply_velocity(original):
 def apply(controller,action,dt,cfg):
  measured,_=controller.pose()
  before=dict(measured_ee_position_m=np.asarray(measured).tolist(),
   integration_input_goal_position_m=np.asarray(controller.goal[0]).tolist(),
   integration_input_goal_quaternion_xyzw=controller.goal[1].as_quat().tolist(),
   integration_input_width_m=float(controller.width_goal),
   integration_action=np.asarray(action).tolist(),integration_dt_s=float(dt),
   integration_scales=copy.deepcopy(cfg))
  command=original(controller,action,dt,cfg)
  return dict(command,**before)
 return apply
