import unittest
from tasks.task1_dimm_insert.env.evidence import Evidence
from tasks.task1_dimm_insert.env.reward import StageReward
def obs(t,z=.1):
 return dict(sim_time=t,sensor_valid=True,task_truth={"poses":{"dimm":[0,0,z,0,0,0,1]}},
             ee_position_m=[0,0,z+.02],ee_quaternion_xyzw=[0,0,0,1])
def frames(start):
 out=[]
 for i in range(24):
  t=start+(i+1)/240
  m=dict(left_press_N=0.,right_press_N=0.,dimm_environment_N=0.,finger_left_N=2.,finger_right_N=2.,
         forbidden_contact=0.,unclassified_grasp_contact=0.)
  c=dict(a="/World/MatchedDIMM/PCB",b="/World/Robot/panda_leftfinger/collision",
         normal_force_N=2.,separation_m=0.,grasp_region="ALLOWED")
  out.append((i+1,dict(sim_time=t,metrics=m,contacts=[c],controller_error=None)))
 return out
class Tests(unittest.TestCase):
 def test_stable_tracking_and_clearance(self):
  x=Evidence(obs(0,.1))
  for i in range(4):
   e=x.update(obs((i+1)*.1,.16),i*.1,frames(i*.1))
  self.assertTrue(e["dimm_tracks_gripper"]);self.assertTrue(e["rack_clear"])
  self.assertNotIn("no_unintended_penetration",e)
  self.assertEqual(StageReward().update(.4,e)["reward"],0)
 def test_missing_physics(self):
  e=Evidence(obs(0)).update(obs(.1),0,[])
  self.assertFalse(e["valid"]);self.assertFalse(e["grasp_in_allowed_band"])
 def test_missing_metric(self):
  f=frames(0);del f[4][1]["metrics"]["finger_left_N"]
  self.assertFalse(Evidence(obs(0)).update(obs(.1),0,f)["valid"])
 def test_forbidden_and_penetrating(self):
  f=frames(0);f[0][1]["contacts"][0].update(grasp_region="FORBIDDEN",separation_m=-.001)
  e=Evidence(obs(0)).update(obs(.1),0,f)
  self.assertFalse(e["grasp_in_allowed_band"])
  self.assertFalse(e["recorded_dimm_contacts_under_penetration_limit"])
 def test_force_and_motion_not_grasp(self):
  x=Evidence(obs(0))
  for i in range(4):
   o=obs((i+1)*.1);o["ee_position_m"][0]=i*.01
   e=x.update(o,i*.1,frames(i*.1))
  self.assertFalse(e["dimm_tracks_gripper"])
 def test_gap(self):
  f=frames(0);del f[5]
  self.assertFalse(Evidence(obs(0)).update(obs(.1),0,f)["valid"])
if __name__=="__main__":unittest.main()
