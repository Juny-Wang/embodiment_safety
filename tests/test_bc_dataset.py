import unittest
from tasks.task1_dimm_insert.env.bc_dataset import command_at_observation
class BCCommandTests(unittest.TestCase):
 def row(self):
  return dict(observations=dict(sim_time=1.),next_observations=dict(sim_time=1.1),
   actions=[.3]*7,diagnostics=dict(executed_substeps=[
    dict(start=1.,end=1.05,action=[.1]*7,diagnostics=dict(controller_commands=[dict(time=1.)])),
    dict(start=1.05,end=1.1,action=[.5]*7,diagnostics=dict(controller_commands=[dict(time=1.05)]))]))
 def test_first_command_not_average(self):
  a,dt,_=command_at_observation(self.row())
  self.assertEqual(a,[.1]*7);self.assertAlmostEqual(dt,.05)
 def test_reject_shifted_command(self):
  r=self.row();r["diagnostics"]["executed_substeps"][0]["start"]=1.01
  with self.assertRaises(ValueError):command_at_observation(r)
if __name__=="__main__":unittest.main()
