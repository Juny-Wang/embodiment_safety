import unittest
from tasks.task1_dimm_insert.env.transition import completed_partial
class PartialTerminal(unittest.TestCase):
 def tail(self,n=23,pending=True):
  return dict(observations=dict(sim_time=1.),next_observations=dict(sim_time=1.+n/240),controller_commands=[dict(time=1.+i/240) for i in range(n+int(pending))])
 def test_pending_not_executed(self):
  raw=self.tail();out=completed_partial(raw)
  self.assertEqual(out["physics_steps"],23)
  self.assertEqual(len(out["controller_commands"]),23)
  self.assertEqual(len(out["unobserved_commands"]),1)
  self.assertEqual(len(raw["controller_commands"]),24)
 def test_short(self):
  self.assertEqual(completed_partial(self.tail(7,False))["physics_steps"],7)
 def test_missing(self):
  raw=self.tail();raw["controller_commands"].pop(2)
  with self.assertRaises(ValueError):completed_partial(raw)
 def test_timestamp(self):
  raw=self.tail();raw["controller_commands"][3]["time"]+=.001
  with self.assertRaises(ValueError):completed_partial(raw)
 def test_zero(self):
  with self.assertRaises(ValueError):completed_partial(self.tail(0))
