import unittest
from runners.eval_diffusion import wall_wait_timeout
from tasks.task1_dimm_insert.env.termination import EpisodeEnd
from tests.test_warning_only_evaluation import observation,evidence
class UnlimitedEvaluationTests(unittest.TestCase):
 def test_wall_zero_disables_timeout(self):
  self.assertIsNone(wall_wait_timeout(0))
  self.assertEqual(wall_wait_timeout(1800),1800)
 def test_invalid_wall_budget_rejected(self):
  for v in [-1,float('nan'),float('inf')]:
   with self.assertRaises(ValueError):wall_wait_timeout(v)
 def test_no_simulation_time_limit(self):
  end=EpisodeEnd(observation(0),timeout_s=None,safety_mode='warn')
  for t in [.1,240.,1e6]:
   r=end.update(observation(t),evidence())
   self.assertFalse(r['truncated']);self.assertFalse(r['terminated'])
 def test_default_simulation_limit_preserved(self):
  end=EpisodeEnd(observation(0))
  self.assertTrue(end.update(observation(240),evidence())['truncated'])
if __name__=='__main__':unittest.main()
