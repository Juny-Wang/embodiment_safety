import unittest
from algos.diffusion_policy.common import prediction_type
class TargetTests(unittest.TestCase):
 def test_old_checkpoint_defaults_to_epsilon(self):
  self.assertEqual(prediction_type({"state_dim":31}),"epsilon")
 def test_explicit_sample(self):
  self.assertEqual(prediction_type({"prediction_type":"sample"}),"sample")
 def test_unsupported_rejected(self):
  with self.assertRaises(ValueError):prediction_type({"prediction_type":"other"})
if __name__=="__main__":unittest.main()
