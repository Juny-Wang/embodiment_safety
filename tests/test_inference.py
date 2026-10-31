import unittest
from algos.serl.inference import sample_policy
class Inference(unittest.TestCase):
 def test_deterministic_uses_vendor_argmax_without_seed(self):
  class Agent:
   def sample_actions(self,*,observations,argmax=False,seed=None):
    assert argmax and seed is None
    return "mode"
  self.assertEqual(sample_policy(Agent(),{},key=123,deterministic=True),"mode")
 def test_training_keeps_seeded_sampling(self):
  class Agent:
   def sample_actions(self,*,observations,argmax=False,seed=None):
    assert not argmax and seed==123
    return "sample"
  self.assertEqual(sample_policy(Agent(),{},key=123,deterministic=False),"sample")
