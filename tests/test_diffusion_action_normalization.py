import unittest
import numpy as np
import torch
from algos.diffusion_policy.common import VENDOR
from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
class ActionRangeTest(unittest.TestCase):
 def test_small_nonconstant_velocity_is_scaled_and_invertible(self):
  data=np.array([[0.,0.,0.],[2e-5,-4e-5,0.]],dtype=np.float32)
  old=SingleFieldLinearNormalizer.create_fit(data,mode="limits")
  new=SingleFieldLinearNormalizer.create_fit(data,mode="limits",range_eps=1e-8)
  self.assertEqual(float(old.params_dict["scale"][0]),1.)
  self.assertGreater(float(new.params_dict["scale"][0]),90000.)
  self.assertGreater(float(new.params_dict["scale"][1]),49000.)
  self.assertTrue(torch.allclose(new.unnormalize(new.normalize(data)),torch.from_numpy(data),atol=1e-10))
  self.assertTrue(torch.isfinite(new.normalize(data)).all())
 def test_network_residual_not_converted_to_large_angular_rate(self):
  data=np.array([[0.],[2e-5]],dtype=np.float32)
  new=SingleFieldLinearNormalizer.create_fit(data,mode="limits",range_eps=1e-8)
  z=new.normalize(np.zeros((1,1),dtype=np.float32))
  delta=new.unnormalize(z+0.001)-new.unnormalize(z)
  self.assertLess(abs(float(delta[0,0])*90),1e-5)
if __name__=="__main__":unittest.main()
