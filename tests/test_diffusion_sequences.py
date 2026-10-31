import unittest
from algos.diffusion_policy.data import sequence_indices
class DiffusionSequenceTests(unittest.TestCase):
 def test_gap_and_episode_boundaries(self):
  ep=["a"]*12+["b"]*20;ticks=list(range(12))+list(range(20))
  a,b=sequence_indices(ep,ticks,4)
  for k in a+b:
   self.assertEqual(len(set(ep[k:k+4])),1)
   self.assertEqual(ticks[k+3]-ticks[k],3)
 def test_split_no_overlap(self):
  a,b=sequence_indices(["a"]*1000,list(range(1000)),16)
  ta={i for k in a for i in range(k,k+16)};va={i for k in b for i in range(k,k+16)}
  self.assertTrue(a and b);self.assertFalse(ta&va)
if __name__=="__main__":unittest.main()
