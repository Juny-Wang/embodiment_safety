import unittest
from algos.diffusion_policy.data import complete_native_opening_starts

class CompleteNativeOpeningTest(unittest.TestCase):
 def test_all_contiguous_and_no_other_task_rows(self):
  rows=[dict(episode="policy_reset_pair_002/native",tick=i) for i in range(520)]
  ids=complete_native_opening_starts(rows,16)
  self.assertEqual(ids,list(range(495)))
  self.assertIn(145,ids)  # spans prior validation boundary
  self.assertEqual(rows[ids[-1]+15]["tick"],509)
 def test_no_gap_or_cross_episode(self):
  rows=[dict(episode="policy_reset_pair_002/native",tick=i) for i in range(40)]
  rows[20]["tick"]=999
  ids=complete_native_opening_starts(rows,16)
  self.assertFalse(any(k<=20<=k+15 for k in ids))
  rows[20]["tick"]=20
  rows[20]["episode"]="other"
  ids=complete_native_opening_starts(rows,16)
  self.assertFalse(any(k<=20<=k+15 for k in ids))
if __name__=="__main__":unittest.main()
