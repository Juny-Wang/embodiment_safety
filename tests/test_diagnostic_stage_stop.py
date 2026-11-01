import unittest
from tasks.task1_dimm_insert.env.diagnostic_stop import validate_stage_stop,stage_verified
class StopTests(unittest.TestCase):
 def test_requires_real_event(self):
  self.assertFalse(stage_verified({"events":[],"current_status":{"both_open":{"status":"pass"}}},"both_open"))
  self.assertTrue(stage_verified({"events":[{"stage":"both_open"}]},"both_open"))
 def test_failure_and_truncation_override_event(self):
  for key,value in [("failure","contact_penetration_limit"),("terminated",True),("truncated",True)]:
   self.assertFalse(stage_verified({"events":[{"stage":"both_open"}],key:value},"both_open"))
 def test_evaluation_only(self):
  self.assertEqual(validate_stage_stop("both_open",True,False),"both_open")
  for evaluation,learn in [(False,False),(True,True)]:
   with self.assertRaises(ValueError):validate_stage_stop("both_open",evaluation,learn)
 def test_default_disabled(self):
  self.assertEqual(validate_stage_stop("",False,True),"")
  self.assertFalse(stage_verified({"events":[{"stage":"both_open"}]},""))
if __name__=="__main__":unittest.main()
