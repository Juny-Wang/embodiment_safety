import json,unittest,math
from pathlib import Path
from safety_core import SafetyEngine,dimm_contact_kind,unexpected_robot_contact
CFG=json.loads((Path(__file__).parent/"safety_config.json").read_text())
class SafetyTests(unittest.TestCase):
 def test_warning_hysteresis_and_latched_hard_stop(self):
  e=SafetyEngine(CFG)
  get=lambda x:e.evaluate({"left_press_N":x},0,.001)
  self.assertEqual(get(2.5)["rows"][0]["color"],"YELLOW")
  self.assertEqual(get(2.2)["rows"][0]["color"],"YELLOW")
  self.assertEqual(get(2.)["rows"][0]["color"],"GREEN")
  self.assertTrue(get(3.)["stop"])
  self.assertTrue(get(0.)["stop"])
  self.assertTrue(e.acknowledge({"left_press_N":0},0,.001))
  self.assertFalse(get(0.)["stop"])
 def test_not_clear_while_unsafe(self):
  e=SafetyEngine(CFG);self.assertFalse(e.acknowledge({"left_press_N":4},0,.001))
 def test_gray_not_green_for_unknown_or_unconfigured(self):
  r=SafetyEngine(CFG).evaluate({"left_press_N":float("nan"),"left_torque_Nm":.01},0,.001)
  rows={x["key"]:x for x in r["rows"]}
  self.assertEqual(rows["left_press_N"]["color"],"GRAY")
  self.assertEqual(rows["left_torque_Nm"]["color"],"GRAY")
  self.assertFalse(r["task_success"])
 def test_tilt_gated_by_real_contact(self):
  e=SafetyEngine(CFG)
  self.assertFalse(e.evaluate({"tilt_deg":20,"dimm_contact_N":0},4,.001)["stop"])
  self.assertTrue(e.evaluate({"tilt_deg":20,"dimm_contact_N":.1},4,.001)["stop"])
 def test_step_specific_and_sensor_fault(self):
  e=SafetyEngine(CFG)
  self.assertFalse(e.evaluate({"axial_contact_N":6},0,.001)["stop"])
  self.assertTrue(e.evaluate({"axial_contact_N":6},4,.001)["stop"])
  self.assertTrue(SafetyEngine(CFG).evaluate({"sensor_error":"lost"},0,.001)["stop"])
 def test_single_frame_peak_is_not_filtered(self):
  e=SafetyEngine(CFG)
  e.evaluate({"left_press_N":0},0,.001)
  r=e.evaluate({"left_press_N":3.1},0,.001)
  self.assertTrue(r["stop"]);self.assertAlmostEqual(r["rows"][0]["exposure_s"],.001)
 def test_gripper_contact_is_not_insertion_resistance(self):
  d="/World/MatchedDIMM";r="/World/Robot"
  self.assertEqual(dimm_contact_kind(d+"/proxy",r+"/panda_leftfinger/collision",d,r),"finger_left")
  self.assertEqual(dimm_contact_kind(d+"/proxy","/World/MatchedSocket/guide",d,r),"environment")
  self.assertIsNone(dimm_contact_kind(d+"2/proxy",r+"/panda_leftfinger/collision",d,r))
 def test_unexpected_robot_contact_is_phase_specific(self):
  r="/World/Robot";d="/World/MatchedDIMM";s="/World/MatchedSocket/Asset"
  finger=r+"/panda_leftfinger/collision"
  self.assertFalse(unexpected_robot_contact(finger,s+"/LeftLatch/Geometry",r,d,s,0))
  self.assertTrue(unexpected_robot_contact(finger,s+"/LeftLatch/Geometry",r,d,s,4))
  self.assertFalse(unexpected_robot_contact(finger,d+"/proxy",r,d,s,1))
  self.assertTrue(unexpected_robot_contact(r+"/panda_link3/collision","/World/table",r,d,s,1))
  self.assertFalse(unexpected_robot_contact(r+"/panda_link0/collision","/World/table",r,d,s,1))
 def test_peak_survives_short_overload(self):
  e=SafetyEngine(CFG);e.evaluate({"left_press_N":3.5},0,.001)
  r=e.evaluate({"left_press_N":0.},0,.001)
  self.assertTrue(r["stop"]);self.assertEqual(r["rows"][0]["peak"],3.5)
  self.assertIn("left_press_N",r["latched_reasons"])
 def test_bad_time_rejected(self):
  with self.assertRaises(ValueError):SafetyEngine(CFG).evaluate({},0,0)
if __name__=="__main__":unittest.main()
