import unittest
from algos.diffusion_policy.data import complete_native_starts, complete_native_opening_starts
from tasks.task1_dimm_insert.env.reward import StageReward
from tasks.task1_dimm_insert.env.diagnostic_stop import stage_verified
from tasks.task1_dimm_insert.env.termination import EpisodeEnd

class FullFlowCoverageTest(unittest.TestCase):
 def test_complete_native_flow_includes_handoffs_and_tail(self):
  rows=[dict(episode="policy_reset_pair_002/native",tick=i) for i in range(1349)]
  ids=complete_native_starts(rows,16)
  self.assertEqual(ids,list(range(1334)))
  self.assertIn(505,ids)
  self.assertEqual(rows[ids[-1]+15]["tick"],1348)
  self.assertEqual(len(complete_native_opening_starts(rows,16)),495)
 def test_no_historical_rows_gaps_or_cross_episode(self):
  rows=[dict(episode="policy_reset_pair_002/native",tick=i) for i in range(50)]
  rows[20]["tick"]=999
  rows[35]["episode"]="historical"
  ids=complete_native_starts(rows,16)
  self.assertFalse(any(k<=20<=k+15 or k<=35<=k+15 for k in ids))
 def test_invalid_horizon(self):
  with self.assertRaises(ValueError):complete_native_starts([],0)

class OpeningChecksRemainTest(unittest.TestCase):
 def evidence(self,clear=True):
  return dict(valid=True,sim_time=.1,source="test measured evidence",
   sensors_valid=True,no_unintended_penetration=True,no_safety_violation=True,
   insertion_path_clear=clear,hands_clear=True,opening_survives_withdrawal=True,
   key_orientation_correct=True,lateral_alignment_ok=True,orientation_ok=True)
 def test_no_diagnostic_stop_does_not_disable_opening_label(self):
  label=StageReward().update(.1,self.evidence())
  self.assertEqual(label["current_status"]["both_open"]["status"],"pass")
  self.assertIn("both_open",[e["stage"] for e in label["events"]])
  self.assertFalse(stage_verified(label,""))
  self.assertTrue(stage_verified(label,"both_open"))
 def test_blocked_path_cannot_receive_opening_or_insertion_ready(self):
  label=StageReward().update(.1,self.evidence(False))
  self.assertEqual(label["current_status"]["both_open"]["status"],"fail")
  self.assertEqual(label["current_status"]["insertion_ready"]["status"],"fail")
  self.assertEqual(label["events"],[])
 def test_seated_pose_without_prior_opening_cannot_complete(self):
  end=EpisodeEnd(dict(sim_time=0.))
  obs=dict(sim_time=.1,task_truth=dict(metrics=dict(left_angle_deg=0.,
   right_angle_deg=0.,dimm_position_in_socket_m=[0,0,.0003963]),
   poses=dict(dimm=[0,0,0,0,0,0,1]),
   assistance=dict(sides=dict(left=dict(stiffness=0.),right=dict(stiffness=0.)))),
   gripper_width_m=.02,ee_position_m=[0,0,.1])
  ev=dict(valid=True,sensors_valid=True,no_safety_violation=True,
   recorded_dimm_contacts_under_penetration_limit=True,fixed_base_contact_force_N=.2,
   measurements=dict(force_peaks_N=dict(finger_left_N=0.,finger_right_N=0.)))
  for i in range(1,40):
   obs["sim_time"]=i*.1
   result=end.update(obs,ev)
   self.assertFalse(result["success"])
   self.assertFalse(result["closure_history"])
if __name__=="__main__":unittest.main()
