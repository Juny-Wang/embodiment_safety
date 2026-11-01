import unittest, copy
from tasks.task1_dimm_insert.env.warning_mode import validate_safety_mode, WarningTracker
from tasks.task1_dimm_insert.env.termination import EpisodeEnd
from tasks.task1_dimm_insert.env.episode import Episode
def observation(t):
    return dict(sim_time=t,gripper_width_m=.02,ee_position_m=[0.,0.,1.],
      task_truth=dict(metrics=dict(left_angle_deg=0.,right_angle_deg=0.,dimm_position_in_socket_m=[.2,0,0]),
      poses=dict(dimm=[0.,0.,.75,0.,0.,0.,1.]),assistance={}))
def evidence():
    return dict(valid=True,sensors_valid=True,no_safety_violation=True,
       recorded_dimm_contacts_under_penetration_limit=True, fixed_base_contact_force_N=0.,
       grasp_in_allowed_band=False,bilateral_grasp_stable=False,dimm_tracks_gripper=False,rack_clear=False,
       measurements=dict(force_peaks_N=dict(finger_left_N=26.,finger_right_N=0.)))
class WarningModeTests(unittest.TestCase):
    def test_guard(self):
        self.assertEqual(validate_safety_mode("stop",False,True),"stop")
        self.assertEqual(validate_safety_mode("warn",True,False),"warn")
        for eval_,learn in [(False,False),(True,True),(False,True)]:
            with self.assertRaises(ValueError):validate_safety_mode("warn",eval_,learn)
        with self.assertRaises(ValueError):validate_safety_mode("off",True,False)
    def test_default_still_stops(self):
        end=EpisodeEnd(observation(0));e=evidence();e["no_safety_violation"]=False
        r=end.update(observation(.1),e)
        self.assertTrue(r["terminated"]);self.assertEqual(r["failure"],"safety_violation")
    def test_warn_continues_and_keeps_truth(self):
        end=EpisodeEnd(observation(0),safety_mode="warn");e=evidence()
        e["no_safety_violation"]=False;e["recorded_dimm_contacts_under_penetration_limit"]=False
        r=end.update(observation(.1),e)
        self.assertFalse(r["terminated"]);self.assertFalse(r["success"]);self.assertIsNone(r["failure"])
        self.assertEqual(r["warnings"],["safety_violation","contact_penetration_limit"])
        self.assertFalse(r["valid_for_training"]);self.assertFalse(e["no_safety_violation"])
        self.assertFalse(end.update(observation(.2),e)["terminated"])
    def test_sensor_failure_still_stops(self):
        end=EpisodeEnd(observation(0),safety_mode="warn");e=evidence();e["valid"]=False
        self.assertEqual(end.update(observation(.1),e)["failure"],"sensor_invalid")
    def test_wall_independent_simulation_timeout_retained(self):
        end=EpisodeEnd(observation(0),timeout_s=.1,safety_mode="warn");e=evidence();e["no_safety_violation"]=False
        r=end.update(observation(.1),e)
        self.assertTrue(r["truncated"]);self.assertFalse(r["terminated"])
    def test_reward_does_not_accept_unsafe_opening(self):
        ep=Episode([0,0,0,1],safety_mode="warn");ep.reset(observation(0))
        e=evidence();e["no_safety_violation"]=False
        for t in [.1,.2,.3,.4]:
            o=observation(t);o["task_truth"]["metrics"].update(left_angle_deg=-16,right_angle_deg=16)
            r=ep.step(o,e,dict(left_press_N=0.,right_press_N=0.))
            self.assertFalse(r["terminated"]);self.assertEqual(r["reward"],0)
        self.assertEqual(ep.evaluator.reward.progress(),[0,0,0,0])
    def test_warning_is_not_safe_terminal_success(self):
        end=EpisodeEnd(observation(0),safety_mode="warn")
        end.seen_open=[True,True];end.closure_seen=True;end.hold_start=0.
        o=observation(3);o["task_truth"]["metrics"]["dimm_position_in_socket_m"]=[0,0,.0003963]
        o["task_truth"]["assistance"]={"sides":{"left":{"stiffness":0},"right":{"stiffness":0}}}
        e=evidence();e.update(no_safety_violation=False,fixed_base_contact_force_N=.2)
        self.assertFalse(end.update(o,e)["success"])
    def test_tracker_preserves_peak_and_rejects_invalid(self):
        w=WarningTracker();m=dict(finger_left_N=26.,forbidden_contact=0.,unclassified_grasp_contact=0.)
        r=w.observe(1.,m,dict(finger_left_N=5.))
        self.assertEqual(r["violations"]["finger_left_N"]["value"],26.)
        self.assertFalse(w.summary()["safety_pass"])
        m["finger_left_N"]=float("nan")
        with self.assertRaises(ValueError):w.observe(2.,m,dict(finger_left_N=5.))
if __name__=="__main__":unittest.main()
