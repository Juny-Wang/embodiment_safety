import unittest
from tasks.task1_dimm_insert.env.reward import StageReward,STAGES
from tasks.task1_dimm_insert.env.policy import attach_progress
import numpy as np
def ev(t,**kw):
    return dict(valid=True,sim_time=t,source="synthetic_unit_fixture",
      sensors_valid=True,no_unintended_penetration=True,no_safety_violation=True,**kw)
def stage(i,t):
    return ev(t,**{k:True for k in STAGES[i][2] if k!="no_unintended_penetration"})
class Tests(unittest.TestCase):
    def test_once_and_reset(self):
        r=StageReward()
        self.assertEqual(r.update(.1,stage(0,.1))["reward"],.05)
        self.assertEqual(r.update(.2,stage(0,.2))["reward"],0)
        self.assertEqual(r.reset(),[0,0,0,0])
        self.assertEqual(r.update(.1,stage(0,.1))["reward"],.05)
    def test_unknown_not_rewarded(self):
        r=StageReward();x=r.update(.1,{})
        self.assertEqual(x["reward"],0);self.assertEqual(x["current_status"]["both_open"]["status"],"unknown")
    def test_stale_sensor_and_penetration(self):
        for changes in [dict(sim_time=0),dict(sensors_valid=False),dict(no_unintended_penetration=False),dict(source="")]:
            e=stage(3,.1);e.update(changes)
            self.assertEqual(StageReward().update(.1,e)["reward"],0)
    def test_total_and_failure(self):
        r=StageReward();total=0
        for i in range(4):
            t=(i+1)/10;total+=r.update(t,stage(i,t))["reward"]
        e=stage(4,.5);e.update(stage(3,.5))
        result=r.update(.5,e,terminated=True)
        self.assertAlmostEqual(total+result["reward"],1.4)
        self.assertTrue(result["success"]);self.assertEqual(result["mask"],0)
        with self.assertRaises(RuntimeError):r.update(.6,e)
        r.reset();r.update(.1,stage(0,.1))
        result=r.update(.2,stage(1,.2),terminated=True,failure="sensor_failure")
        self.assertFalse(result["success"]);self.assertEqual(result["reward"],0)
        self.assertEqual(result["progress_after"],[1,0,0,0])
    def test_complete_requires_current_lock(self):
        r=StageReward()
        with self.assertRaises(ValueError):r.update(.1,stage(4,.1),terminated=True)
    def test_truncation_and_history(self):
        r=StageReward();x=r.update(.1,stage(0,.1),truncated=True)
        self.assertEqual(x["mask"],1);self.assertFalse(x["success"])
        obs={"state":np.zeros((1,27),np.float32)}
        a=attach_progress(obs,x["progress_before"]);b=attach_progress(obs,x["progress_after"])
        self.assertEqual(a["state"].shape,(1,31))
        self.assertEqual(a["state"][0,-4],0);self.assertEqual(b["state"][0,-4],1)
    def test_angle_alone_and_bad_time(self):
        r=StageReward();self.assertEqual(r.update(.1,ev(.1,left_angle_deg=-18,right_angle_deg=18))["reward"],0)
        with self.assertRaises(ValueError):r.update(.1,ev(.1))
if __name__=="__main__":unittest.main()
