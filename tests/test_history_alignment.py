import copy,json,unittest
from pathlib import Path
from runners.common import config,resolve
from tasks.task1_dimm_insert.env.dataset import Dataset
from tasks.task1_dimm_insert.env.align_history import check_window
class Alignment(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.cfg=config();d=Dataset(resolve(cls.cfg["dataset"]))
  cls.fragments=d.rows[d.manifest["native_transitions"]:]
  item=next(r for r in cls.fragments if Path(r["episode"]).name.startswith("03_"))
  with (Path(item["episode"])/"transitions.jsonl").open() as f:
   for i,l in enumerate(f):
    if i==item["source_indices"][0]:cls.row=json.loads(l);break
 def test_actual_historical_window(self):
  action,reason=check_window(self.row,self.row["diagnostics"]["executed_substeps"],self.cfg["action"])
  self.assertIsNone(reason);self.assertEqual(len(action),7)
  self.assertTrue(any(r["label"]["reward"]>0 for r in self.fragments))
 def test_average_action_not_accepted(self):
  r=copy.deepcopy(self.row);s=r["diagnostics"]["executed_substeps"];s[1]["action"][0]+=.001
  self.assertEqual(check_window(r,s,self.cfg["action"])[1],"different_substep_actions")
 def test_bad_target_not_accepted(self):
  r=copy.deepcopy(self.row);s=r["diagnostics"]["executed_substeps"]
  s[0]["diagnostics"]["controller_commands"][4]["target_position_m"][0]+=.001
  self.assertEqual(check_window(r,s,self.cfg["action"])[1],"position_integration")
 def test_missing_command_not_accepted(self):
  r=copy.deepcopy(self.row);s=r["diagnostics"]["executed_substeps"]
  s[0]["diagnostics"]["controller_commands"].pop()
  self.assertEqual(check_window(r,s,self.cfg["action"])[1],"command_coverage")
