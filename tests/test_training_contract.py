import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from runners.common import ROOT,config,resolve
from tasks.task1_dimm_insert.env.dataset import Dataset
from tasks.task1_dimm_insert.env.policy import attach_progress
from tasks.task1_dimm_insert.env.transition import labels
from algos.serl.replay import Journal
class Contract(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.cfg=config();cls.data=Dataset(resolve(cls.cfg["dataset"]))
 def test_terminal_and_once(self):
  rows=self.data.rows[:self.data.manifest.get("native_transitions",len(self.data))]
  self.assertTrue(rows[-1]["label"]["success"])
  self.assertEqual(rows[-1]["label"]["mask"],0)
  names=[e["stage"] for r in rows for e in r["label"]["events"]]
  self.assertEqual(len(names),5);self.assertEqual(len(set(names)),5)
  self.assertAlmostEqual(sum(r["label"]["reward"] for r in rows),1.4)
  self.assertTrue(all(abs(r["dt_s"]-.1)<1e-6 for r in rows))
 def test_progress_boundary(self):
  rows=self.data.rows[:self.data.manifest.get("native_transitions",len(self.data))]
  for previous,current in zip(rows,rows[1:]):
   self.assertEqual(previous["label"]["progress_after"],current["label"]["progress_before"])
  self.assertEqual(rows[0]["label"]["progress_before"],[0]*4)
 def test_no_privileged_policy_inputs(self):
  row=self.data.get(0)
  for key in ("observations","next_observations"):
   self.assertEqual(set(row[key]),{"state","external","wrist_left","wrist_right"})
   self.assertEqual(row[key]["state"].shape,(1,31))
 def test_short_transition(self):
  with self.assertRaises(ValueError):labels(7/240,7,truncated=True)
  x=labels(7/240,7,terminated=True,reason="force_limit")
  self.assertEqual(x["mask"],0)
 def test_journal_integrity(self):
  tr,meta=next(Journal(resolve(self.cfg["replay"])).records())
  with tempfile.TemporaryDirectory(dir=ROOT/"outputs/task1_migration") as d:
   j=Journal(d);j.append("one",tr,meta)
   self.assertEqual(len(list(j.records())),1)
   with self.assertRaises(FileExistsError):j.append("one",tr,meta)
   with self.assertRaises(ValueError):j.append("bad",tr,dict(meta,dt_s=float("nan")))
   with self.assertRaises(ValueError):j.append("bad",tr,dict(meta,interface_version="state27"))
   p=Path(d)/"one.npz"
   with p.open("ab") as f:f.write(b"corrupt")
   with self.assertRaises(ValueError):list(j.records())
if __name__=="__main__":unittest.main()
