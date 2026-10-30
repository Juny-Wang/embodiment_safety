"""Training code/data integrity checks; no Isaac or PhysX startup."""
import ast,json,hashlib,importlib,subprocess
from pathlib import Path
from .common import ROOT,config,resolve
def main():
 c=config()
 files=[]
 for rel in ("tasks/task1_dimm_insert","algos/serl","runners","tests"):
  for p in (ROOT/rel).rglob("*.py"):
   ast.parse(p.read_text(),filename=str(p));files.append(p)
 from tasks.task1_dimm_insert.env.data import verify_manifest
 verify_manifest(ROOT/c["scene"].replace("collection.usda","manifest.json"))
 from tasks.task1_dimm_insert.env.dataset import Dataset,digest
 from algos.serl.replay import Journal
 data=Dataset(resolve(c["dataset"]));count=0
 for p,h in data.manifest["image_hashes"].items():
  if digest(p)!=h:raise ValueError("RGB hash mismatch: "+p)
 for tr,meta in Journal(resolve(c["replay"])).records():count+=1
 if not count:raise ValueError("Empty online buffer")
 for i in (0,len(data)-1):data.get(i)
 report=dict(passed=True,parsed_python_files=len(files),training_transitions=len(data),
  online_transitions=count,rgb_files_verified=len(data.manifest["image_hashes"]),
  reward_version=c["reward_version"],interface_version=c["interface_version"],
  independent_demonstrations=1,held_out_evaluation_episodes=0,
  full_policy_training_verified=False)
 out=ROOT/"outputs/task1_migration/preflight.json";out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
