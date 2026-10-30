"""Task x algorithm entry. No task logic or optimizer implementation here."""
import argparse,json
from datetime import datetime,timezone
from .common import ROOT,config,resolve,setup
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--task",default="task1_dimm_insert",choices=["task1_dimm_insert"])
 p.add_argument("--algo",default="serl",choices=["serl"])
 p.add_argument("--config",default="configs/experiments/task1_serl.json")
 p.add_argument("--updates",type=int);p.add_argument("--output");p.add_argument("--checkpoint")
 a=p.parse_args();c=config(a.config);setup(c["platform"])
 from tasks.task1_dimm_insert.env.dataset import Dataset
 from algos.serl.learner import train
 out=resolve(a.output) if a.output else resolve(c["output_root"])/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
 if not out.is_relative_to(ROOT/"outputs"):raise ValueError("Outputs must remain inside repository")
 result=train(Dataset(resolve(c["dataset"])),resolve(c["replay"]),out,updates=a.updates if a.updates is not None else c["updates"],
  batch_size=c["batch_size"],utd_ratio=c["utd_ratio"],seed=c["seed"],
  checkpoint=a.checkpoint or c["checkpoint"],discount=c["discount"])
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
