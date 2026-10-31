"""Launch one fresh native Isaac process; no saved stage mutation."""
import os,argparse,subprocess,json
from pathlib import Path
from .common import ROOT,config,resolve
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--config",default="configs/experiments/task1_serl.json")
 p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True)
 p.add_argument("--smoke",action="store_true");p.add_argument("--learn",action="store_true")
 p.add_argument("--max-steps",type=int);p.add_argument("--seed",type=int);p.add_argument("--deterministic",action="store_true")
 a=p.parse_args();c=config(a.config);out=resolve(a.output)
 checkpoint=resolve(a.checkpoint)
 if not checkpoint.is_file() or not checkpoint.with_suffix(".json").is_file():
  raise FileNotFoundError("Completed checkpoint and metadata required before starting Isaac: "+str(checkpoint))
 if not out.is_relative_to(ROOT/"outputs"):raise ValueError("Repository outputs only")
 if a.max_steps is not None and a.max_steps<1:raise ValueError("max-steps")
 env=os.environ.copy();env.update(TASK1_CONFIG=str(resolve(a.config)),SERL_INPUT_CHECKPOINT=str(resolve(a.checkpoint)),
  TASK1_SMOKE="1" if a.smoke else "0",TASK1_ONLINE_LEARN="1" if a.learn else "0",
  TASK1_MAX_STEPS=str(a.max_steps or (8 if a.smoke else round(c["timeout_s"]*10))),
  TASK1_EPISODE_SEED=str(a.seed if a.seed is not None else c["seed"]),
  TASK1_DETERMINISTIC="1" if a.deterministic else "0")
 cmd=[os.environ.get("ISAAC_PYTHON","/mnt/data2/isaacsim-5.0.0/python.sh"),"-B",
  str(ROOT/"tasks/task1_dimm_insert/env/actor.py"),str(out)]
 result=subprocess.run(cmd,cwd=ROOT,env=env)
 if result.returncode or (out/"actor_error.json").exists() or (out/"worker_error.json").exists():
  raise RuntimeError("Rollout failed; inspect "+str(out))
 if not (out/"result.json").exists():raise RuntimeError("Missing rollout result")
 print((out/"result.json").read_text())
if __name__=="__main__":main()
