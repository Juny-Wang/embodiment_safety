"""Repository-local configuration and explicit dependency setup."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def config(path="configs/experiments/task1_serl.json"):
 p=Path(path);p=p if p.is_absolute() else ROOT/p
 c=json.loads(p.read_text())
 if c["task"]!="task1_dimm_insert" or c["algo"]!="serl":raise ValueError("Unsupported task/algo combination")
 if c["physics_hz"]!=240 or c["control_hz"]!=10:raise ValueError("Current action contract requires 240Hz physics and 10Hz control")
 return c
def resolve(path):
 p=Path(path);return p if p.is_absolute() else ROOT/p
def setup(platform="cpu"):
 os.environ["JAX_PLATFORMS"]=platform
 os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
 deps=Path(os.environ.get("SERL_PYTHON_DEPS","/mnt/data2/wjy_SERL/SERL-runs/task1_serl_py310_jax"))
 # Dependencies contain installed third-party wheels; project SERL code is vendored here.
 sys.path[:0]=[str(ROOT/"algos/serl/vendor"),str(deps)]
