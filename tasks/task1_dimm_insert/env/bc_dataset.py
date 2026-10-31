"""BC observation/command pairs. These are NOT held-100ms SAC transitions."""
import json
from pathlib import Path
import numpy as np
from .dataset import digest
from . import policy

def command_at_observation(row):
    start=float(row["observations"]["sim_time"])
    subs=row.get("diagnostics",{}).get("executed_substeps",[])
    if subs:
        s=subs[0]
        if abs(float(s["start"])-start)>1e-6:
            raise ValueError("First command does not match observation boundary")
        action=s["action"]; duration=float(s["end"])-float(s["start"])
        diagnostics=s["diagnostics"]
    else:
        action=row["actions"]
        duration=float(row["next_observations"]["sim_time"])-start
        diagnostics=row["diagnostics"]
    a=np.asarray(action,float)
    if a.shape!=(7,) or not np.isfinite(a).all() or np.max(np.abs(a))>1.00001:
        raise ValueError("Invalid command")
    if duration<=0 or duration>.100001:
        raise ValueError("Invalid source command duration")
    commands=diagnostics.get("controller_commands",[])
    if not commands or abs(float(commands[0]["time"])-start)>1e-6:
        raise ValueError("Missing controller command at observation boundary")
    return a.tolist(),duration,diagnostics

class BCDataset:
    def __init__(self,path):
        self.path=Path(path)
        self.manifest=json.loads((self.path/"manifest.json").read_text())
        if self.manifest.get("kind")!="bc_observation_command_pairs_v001":
            raise ValueError("Not a BC pair dataset")
        if digest(self.path/"pairs.jsonl")!=self.manifest["index_sha256"]:
            raise ValueError("BC index changed")
        for p,h in self.manifest["source_hashes"].items():
            if digest(p)!=h:raise ValueError("BC source changed: "+p)
        self.rows=[json.loads(x) for x in (self.path/"pairs.jsonl").open()]
    def __len__(self):return len(self.rows)
    def get(self,i):
        r=self.rows[i]
        return dict(observations=policy.load_observation(r["observations"],r["episode"],r["progress_before"]),
                    actions=np.asarray(r["actions"],np.float32))
