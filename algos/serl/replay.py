"""Append-only single-writer replay journal. JSON commit follows durable NPZ."""
from pathlib import Path
import json,hashlib,os
import numpy as np
VERSION="stage31_replay_journal_v001"
ROOT=Path("/mnt/data2/wjy_SERL/Embodied_Safety")
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Journal:
 def __init__(self,path):
  self.path=Path(path).resolve()
  if not self.path.is_relative_to(ROOT/"outputs"):raise ValueError("Remote output only")
  self.path.mkdir(parents=True,exist_ok=True)
 def append(self,key,tr,meta):
  if not key or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in key):raise ValueError("Bad key")
  required=("reward_version","interface_version","terminated","truncated","dt_s","physics_steps","eligible","source")
  if any(k not in meta for k in required):raise ValueError("Incomplete replay provenance")
  if meta["reward_version"]!="task1_stage_once_v001" or meta["interface_version"]!="task1_rgb_state31_reward_history_v001":raise ValueError("Interface mismatch")
  n=meta["physics_steps"];dt=meta["dt_s"]
  if not isinstance(n,int) or not 1<=n<=24 or not np.isfinite(dt) or abs(dt-n/240)>1e-6:raise ValueError("Invalid interval")
  if meta["terminated"] and meta["truncated"]:raise ValueError("Conflicting termination")
  if n<24 and not meta["terminated"]:raise ValueError("Short nonterminal")
  if float(tr["masks"])!=(0. if meta["terminated"] else 1.) or bool(tr["dones"])!=bool(meta["terminated"] or meta["truncated"]):raise ValueError("Bad replay mask")
  data={}
  for side in ("observations","next_observations"):
   o=tr[side]
   if set(o)!={"state","external","wrist_left","wrist_right"}:raise ValueError("Policy keys")
   if o["state"].shape!=(1,31) or not np.isfinite(o["state"]).all():raise ValueError("State")
   if not np.isin(o["state"][0,27:],[0,1]).all():raise ValueError("History bits")
   for k,a in o.items():
    if k!="state" and (a.shape!=(1,256,256,3) or a.dtype!=np.uint8):raise ValueError("RGB")
    data[side+"__"+k]=a
  for k in ("actions","rewards","masks","dones"):data[k]=np.asarray(tr[k])
  if data["actions"].shape!=(7,) or not np.isfinite(data["actions"]).all() or np.max(np.abs(data["actions"]))>1.00001:raise ValueError("Action")
  if not np.isfinite(data["rewards"]).all():raise ValueError("Reward")
  commit=self.path/(key+".json")
  if commit.exists():raise FileExistsError("Replay key already committed")
  temp=self.path/(key+".npz.tmp");blob=self.path/(key+".npz")
  with temp.open("wb") as f:
   np.savez_compressed(f,**data);f.flush();os.fsync(f.fileno())
  temp.replace(blob)
  record=dict(meta,journal_version=VERSION,key=key,npz=blob.name,sha256=digest(blob))
  pending=self.path/(key+".json.tmp")
  with pending.open("w") as f:json.dump(record,f);f.flush();os.fsync(f.fileno())
  pending.replace(commit)
  fd=os.open(self.path,os.O_RDONLY)
  try:os.fsync(fd)
  finally:os.close(fd)
  return record
 def records(self,eligible_only=True):
  for p in sorted(self.path.glob("*.json")):
   m=json.loads(p.read_text())
   if m["journal_version"]!=VERSION or m["interface_version"]!="task1_rgb_state31_reward_history_v001" or m["reward_version"]!="task1_stage_once_v001":raise ValueError("Stored version")
   b=self.path/m["npz"]
   if b.parent.resolve()!=self.path or digest(b)!=m["sha256"]:raise ValueError("Replay checksum/path")
   if eligible_only and not m["eligible"]:continue
   with np.load(b,allow_pickle=False) as f:
    tr={s:{k:f[s+"__"+k].copy() for k in ("state","external","wrist_left","wrist_right")} for s in ("observations","next_observations")}
    tr.update({k:f[k].copy() for k in ("actions","rewards","masks","dones")})
   yield tr,m
