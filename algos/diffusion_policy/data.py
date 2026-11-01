"""Contiguous 10 Hz sequences. Never bridge missing observations or episode boundaries."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .common import CAMERAS
from .representation import encode_state, LEGACY

def split_bin(tick):
 # Same time bins stay together even across duplicated trajectories.
 bucket=int(tick)//50
 return "validation" if bucket%8==3 else "train"

def sequence_indices(episode,ticks,horizon=16):
 trains=[];valid=[]
 for k in range(len(ticks)-horizon+1):
  ids=list(range(k,k+horizon))
  if any(episode[i]!=episode[k] or ticks[i]!=ticks[k]+i-k for i in ids):continue
  kinds={split_bin(ticks[i]) for i in ids}
  if len(kinds)!=1:continue
  (valid if kinds=={"validation"} else trains).append(k)
 return trains,valid

FIT_SCOPES = ("native-opening-fit-all", "native-full-fit-all")
SCOPES = ("all", "native-opening", *FIT_SCOPES)

def complete_native_starts(rows,horizon,tick_stop=None):
 if horizon < 1:raise ValueError("Positive horizon required")
 starts=[]
 for k in range(len(rows)-horizon+1):
  first=rows[k]
  if "policy_reset_pair_002" not in first["episode"]:continue
  if all(rows[i]["episode"]==first["episode"] and
         rows[i]["tick"]==first["tick"]+i-k and rows[i]["tick"]>=0
         and (tick_stop is None or rows[i]["tick"]<tick_stop)
         for i in range(k,k+horizon)):
   starts.append(k)
 return starts

def complete_native_opening_starts(rows,horizon):
 return complete_native_starts(rows,horizon,tick_stop=510)

class SequenceData(Dataset):
 def __init__(self,path,split,representation=LEGACY,scope="all"):
  self.path=Path(path);self.meta=json.loads((self.path/"manifest.json").read_text())
  self.starts=list(self.meta[split]);self.horizon=self.meta["horizon"]
  self.representation=representation
  self.rows=[json.loads(l) for l in (self.path/"rows.jsonl").open()]
  if scope not in SCOPES:raise ValueError("Unknown data scope")
  if scope in FIT_SCOPES:
   # Both named splits probe training samples for functional fitting.
   # Full-flow scope retains every contiguous native window, including handoffs.
   self.starts=complete_native_starts(self.rows,self.horizon,
       tick_stop=510 if scope=="native-opening-fit-all" else None)
  if scope=="native-opening":
   self.starts=[k for k in self.starts if all("policy_reset_pair_002" in self.rows[i]["episode"] and 0<=self.rows[i]["tick"]<510 for i in range(k,k+self.horizon))]
  if not self.starts:raise ValueError("Empty sequence selection")
  self.state=np.load(self.path/"state.npy",mmap_mode="r");self.action=np.load(self.path/"action.npy",mmap_mode="r")
  self.images={k:np.load(self.path/(k+".npy"),mmap_mode="r") for k in CAMERAS}
 def __len__(self):return len(self.starts)
 def __getitem__(self,i):
  k=self.starts[i]
  obs={c:torch.from_numpy(np.array(v[k:k+2],copy=True)).permute(0,3,1,2).float()/255 for c,v in self.images.items()}
  obs["state"]=torch.from_numpy(encode_state(self.state[k:k+2],self.representation))
  return dict(obs=obs,action=torch.from_numpy(np.array(self.action[k:k+self.horizon],copy=True)))
