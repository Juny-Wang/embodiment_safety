"""Contiguous 10 Hz sequences. Never bridge missing observations or episode boundaries."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .common import CAMERAS

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

class SequenceData(Dataset):
 def __init__(self,path,split):
  self.path=Path(path);self.meta=json.loads((self.path/"manifest.json").read_text())
  self.starts=self.meta[split];self.horizon=self.meta["horizon"]
  self.state=np.load(self.path/"state.npy",mmap_mode="r");self.action=np.load(self.path/"action.npy",mmap_mode="r")
  self.images={k:np.load(self.path/(k+".npy"),mmap_mode="r") for k in CAMERAS}
 def __len__(self):return len(self.starts)
 def __getitem__(self,i):
  k=self.starts[i]
  obs={c:torch.from_numpy(np.array(v[k:k+2],copy=True)).permute(0,3,1,2).float()/255 for c,v in self.images.items()}
  obs["state"]=torch.from_numpy(np.array(self.state[k:k+2],copy=True))
  return dict(obs=obs,action=torch.from_numpy(np.array(self.action[k:k+self.horizon],copy=True)))
