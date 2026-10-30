"""Prepare causal, held-action demonstrations without rewriting original recordings."""
import json,hashlib
from pathlib import Path
import numpy as np
from .evidence import Evidence,physics_rows
from .episode import Episode
from . import policy
def digest(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for x in iter(lambda:f.read(1048576),b""):h.update(x)
 return h.hexdigest()
class Dataset:
 def __init__(self,path,verify=True):
  self.path=Path(path)
  self.manifest=json.loads((self.path/"manifest.json").read_text())
  if self.manifest["reward_version"]!="task1_stage_once_v001":raise ValueError("Reward version")
  if self.manifest["interface_version"]!="task1_rgb_state31_reward_history_v001":raise ValueError("Interface")
  if verify:
   for p,h in self.manifest["source_hashes"].items():
    if digest(p)!=h:raise ValueError("Dataset source changed: "+p)
   if digest(self.path/"transitions.jsonl")!=self.manifest["index_sha256"]:raise ValueError("Dataset index changed")
  with (self.path/"transitions.jsonl").open() as f:self.rows=[json.loads(l) for l in f]
  self.episode=Path(self.manifest["episode"])
 def __len__(self):return len(self.rows)
 def get(self,i):
  r=self.rows[i];l=r["label"]
  return dict(observations=policy.load_observation(r["observations"],self.episode,l["progress_before"]),
   next_observations=policy.load_observation(r["next_observations"],self.episode,l["progress_after"]),
   actions=np.asarray(r["actions"],np.float32),rewards=np.float32(l["reward"]),
   masks=np.float32(l["mask"]),dones=bool(l["terminated"] or l["truncated"]))
def prepare(cfg,root):
 from pxr import Usd,UsdGeom
 root=Path(root);out=root/cfg["dataset"];out.mkdir(parents=True,exist_ok=False)
 ep=Path(cfg["data_sources"]["native_episode"])
 acceptance=json.loads((ep/"acceptance.json").read_text())
 if acceptance.get("accepted") is not True:raise ValueError("Unaccepted source")
 stage=Usd.Stage.Open(str(root/cfg["scene"]))
 prim=stage.GetPrimAtPath("/World/MatchedSocket/Asset")
 if not prim:raise ValueError("Socket frame missing")
 q=UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractRotationQuat().GetNormalized()
 quat=[*q.GetImaginary(),q.GetReal()]
 with (ep/"transitions.jsonl").open() as f:rows=[json.loads(l) for l in f]
 monitor=Evidence(rows[0]["observations"]);episode=Episode(quat,timeout_s=cfg["timeout_s"]);episode.reset(rows[0]["observations"])
 stream=iter(physics_rows(ep/"physics.jsonl"));pending=next(stream,None)
 events=[];count=0;total=0.;image_hashes={}
 with (out/"transitions.jsonl").open("x") as f:
  for i,r in enumerate(rows):
   start=r["observations"]["sim_time"];o=r["next_observations"];end=o["sim_time"]
   commands=r["diagnostics"]["controller_commands"]
   if abs(end-start-.1)>1e-6 or len(commands)!=24:raise ValueError("Not a 100ms action")
   for command in commands:
    if not np.allclose(command["integration_action"],r["actions"],atol=1e-9,rtol=0):raise ValueError("Not a held action")
    if abs(command["integration_dt_s"]-1/240)>1e-6:raise ValueError("Physics dt mismatch")
   frames=[]
   while pending is not None and pending[1]["sim_time"]<=end+1e-7:
    if pending[1]["sim_time"]>start+1e-7:frames.append(pending)
    pending=next(stream,None)
   measured=monitor.update(o,start,frames)
   label=episode.step(o,measured,r["diagnostics"]["contact_peaks"])
   if not label["valid_for_training"]:raise ValueError("Invalid training evidence")
   # Validate every RGB, not just a first/last image.
   for obs,progress in ((r["observations"],label["progress_before"]),(o,label["progress_after"])):
    policy.load_observation(obs,ep,progress)
    for info in obs["images"].values():
     path=ep/info["file"]
     image_hashes.setdefault(str(path),digest(path))
   f.write(json.dumps(dict(index=i,observations=r["observations"],actions=r["actions"],
    next_observations=o,label=label,dt_s=end-start,physics_steps=24))+"\n")
   count+=1;total+=label["reward"];events.extend(dict(index=i,**e) for e in label["events"])
   if label["terminated"] or label["truncated"]:break
 if not label["success"] or abs(total-1.4)>1e-8:raise ValueError("Causal demo did not complete")
 h={str(ep/n):digest(ep/n) for n in ("transitions.jsonl","physics.jsonl","acceptance.json")}
 historical=json.loads(Path(cfg["data_sources"]["historical_manifest"]).read_text())
 manifest=dict(reward_version=label["reward_version"],interface_version=label["interface_version"],
  lifecycle_version=label["lifecycle_version"],episode=str(ep),transitions=count,source_rows=len(rows),
  ignored_post_terminal_rows=len(rows)-count,return_total=total,events=events,socket_quaternion_xyzw=quat,
  source_hashes=h,index_sha256=digest(out/"transitions.jsonl"),rgb_files=len(image_hashes),
  image_hashes=image_hashes,independent_training_episodes=1,evaluation_episodes=0,
  historical=dict(episodes=len(historical["episodes"]),motion_group_sizes=[len(g) for g in historical["motion_groups"]],
   imported=False,reason="Old averaged substep execution differs from held-100ms contract; keep raw provenance."),
  scene=str(root/cfg["scene"]),scene_sha256=digest(root/cfg["scene"]))
 (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
 print(json.dumps({k:v for k,v in manifest.items() if k not in ("image_hashes","source_hashes")},indent=2))
 return manifest
