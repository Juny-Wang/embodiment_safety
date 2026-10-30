"""Align historical commands without inventing intermediate observations or actions."""
import json,copy
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
from .dataset import digest
from .evidence import Evidence,physics_rows
from .episode import Episode
from . import policy
def windows(rows):
 i=0
 while i<len(rows):
  r=rows[i];dt=r["next_observations"]["sim_time"]-r["observations"]["sim_time"]
  if abs(dt-.1)<1e-6:
   yield [i],r,r.get("diagnostics",{}).get("executed_substeps",[])
   i+=1
  elif abs(dt-.05)<1e-6 and i+1<len(rows):
   b=rows[i+1]
   if abs(b["observations"]["sim_time"]-r["next_observations"]["sim_time"])>1e-7:raise ValueError("Gap")
   merged=dict(r,next_observations=b["next_observations"])
   subs=[dict(start=x["observations"]["sim_time"],end=x["next_observations"]["sim_time"],action=x["actions"],diagnostics=x["diagnostics"]) for x in (r,b)]
   yield [i,i+1],merged,subs
   i+=2
  else:
   yield [i],r,[]
   i+=1
def check_window(r,subs,cfg):
 start=r["observations"]["sim_time"];end=r["next_observations"]["sim_time"]
 if abs(end-start-.1)>1e-6:return None,"not_100ms"
 if not subs:return None,"missing_substeps"
 a=np.asarray(subs[0]["action"],float)
 if a.shape!=(7,) or not np.isfinite(a).all() or np.max(np.abs(a))>1.00001:return None,"invalid_action"
 if any(not np.allclose(s["action"],a,atol=1e-9,rtol=0) for s in subs):return None,"different_substep_actions"
 t=start;commands=[]
 for s in subs:
  if abs(s["start"]-t)>1e-7:return None,"substep_gap"
  t=s["end"];commands.extend(s["diagnostics"]["controller_commands"])
 if abs(t-end)>1e-7 or len(commands)!=24:return None,"command_coverage"
 c=r["observations"]["controller_state"]
 p=np.asarray(c["goal_position"],float);q=Rotation.from_quat(c["goal_quaternion_xyzw"]);width=c["gripper_target"]
 for i,command in enumerate(commands):
  if abs(command["time"]-(start+i/240))>1e-6:return None,"command_time"
  if command.get("goal_lead_limited") or command.get("soft_xy_attract",{}).get("active"):return None,"assisted_or_clamped"
  p=p+a[:3]*cfg["translation_m_s"]/240
  q=Rotation.from_rotvec(a[3:6]*cfg["rotation_rad_s"]/240)*q
  width=float(np.clip(width+a[6]*cfg["gripper_m_s"]/240,0,.08))
  if np.linalg.norm(p-command["target_position_m"])>1e-7:return None,"position_integration"
  if (q.inv()*Rotation.from_quat(command["target_quaternion_xyzw"])).magnitude()>1e-7:return None,"rotation_integration"
  if abs(width-command["target_gripper_width_m"])>1e-7:return None,"gripper_integration"
 end_c=r["next_observations"]["controller_state"]
 if np.linalg.norm(p-end_c["goal_position"])>1e-7 or abs(width-end_c["gripper_target"])>1e-7:return None,"endpoint_target"
 if (q.inv()*Rotation.from_quat(end_c["goal_quaternion_xyzw"])).magnitude()>1e-7:return None,"endpoint_rotation"
 return a.tolist(),None
def align(cfg,root,output):
 root=Path(root);out=Path(output).resolve()
 if not out.is_relative_to(root/"outputs"):raise ValueError("Repository outputs only")
 out.mkdir(parents=True,exist_ok=False)
 base=root/cfg.get("native_dataset",cfg["dataset"]);native=json.loads((base/"manifest.json").read_text())
 historical_path=Path(cfg["data_sources"]["historical_manifest"])
 if not historical_path.exists():historical_path=Path(cfg["data_sources"]["historical_source_manifest"])
 historical=json.loads(historical_path.read_text())
 # Migrate historical labels explicitly, preserving their original evidence mode.
 shadow=out/"historical_labels";shadow.mkdir()
 for e in historical["episodes"]:
  p=Path(e["labels"])
  if digest(p)!=e["labels_sha256"]:raise ValueError("Historical label hash")
  (shadow/p.name).write_bytes(p.read_bytes())
  e["original_labels"]=str(p);e["labels"]=str(shadow/p.name)
 (shadow/"manifest.json").write_text(json.dumps(historical,indent=2))
 seen=set();summaries=[];selected=[];source_hashes=dict(native["source_hashes"]);images=dict(native["image_hashes"])
 for item in historical["episodes"]:
  ep=Path(item["episode"]);group=item["motion_hash"]
  if group in seen:
   summaries.append(dict(episode=ep.name,duplicate=True,selected=0));continue
  seen.add(group)
  for name,h in item["source_hashes"].items():
   if digest(ep/name)!=h:raise ValueError("Historical source changed")
   source_hashes[str(ep/name)]=h
  with (ep/"transitions.jsonl").open() as f:rows=[json.loads(l) for l in f]
  monitor=Evidence(rows[0]["observations"]);episode=Episode(native["socket_quaternion_xyzw"],timeout_s=cfg["timeout_s"])
  episode.reset(rows[0]["observations"])
  stream=iter(physics_rows(ep/"physics.jsonl"));pending=next(stream,None)
  counts=Counter();events=[];reward_total=0.;chosen=0
  with (out/(ep.name+".alignment.jsonl")).open("x") as audit:
   for indices,r,subs in windows(rows):
    start=r["observations"]["sim_time"];end=r["next_observations"]["sim_time"];frames=[]
    while pending is not None and pending[1]["sim_time"]<=end+1e-7:
     if pending[1]["sim_time"]>start+1e-7:frames.append(pending)
     pending=next(stream,None)
    e=monitor.update(r["next_observations"],start,frames)
    peaks=e["measurements"]["force_peaks_N"]
    label=episode.step(r["next_observations"],e,peaks)
    reward_total+=label["reward"];events.extend(dict(source_indices=indices,**x) for x in label["events"])
    action,reason=check_window(r,subs,cfg["action"])
    if not label["valid_for_training"]:reason="invalid_evidence"
    if reason is None:
     try:
      for obs,progress in ((r["observations"],label["progress_before"]),(r["next_observations"],label["progress_after"])):
       policy.load_observation(obs,ep,progress)
     except (ValueError,KeyError,OSError) as ex:reason="rgb_or_state:"+str(ex)
    if reason is None:
     for obs in (r["observations"],r["next_observations"]):
      for info in obs["images"].values():
       p=ep/info["file"];images[str(p)]=digest(p)
     selected.append(dict(index=len(selected),episode=str(ep),source_indices=indices,
      observations=r["observations"],next_observations=r["next_observations"],actions=action,
      label=label,dt_s=end-start,physics_steps=24,provenance="exact_substeps_verified_controller_targets"))
     chosen+=1
    counts[reason or "selected"]+=1
    audit.write(json.dumps(dict(source_indices=indices,reason=reason,label=label))+"\n")
    if label["terminated"] or label["truncated"]:break
  summary=dict(episode=ep.name,duplicate=False,selected=chosen,counts=dict(counts),causal_reward_total=reward_total,events=events)
  summaries.append(summary);print(json.dumps(summary),flush=True)
 with (base/"transitions.jsonl").open() as f:native_rows=[json.loads(l) for l in f]
 for r in native_rows:r["episode"]=native["episode"]
 merged=native_rows+selected
 with (out/"transitions.jsonl").open("x") as f:
  for i,r in enumerate(merged):f.write(json.dumps(dict(r,index=i))+"\n")
 m=dict(native,transitions=len(merged),index_sha256=digest(out/"transitions.jsonl"),source_hashes=source_hashes,
  image_hashes=images,rgb_files=len(images),historical_alignment=summaries,
  native_transitions=len(native_rows),historical_selected_transitions=len(selected),
  independent_training_episodes=1,training_fragment_sources=sum(s["selected"]>0 for s in summaries),
  historical=dict(imported=True,full_episodes_imported=False,selected_transitions=len(selected),deduplicated=True),
  limitations=["Only exact held100ms historical fragments imported, not reconstructed full trajectories",
   "Recorded target integration is checked; historic IK tuning is not fully recorded",
   "Stage history computed through rejected windows too; no reward reset at fragment boundaries"])
 (out/"manifest.json").write_text(json.dumps(m,indent=2))
 print(json.dumps(dict(total=len(merged),historical_selected=len(selected),output=str(out)),indent=2))
 return m
if __name__=="__main__":
 import argparse
 from runners.common import ROOT,config
 p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args()
 align(config(),ROOT,a.output)
