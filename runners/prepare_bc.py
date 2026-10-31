"""Include every historical demonstration as BC pairs; never import them into SAC replay."""
import argparse,json,collections
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .common import ROOT,config,resolve
from tasks.task1_dimm_insert.env.dataset import digest
from tasks.task1_dimm_insert.env.bc_dataset import command_at_observation
from tasks.task1_dimm_insert.env import policy

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args()
    out=resolve(a.output).resolve()
    if not out.is_relative_to(ROOT/"outputs"):raise ValueError("Outputs only")
    out.mkdir(parents=True,exist_ok=False)
    cfg=config()
    historical=resolve(cfg["data_sources"]["historical_manifest"])
    hist=json.loads(historical.read_text())
    native=resolve(cfg["native_dataset"])
    nm=json.loads((native/"manifest.json").read_text())
    sources=[dict(episode=nm["episode"],index=native/"transitions.jsonl",native=True,
                  group="native",labels=None)]
    sources += [dict(episode=e["episode"],index=Path(e["episode"])/"transitions.jsonl",native=False,
                     group=e["motion_hash"],labels=Path(e["labels"])) for e in hist["episodes"]]
    manifest=dict(kind="bc_observation_command_pairs_v001",bc_only=True,sac_replay_eligible=False,
        reward_version=nm["reward_version"],interface_version=nm["interface_version"],
        source_hashes={str(historical):digest(historical)},image_hashes={},episodes=[],
        action_semantics="Recorded velocity command at observation boundary; first executed substep, never averaged",
        limitations=["Historical command duration is 50 or 100 ms; inference remains 100 ms and needs closed-loop validation.",
          "Historical history bits use existing shadow labels before the action; not relabeled as causal SAC rewards.",
          "Repeated motion groups are retained; folder count is not independent trajectory count.",
          "Temporal-block validation uses identical time bins across repeated groups; not independent generalization."])
    count=0
    with (out/"pairs.jsonl").open("x") as dst,(out/"rejections.jsonl").open("x") as rej:
      for source in sources:
        ep=Path(source["episode"]);index=source["index"]
        manifest["source_hashes"][str(index)]=digest(index)
        labels=None
        if source["labels"]:
            manifest["source_hashes"][str(source["labels"])]=digest(source["labels"])
            labels=[json.loads(x) for x in source["labels"].open()]
        stats=collections.Counter();stage_counts=collections.Counter();accepted=0;first_time=None
        for i,line in enumerate(index.open()):
            row=json.loads(line);obs=row["observations"]
            if first_time is None:first_time=float(obs["sim_time"])
            try:
                label=row["label"] if source["native"] else labels[i]
                if not source["native"] and int(label["index"])!=i:raise ValueError("Label index mismatch")
                if source["native"]:
                    action=row["actions"];duration=row["dt_s"];diag={}
                else:
                    action,duration,diag=command_at_observation(row)
                    # Check normalization against actual first integrated target when it was not clamped.
                    c=diag["controller_commands"][0];state=obs["controller_state"]
                    if not c.get("goal_lead_limited") and not c.get("soft_xy_attract",{}).get("active"):
                        expected=np.asarray(state["goal_position"])+np.asarray(action[:3])*cfg["action"]["translation_m_s"]/240
                        rot=Rotation.from_rotvec(np.asarray(action[3:6])*cfg["action"]["rotation_rad_s"]/240)*Rotation.from_quat(state["goal_quaternion_xyzw"])
                        width=np.clip(state["gripper_target"]+action[6]*cfg["action"]["gripper_m_s"]/240,0,.08)
                        if np.linalg.norm(expected-c["target_position_m"])>1e-6:raise ValueError("Command translation scale mismatch")
                        if (rot.inv()*Rotation.from_quat(c["target_quaternion_xyzw"])).magnitude()>1e-6:raise ValueError("Command rotation scale mismatch")
                        if abs(width-c["target_gripper_width_m"])>1e-6:raise ValueError("Command gripper scale mismatch")
                policy.load_observation(obs,ep,label["progress_before"])
                for info in obs["images"].values():
                    image=ep/info["file"]
                    if str(image) not in manifest["image_hashes"]:
                        manifest["image_hashes"][str(image)]=digest(image)
                stage=row.get("diagnostics",{}).get("stage","native")
                dst.write(json.dumps(dict(index=count,source_index=i,episode=str(ep),observations=obs,
                    actions=action,progress_before=label["progress_before"],command_duration_s=duration,
                    split_index=int(round((float(obs["sim_time"])-first_time)*10)),
                    motion_group=source["group"],source_stage=stage))+"\n")
                count+=1;accepted+=1;stage_counts[stage]+=1;stats[str(round(duration,4))+"s"]+=1
            except (ValueError,KeyError,OSError,IndexError) as exc:
                reason=str(exc);stats["rejected:"+reason]+=1
                rej.write(json.dumps(dict(episode=str(ep),source_index=i,reason=reason))+"\n")
        item=dict(episode=str(ep),samples=accepted,source_rows=i+1,counts=dict(stats),stage_counts=dict(stage_counts),motion_group=source["group"])
        manifest["episodes"].append(item)
        print(json.dumps(item),flush=True)
        (out/"progress.json").write_text(json.dumps(manifest["episodes"],indent=2))
    manifest.update(samples=count,index_sha256=digest(out/"pairs.jsonl"),included_directories=sum(e["samples"]>0 for e in manifest["episodes"]))
    if any(e["samples"]==0 for e in manifest["episodes"]):raise ValueError("A source has no usable samples; inspect progress")
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(samples=count,included_directories=manifest["included_directories"],output=str(out))),flush=True)
if __name__=="__main__":main()
