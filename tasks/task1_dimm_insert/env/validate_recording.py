"""python -m tasks.task1_dimm_insert.env.validate_recording EPISODE_DIRECTORY"""
import argparse,json,math
from pathlib import Path
from tasks.task1_dimm_insert.env.data import dump
def validate(path,write=False):
    p=Path(path);meta=json.loads((p/"metadata.json").read_text());errors=[];n=0;last=None;stale=0
    image_checks={};image_failures=[]
    from tasks.task1_dimm_insert.env.image_quality import inspect_file
    try:
        for line in (p/"transitions.jsonl").read_text().splitlines():
            r=json.loads(line);o=r["observations"];nxt=r["next_observations"];a=r["actions"]
            t=o["sim_time"];nt=nxt["sim_time"]
            if nt<=t or (last is not None and abs(t-last)>1e-7):errors.append("non_contiguous_time")
            if abs((nt-t)-meta["control_dt"])>meta["physics_dt"]*1.1:errors.append("control_interval")
            if len(a)!=7 or any(not math.isfinite(x) or abs(x)>1.00001 for x in a):errors.append("action_contract")
            for obs in [o,nxt]:
                if not obs.get("sensor_valid"):errors.append("sensor_invalid")
                for key in meta["required_cameras"]:
                    image=obs.get("images",{}).get(key)
                    if not image:errors.append("missing_camera:"+key);continue
                    f=p/image["file"]
                    if not f.is_file():errors.append("missing_image_file")
                    else:
                        if str(f) not in image_checks:
                            image_checks[str(f)]=inspect_file(f)
                            if not image_checks[str(f)]["valid"]:
                                image_failures.append(dict(camera=key,file=str(f),**image_checks[str(f)]))
                        health=image_checks[str(f)]
                        if not health["valid"]:errors.append("image_quality:"+key+":"+health["reason"])
                    age=obs["sim_time"]-image["sim_time"]
                    if age < -meta["physics_dt"] or age>meta["max_image_age_s"]:errors.append("image_age");stale+=1
                for field in ["joint_positions","joint_velocities","ee_position_m","ee_quaternion_xyzw"]:
                    if not all(math.isfinite(x) for x in obs[field]):errors.append("nonfinite_state")
            last=nt;n+=1
    except Exception as e:errors.append("parse_error:"+repr(e))
    if n==0:errors.append("empty_episode")
    if meta.get("status")!="closed":errors.append("unclosed_episode")
    if n!=meta.get("transitions"):errors.append("transition_count")
    errors=sorted(set(errors))
    result={"data_valid":not errors,"errors":errors,"transitions":n,"stale_image_checks":stale,
            "task_success":meta.get("task_success") is True,"safety_pass":meta.get("safety_pass"),
            "eligible_safe_demo":not errors and meta.get("task_success") is True and meta.get("safety_pass") is True}
    result["image_quality"]={"checked_files":len(image_checks),"failed_files":len(image_failures),"examples":image_failures[:9]}
    if write:
        dump(p/"validation.json",result);meta["data_valid"]=not errors;dump(p/"metadata.json",meta)
    return result
def main():
    parser=argparse.ArgumentParser();parser.add_argument("episode");parser.add_argument("--write",action="store_true");args=parser.parse_args()
    r=validate(args.episode,args.write);print(json.dumps(r,indent=2));return 0 if r["data_valid"] else 1
if __name__=="__main__":raise SystemExit(main())
