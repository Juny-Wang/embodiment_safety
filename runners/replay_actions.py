"""Replay native recorded 100ms commands in the unchanged Task1 actor."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from .common import ROOT,resolve
from .eval_diffusion import stop_group,write_json
from tasks.task1_dimm_insert.env.recorded_action_worker import validate_rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--steps",type=int,default=510)
    p.add_argument("--wall-timeout-s",type=float,default=1800)
    a=p.parse_args()
    out=resolve(a.output).resolve();source=resolve(a.source).resolve()
    if not out.is_relative_to(ROOT/"outputs") or out.exists():
        raise ValueError("New repository output directory required")
    if a.steps<1 or not 0<a.wall_timeout_s<=1800:
        raise ValueError("Invalid diagnostic budget")
    payload=source.read_bytes();rows=[json.loads(s) for s in payload.splitlines()][:a.steps]
    if len(rows)!=a.steps:raise ValueError("Insufficient actions")
    validate_rows(rows)
    env=os.environ.copy()
    env.update(TASK1_POLICY_BACKEND="recorded_action",TASK1_EVAL="1",TASK1_ONLINE_LEARN="0",
        TASK1_RECORDED_SOURCE=str(source),TASK1_RECORDED_SHA256=hashlib.sha256(payload).hexdigest(),
        TASK1_MAX_STEPS=str(a.steps),TASK1_SMOKE="0",TASK1_EPISODE_SEED="20261032")
    cmd=["/mnt/data2/isaacsim-5.0.0/python.sh","-B",str(ROOT/"tasks/task1_dimm_insert/env/actor.py"),str(out)]
    out.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    status=out.parent/(out.name+"_status.json")
    with out.with_suffix(".log").open("w") as f:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(status,dict(status="running",pid=proc.pid,steps=a.steps,wall_timeout_s=a.wall_timeout_s,
            source=str(source),output=str(out),not_a_learned_policy=True,started_at_unix=time.time()))
        try:rc=proc.wait(timeout=a.wall_timeout_s)
        except subprocess.TimeoutExpired:
            stop_group(proc)
            write_json(status,dict(status="wall_timeout",wall_s=time.monotonic()-start,not_a_learned_policy=True))
            return
    result=json.loads((out/"result.json").read_text()) if (out/"result.json").exists() else None
    errors=[n for n in ("actor_error.json","worker_error.json") if (out/n).exists()]
    ok=rc==0 and result is not None and not errors
    write_json(status,dict(status="finished" if ok else "error",returncode=rc,
        errors=errors,missing_result=result is None,wall_s=time.monotonic()-start,result=result,not_a_learned_policy=True))
    if not ok:raise RuntimeError("Recorded replay failed; inspect "+str(out))
if __name__=="__main__":
    main()
