"""Wall-time bounded online SERL episodes and an isolated final evaluation."""
import argparse,collections,hashlib,json,os,signal,subprocess,sys,time
from datetime import datetime,timezone
from .common import ROOT,resolve
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True)
 p.add_argument("--seconds",type=int,default=3600);p.add_argument("--evaluation-reserve",type=int,default=600)
 p.add_argument("--episode-wall-seconds",type=int,default=600);p.add_argument("--seed",type=int,default=20261031)
 a=p.parse_args()
 if not 60<=a.evaluation_reserve<a.seconds or a.episode_wall_seconds<30:raise ValueError("Budget")
 out=resolve(a.output).resolve()
 if not out.is_relative_to(ROOT/"outputs"):raise ValueError("Repository outputs only")
 out.mkdir(parents=True,exist_ok=False)
 checkpoint=resolve(a.checkpoint).resolve()
 initial=json.loads(checkpoint.with_suffix(".json").read_text())
 started=time.monotonic();deadline=started+a.seconds;rows=[]
 report=dict(started_utc=datetime.now(timezone.utc).isoformat(),status="running",budget_s=a.seconds,
  initial_checkpoint=str(checkpoint),initial_updates=initial["updates"],episodes=rows,
  reward_version=initial["reward_version"],interface_version=initial["interface_version"])
 def save():
  report["elapsed_s"]=time.monotonic()-started;report["latest_checkpoint"]=str(checkpoint)
  report["network_updates"]=sum(x.get("network_updates",0) for x in rows if x["mode"]=="train")
  report["new_training_transitions"]=sum(x.get("new_transitions",0) for x in rows if x["mode"]=="train")
  report["completed_training_episodes"]=sum(x.get("completed",False) for x in rows if x["mode"]=="train")
  report["failure_counts"]=dict(collections.Counter(x.get("failure_reason") or "no_failure" for x in rows if x.get("completed")))
  q=out/"summary.tmp";q.write_text(json.dumps(report,indent=2));q.replace(out/"summary.json")
 def run(mode,index,budget):
  nonlocal checkpoint
  dest=out/(f"episode_{index:03d}" if mode=="train" else "evaluation")
  cmd=[sys.executable,"-B","-m","runners.rollout" if mode=="train" else "runners.eval",
   "--checkpoint",str(checkpoint),"--output",str(dest),"--seed",str(a.seed+index)]
  if mode=="train":cmd.append("--learn")
  row=dict(mode=mode,index=index,seed=a.seed+index,output=str(dest),input_checkpoint=str(checkpoint),completed=False)
  rows.append(row);report["active_output"]=str(dest);save()
  wall=time.monotonic()
  env=os.environ.copy();env.pop("TASK1_EVAL",None)
  with (out/(dest.name+".log")).open("w") as log:
   proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:row["returncode"]=proc.wait(timeout=budget)
   except subprocess.TimeoutExpired:
    row["error"]="wall_budget_exceeded"
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
     os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    row["returncode"]=proc.returncode
  row["wall_s"]=time.monotonic()-wall
  if row["returncode"]==0 and (dest/"result.json").exists() and (dest/"worker_result.json").exists():
   result=json.loads((dest/"result.json").read_text());worker=json.loads((dest/"worker_result.json").read_text())
   labels=[json.loads(x.read_text()) for x in sorted(dest.glob("stage_transition_*.json"))]
   row.update(completed=True,full_task_success=result["full_task_success"],failure_reason=result["failure_reason"],
    source_unchanged=result["source_unchanged"],new_transitions=worker["new_transitions"],
    network_updates=worker["network_updates"],reward=sum(x["reward"] for x in labels),
    stages=[e["stage"] for x in labels for e in x["events"]],
    full_transitions=result["full_transitions"])
   cp=dest/"checkpoint.msgpack";meta=json.loads(cp.with_suffix(".json").read_text())
   if hashlib.sha256(cp.read_bytes()).hexdigest()!=meta["sha256"] or not result["source_unchanged"]:
    raise RuntimeError("Checkpoint/source integrity")
   if mode=="train":checkpoint=cp
  else:
   for name in ("actor_error.json","worker_error.json"):
    if (dest/name).exists():row[name]=json.loads((dest/name).read_text())
  save();return row["completed"]
 save()
 try:
  i=0
  while deadline-time.monotonic()>a.evaluation_reserve+120:
   i+=1
   if not run("train",i,min(a.episode_wall_seconds,deadline-time.monotonic()-a.evaluation_reserve)):
    report["training_stopped_on_infrastructure_error"]=True;break
  if deadline-time.monotonic()>60:
   run("eval",i+1,deadline-time.monotonic())
  report["status"]="finished"
  evaluation=[x for x in rows if x["mode"]=="eval"]
  report["evaluation_success"]=bool(evaluation and evaluation[-1].get("full_task_success"))
  report["all_completed"]=all(x["completed"] for x in rows)
 except BaseException as exc:
  report["status"]="error";report["error"]=repr(exc);raise
 finally:save()
 print(json.dumps(report,indent=2))
if __name__=="__main__":main()
