"""Pure Python episode writer; no Isaac import or robot side effects."""
import json, math, hashlib, uuid
from datetime import datetime, timezone
from pathlib import Path

def clean(x):
    if hasattr(x, "tolist"): return clean(x.tolist())
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [clean(v) for v in x]
    if isinstance(x,float) and not math.isfinite(x): raise ValueError("Non-finite sample")
    return x

def dump(p,x):
    tmp=p.with_suffix(p.suffix+".tmp")
    tmp.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False))
    tmp.replace(p)

def verify_manifest(path):
    m=json.loads(Path(path).read_text());bad=[]
    for f,h in m["files"].items():
        p=Path(f)
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h: bad.append(f)
    if bad: raise RuntimeError("Frozen source changed/missing: "+repr(bad))
    return m

class Episode:
    def __init__(self,root,metadata):
        self.path=Path(root)/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:8])
        self.path.mkdir(parents=True,exist_ok=False)
        (self.path/"images").mkdir()
        self.meta=dict(metadata,status="recording",schema_version=1)
        self.rows=self.path.joinpath("transitions.jsonl").open("x",buffering=1)
        self.events=self.path.joinpath("physics.jsonl").open("x",buffering=1)
        self.count=0;self.previous=None;self.closed=False
        dump(self.path/"metadata.json",self.meta)
    def physics(self,frame):
        if not self.closed:self.events.write(json.dumps(clean(frame),allow_nan=False)+"\n")
    def append(self,obs,action,next_obs,diagnostic):
        if self.closed:raise RuntimeError("Episode closed")
        t=float(obs["sim_time"]);nt=float(next_obs["sim_time"])
        if nt<=t:raise ValueError("Non-positive transition interval")
        if self.previous is not None and abs(t-self.previous)>1e-7:raise ValueError("Transition gap")
        if len(action)!=7 or any(not math.isfinite(float(a)) or abs(a)>1.00001 for a in action):
            raise ValueError("Action outside 7D normalized contract")
        row=clean(dict(index=self.count,observations=obs,actions=action,next_observations=next_obs,
                       diagnostics=diagnostic))
        self.rows.write(json.dumps(row,allow_nan=False)+"\n");self.count+=1;self.previous=nt
    def finish(self,outcome,reason="",checks=None):
        if self.closed:return self.path
        self.rows.close();self.events.close();self.closed=True
        checks=checks or {}
        # Human success request is never equivalent to mechanical acceptance.
        task_success=bool(outcome=="success_request" and checks.get("mechanical_success") is True)
        self.meta.update(status="closed",outcome=outcome,reason=reason,transitions=self.count,
                         task_success=task_success,safety_pass=checks.get("safety_pass"),
                         data_valid=None,checks=checks)
        dump(self.path/"metadata.json",self.meta)
        return self.path
