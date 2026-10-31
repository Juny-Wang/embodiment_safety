"""Official DiffusionUnetImagePolicy BC with EMA, resumable checkpoints and GPU training."""
import argparse,json,time,copy,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from .common import ROOT,DEFAULT,REVISION,CAMERAS,build,digest
from .data import SequenceData
from diffusion_policy.model.common.normalizer import LinearNormalizer,SingleFieldLinearNormalizer

def main():
 p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--output",required=True)
 p.add_argument("--steps",type=int,default=20000);p.add_argument("--batch-size",type=int,default=8)
 p.add_argument("--eval-every",type=int,default=500);p.add_argument("--resume")
 a=p.parse_args();out=Path(a.output).resolve()
 if not out.is_relative_to(ROOT/"outputs"):raise ValueError("Outputs only")
 out.mkdir(parents=True,exist_ok=bool(a.resume))
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
 if not torch.cuda.is_available():raise RuntimeError("CUDA required")
 data=SequenceData(a.data,"train");valid=SequenceData(a.data,"validation")
 for name,h in data.meta["file_hashes"].items():
  if digest(Path(a.data)/name)!=h:raise ValueError("Dataset changed")
 model=build(DEFAULT)
 ids=sorted(set(i for k in data.starts for i in range(k,k+data.horizon)))
 norm=LinearNormalizer();norm.fit(dict(action=np.asarray(data.action[ids]),state=np.asarray(data.state[ids])),mode="limits")
 for k in CAMERAS:norm[k]=SingleFieldLinearNormalizer.create_identity()
 model.set_normalizer(norm);model=model.cuda();ema=copy.deepcopy(model).eval()
 optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-6)
 loader=DataLoader(data,batch_size=a.batch_size,shuffle=True,num_workers=2,pin_memory=True,persistent_workers=True)
 # Fixed subset across time; same diffusion RNG for comparable validation.
 val_ids=np.linspace(0,len(valid)-1,min(128,len(valid)),dtype=int).tolist()
 vloader=DataLoader(Subset(valid,val_ids),batch_size=a.batch_size,num_workers=0)
 def device(b):
  if isinstance(b,dict):return {k:device(v) for k,v in b.items()}
  return b.cuda(non_blocking=True)
 @torch.no_grad()
 def evaluate():
  ema.eval();loss=[];errors=[]
  with torch.random.fork_rng(devices=[0]):
   torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
   for b in vloader:
    b=device(b);loss.append(float(ema.compute_loss(b)))
    pred=ema.predict_action(b["obs"])["action"][:,0]
    errors.append((pred-b["action"][:,1]).square().cpu())
  mse=torch.cat(errors).mean(0)
  return dict(diffusion_loss=float(np.mean(loss)),action_mse=float(mse.mean()),per_axis_rmse=mse.sqrt().tolist())
 cfg=dict(**DEFAULT,official_revision=REVISION,steps=a.steps,batch_size=a.batch_size,data=str(Path(a.data).resolve()),
          validation_subset=val_ids,early_stopping=False,control_hz=10)
 (out/"config.json").write_text(json.dumps(cfg,indent=2))
 def save(name,step,best):
  payload=dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),step=step,best=best,config=cfg,
   torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
  tmp=out/(name+".tmp");torch.save(payload,tmp);tmp.replace(out/(name+".pt"))
  meta=dict(algorithm="diffusion_policy",bc_only=True,network_updates=0,training_steps=step,sha256=digest(out/(name+".pt")),
   official_revision=REVISION,interface_version="task1_rgb_state31_reward_history_v001",reward_version="task1_stage_once_v001",
   task_success_verified=False,config=cfg)
  (out/(name+".json")).write_text(json.dumps(meta,indent=2))
 step=0;best=float("inf");iterator=iter(loader)
 if a.resume:
  cp=torch.load(a.resume,map_location="cpu",weights_only=False)
  model.load_state_dict(cp["model"]);ema.load_state_dict(cp["ema"]);optimizer.load_state_dict(cp["optimizer"])
  step=cp["step"];best=cp["best"];torch.set_rng_state(cp["torch_rng"]);torch.cuda.set_rng_state_all(cp["cuda_rng"])
 wall=time.monotonic()
 with (out/"metrics.jsonl").open("a" if a.resume else "x") as log:
  while step<a.steps:
   try:b=next(iterator)
   except StopIteration:iterator=iter(loader);b=next(iterator)
   b=device(b);model.train();optimizer.zero_grad(set_to_none=True)
   loss=model.compute_loss(b)
   if not torch.isfinite(loss):raise ValueError("Nonfinite diffusion loss")
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();step+=1
   decay=min(.9999,1-(1+step)**(-.75))
   with torch.no_grad():
    for pe,pm in zip(ema.parameters(),model.parameters()):pe.lerp_(pm,1-decay)
    for be,bm in zip(ema.buffers(),model.buffers()):be.copy_(bm)
   rec=dict(step=step,loss=float(loss),elapsed_s=time.monotonic()-wall)
   if step%a.eval_every==0 or step==a.steps:
    metric=evaluate();rec["validation"]=metric
    if metric["action_mse"]<best:best=metric["action_mse"];save("best",step,best)
    save("last",step,best)
   log.write(json.dumps(rec)+"\n");log.flush()
   if step%100==0:print(json.dumps(rec),flush=True)
 (out/"result.json").write_text(json.dumps(dict(status="finished",steps=step,best_action_mse=best,
  wall_s=time.monotonic()-wall,task_success_verified=False,official_revision=REVISION),indent=2))
if __name__=="__main__":main()
