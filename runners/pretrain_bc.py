"""Behavior cloning only: no simulator, reward changes, or SAC actor updates."""
import argparse,json,hashlib,time
from pathlib import Path
from .common import ROOT,config,resolve,setup
def main():
 p=argparse.ArgumentParser()
 p.add_argument("--output",required=True);p.add_argument("--updates",type=int,default=400)
 p.add_argument("--batch-size",type=int,default=8);p.add_argument("--learning-rate",type=float,default=1e-4)
 p.add_argument("--eval-every",type=int,default=100)
 p.add_argument("--min-updates",type=int,default=0)
 p.add_argument("--patience",type=int,default=0)
 p.add_argument("--min-delta",type=float,default=1e-5)
 a=p.parse_args();cfg=config();setup(cfg["platform"])
 import numpy as np,jax,jax.numpy as jnp,optax
 from flax import serialization
 from flax.core import freeze
 from flax.traverse_util import flatten_dict
 from algos.serl.agent import make_agent
 from tasks.task1_dimm_insert.env.dataset import Dataset
 if a.updates<1 or a.batch_size<1 or a.eval_every<1:raise ValueError("Positive budget required")
 if not 0<=a.min_updates<=a.updates or a.patience<0 or a.min_delta<0:raise ValueError("Stopping settings")
 out=resolve(a.output).resolve()
 if not out.is_relative_to(ROOT/"outputs"):raise ValueError("Remote repository outputs only")
 out.mkdir(parents=True,exist_ok=False)
 (out/"run_config.json").write_text(json.dumps(vars(a),indent=2))
 data=Dataset(resolve(cfg["dataset"]));rng=np.random.default_rng(cfg["seed"])
 # A local blocked interpolation check, NOT independent task generalization.
 train=[];valid=[];purged=[]
 for i,row in enumerate(data.rows):
  k=int(row.get("index",i))%100
  (valid if 80<=k<90 else purged if 75<=k<95 else train).append(i)
 if not valid or not train:raise ValueError("Empty split")
 split=dict(train=train,validation=valid,purged=purged,kind="temporal blocks; correlated trajectories; not independent generalization")
 (out/"split.json").write_text(json.dumps(split,indent=2))
 cache={}
 def sample(i):
  if i not in cache:
   r=data.rows[i];o=data.get(i)
   cache[i]=dict(observations=o["observations"],actions=o["actions"])
  return cache[i]
 def batch(ids):
  return freeze(jax.tree_util.tree_map(lambda *v:np.stack(v),*[sample(int(i)) for i in ids]))
 sample0=sample(train[0]);agent=make_agent(0,sample0["observations"],sample0["actions"],
  ("external","wrist_left","wrist_right"),cfg["discount"],bc_train_encoder=True)
 initial_params=agent.state.params
 groups={}
 for i in train:
  v=np.asarray(data.rows[i]["actions"])
  group="rotation" if np.linalg.norm(v[3:6])>1e-5 else "gripper" if abs(v[6])>1e-5 else "translation" if np.linalg.norm(v[:3])>1e-5 else "idle"
  groups.setdefault(group,[]).append(i)
 names=sorted(groups)
 (out/"sampling.json").write_text(json.dumps(dict(groups={k:len(v) for k,v in groups.items()},rule="50% uniform; 50% uniform over active action groups"),indent=2))
 tx=optax.adam(a.learning_rate);opt=tx.init(initial_params)
 weights=jnp.array([1/3]*3+[1/3]*3+[1.])
 @jax.jit
 def measure(params,b):
  dist=agent.forward_policy(b["observations"],grad_params=params,train=False)
  pred=dist.mode();error=pred-b["actions"]
  loss=jnp.mean(jnp.sum(error**2*weights,axis=-1))
  return loss,(pred,jnp.mean(error**2,axis=0),dist.distribution.stddev())
 @jax.jit
 def step(params,opt,b):
  (loss,aux),grads=jax.value_and_grad(measure,has_aux=True)(params,b)
  updates,newopt=tx.update(grads,opt,params)
  return optax.apply_updates(params,updates),newopt,loss
 probe=np.array(valid)
 def evaluate(params,ids):
  sums=np.zeros(7);count=0;std=[]
  for offset in range(0,len(ids),a.batch_size):
   chunk=list(ids[offset:offset+a.batch_size]);n=len(chunk)
   padded=chunk+[chunk[-1]]*(a.batch_size-n)
   loss,(pred,err,sd)=measure(params,batch(padded))
   pred=np.asarray(pred)[:n];target=np.array([data.rows[i]["actions"] for i in chunk])
   sums+=((pred-target)**2).sum(axis=0);count+=n;std.extend(np.asarray(sd)[:n].reshape(-1).tolist())
  mse=sums/count
  return dict(group_mse=dict(translation=float(mse[:3].mean()),rotation=float(mse[3:6].mean()),gripper=float(mse[6])),
   weighted_mse=float(mse[:3].mean()+mse[3:6].mean()+mse[6]),per_axis_rmse=np.sqrt(mse).tolist(),
   latent_std_p50=float(np.median(std)),latent_std_p95=float(np.quantile(std,.95)))
 before=evaluate(initial_params,probe);best=before["weighted_mse"];bestparams=initial_params;beststep=0
 params=initial_params;wall=time.monotonic()
 significant_best=best;stale=0;stop_reason="budget_exhausted";counts=np.zeros(len(data),dtype=int)
 actual_updates=0
 with (out/"metrics.jsonl").open("x") as f:
  for n in range(1,a.updates+1):
   ids=[int(rng.choice(train)) if j%2==0 else int(rng.choice(groups[names[int(rng.integers(len(names)))]])) for j in range(a.batch_size)]
   counts+=np.bincount(ids,minlength=len(data))
   params,opt,loss=step(params,opt,batch(ids));value=float(loss);actual_updates=n
   if not np.isfinite(value):raise ValueError("Nonfinite BC loss")
   record=dict(update=n,loss=value,elapsed_s=time.monotonic()-wall)
   if n%a.eval_every==0 or n==a.updates:
    metric=evaluate(params,probe);record["validation"]=metric
    if metric["weighted_mse"]<best:best=metric["weighted_mse"];bestparams=params;beststep=n
    if metric["weighted_mse"]<significant_best-a.min_delta:
     significant_best=metric["weighted_mse"];stale=0
    else:stale+=1
    record["checks_without_significant_improvement"]=stale
    record["best_update"]=beststep
    record["sampled_epoch_equivalents"]=n*a.batch_size/len(train)
   f.write(json.dumps(record)+"\n");f.flush()
   if n%a.eval_every==0:print(json.dumps(record),flush=True)
   if a.patience and n>=a.min_updates and stale>=a.patience:
    stop_reason="validation_plateau";break
 after=evaluate(bestparams,probe)
 # Transfer weights into ordinary SAC architecture; optimizer is fresh, critic is untrained.
 ordinary=make_agent(0,sample0["observations"],sample0["actions"],("external","wrist_left","wrist_right"),cfg["discount"])
 ordinary=ordinary.replace(state=ordinary.state.replace(params=bestparams,target_params=bestparams))
 b=batch(probe[:a.batch_size]);expected=agent.forward_policy(b["observations"],grad_params=bestparams,train=False).mode()
 actual=ordinary.forward_policy(b["observations"],train=False).mode()
 if not np.allclose(np.asarray(expected),np.asarray(actual),atol=1e-6):raise ValueError("BC/SAC inference mismatch")
 changed=[]
 for key,x in flatten_dict(initial_params).items():
  if not np.array_equal(np.asarray(x),np.asarray(flatten_dict(bestparams)[key])):changed.append("/".join(key))
 payload=serialization.to_bytes(ordinary.state);(out/"checkpoint.msgpack").write_bytes(payload)
 meta=dict(passed=True,updates=0,updates_this_run=0,bc_updates=actual_updates,bc_selected_update=beststep,
  reward_version=data.manifest["reward_version"],interface_version=data.manifest["interface_version"],
  dataset_index_sha256=data.manifest["index_sha256"],sampler_rng_state=rng.bit_generator.state,
  sha256=hashlib.sha256(payload).hexdigest(),demo_transitions=len(data),online_transitions=0,
  batch_size=cfg["batch_size"],utd_ratio=cfg["utd_ratio"],discount=cfg["discount"],
  scope="BC initialization; deterministic evaluation only; critic and stochastic std not calibrated",
  bc_only=True,stochastic_rollout_ready=False)
 (out/"checkpoint.json").write_text(json.dumps(meta,indent=2))
 report=dict(before=before,after=after,best_update=beststep,updates=actual_updates,wall_s=time.monotonic()-wall,
  split_counts={k:len(split[k]) for k in ("train","validation","purged")},changed_parameter_paths=changed,
  inference_transfer_equal=True,task_success_verified=False,stochastic_rollout_ready=False,
  independent_generalization_verified=False,stop_reason=stop_reason,
  sampled_epoch_equivalents=actual_updates*a.batch_size/len(train),
  unique_train_examples_seen=int(np.count_nonzero(counts)),
  validation_samples_used_for_gradients=int(counts[valid].sum()),
  train_fit=evaluate(bestparams,train))
 (out/"fit_report.json").write_text(json.dumps(report,indent=2))
 print(json.dumps({k:v for k,v in report.items() if k!="changed_parameter_paths"},indent=2))
if __name__=="__main__":main()
