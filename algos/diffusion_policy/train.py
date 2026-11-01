"""Official DP BC with versioned state preprocessing and explicit diagnostic coverage."""
import argparse, json, time, copy, random
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .common import ROOT, DEFAULT, REVISION, CAMERAS, build, digest, prediction_type
from .data import SequenceData, SCOPES, FIT_SCOPES
from .representation import encode_state, state_config, LEGACY, ROT6D
from .diagnostics import balanced_ids, coverage, label, physical_metrics
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True);p.add_argument("--output",required=True)
    p.add_argument("--steps",type=int,default=20000);p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--eval-every",type=int,default=500);p.add_argument("--resume")
    p.add_argument("--state-representation",choices=[LEGACY,ROT6D],default=LEGACY)
    p.add_argument("--scope",choices=SCOPES,default="all")
    p.add_argument("--prediction-type",choices=["epsilon","sample"],default="epsilon")
    p.add_argument("--action-range-eps",type=float,default=1e-4)
    a=p.parse_args()
    if not 0 < a.action_range_eps <= 1e-4:raise ValueError("action-range-eps must be in (0,1e-4]")
    if min(a.steps,a.batch_size,a.eval_every)<1:raise ValueError("Positive counts required")
    out=Path(a.output).resolve()
    if not out.is_relative_to(ROOT/"outputs"):raise ValueError("Outputs only")
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    contract=state_config(a.state_representation)
    data=SequenceData(a.data,"train",a.state_representation,a.scope)
    valid=SequenceData(a.data,"validation",a.state_representation,a.scope)
    for name,h in data.meta["file_hashes"].items():
        if digest(Path(a.data)/name)!=h:raise ValueError("Dataset changed: "+name)
    manifest_hash=digest(Path(a.data)/"manifest.json")
    rows_hash=digest(Path(a.data)/"rows.jsonl")
    val_ids=balanced_ids(valid);probe_ids=balanced_ids(data)
    cfg={**DEFAULT,**contract,"official_revision":REVISION,"steps":a.steps,
         "prediction_type":a.prediction_type,"action_range_eps":a.action_range_eps,"batch_size":a.batch_size,"data":str(Path(a.data).resolve()),"scope":a.scope,
         "dataset_manifest_sha256":manifest_hash,"rows_sha256":rows_hash,
         "validation_subset":val_ids,"fit_probe_subset":probe_ids,
         "early_stopping":False,"control_hz":10,
         "validation_scope":("in-sample functional fit; no held-out data" if a.scope in FIT_SCOPES else "correlated held-out temporal blocks; not independent generalization"),
         "fit_probe_scope":"in-sample reconstruction, not validation or physical success",
         "rotation_normalization":"identity_minus1_plus1" if a.state_representation==ROT6D else "legacy_fitted"}
    cp=None
    if a.resume:
        cp=torch.load(a.resume,map_location="cpu",weights_only=False)
        if cp["config"].get("action_range_eps",1e-4)!=a.action_range_eps:raise ValueError("Cannot resume across action normalization contracts")
        if prediction_type(cp["config"])!=a.prediction_type:raise ValueError("Cannot resume across different diffusion training targets")
        for key in ("state_representation","state_dim","scope","data","dataset_manifest_sha256","rows_sha256"):
            if cp["config"].get(key)!=cfg[key]:raise ValueError("Resume contract mismatch: "+key)
    out.mkdir(parents=True,exist_ok=bool(a.resume))
    model=build(cfg)
    ids=sorted(set(i for k in data.starts for i in range(k,k+data.horizon)))
    norm=LinearNormalizer()
    norm.fit(dict(action=np.asarray(data.action[ids]),
                  state=encode_state(data.state[ids],a.state_representation)),mode="limits")
    # Keep state normalization unchanged. Preserve small but nonconstant action
    # ranges using the official configurable normalizer, not execution clamps.
    norm["action"]=SingleFieldLinearNormalizer.create_fit(
        np.asarray(data.action[ids]),mode="limits",range_eps=a.action_range_eps)
    if a.state_representation==ROT6D:
        # Rotation matrix entries are inherently bounded [-1,1].
        with torch.no_grad():
            norm.params_dict["state"]["scale"][22:28]=1.
            norm.params_dict["state"]["offset"][22:28]=0.
    for k in CAMERAS:norm[k]=SingleFieldLinearNormalizer.create_identity()
    model.set_normalizer(norm)
    # Official normalizer load reconstructs Parameters on the checkpoint device.
    # Restore before cuda()/optimizer creation so normalizer and network agree.
    if cp:model.load_state_dict(cp["model"])
    model=model.cuda();ema=copy.deepcopy(model).eval()
    if cp:
        ema.load_state_dict(cp["ema"])
        ema=ema.cuda()
    if any(p.device.type!="cuda" for obj in (model,ema) for p in obj.parameters()):
        raise RuntimeError("Policy and normalizer must both be on CUDA")
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-6)
    loader=DataLoader(data,batch_size=a.batch_size,shuffle=True,num_workers=2,pin_memory=True,persistent_workers=True)
    def device(b):
        if isinstance(b,dict):return {k:device(v) for k,v in b.items()}
        return b.cuda(non_blocking=True)
    @torch.no_grad()
    def evaluate(dataset, chosen):
        ema.eval();loss=[];predictions=[];targets=[]
        evalloader=DataLoader(Subset(dataset,chosen),batch_size=a.batch_size,num_workers=0)
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
            for b in evalloader:
                b=device(b);loss.append(float(ema.compute_loss(b)))
                predictions.append(ema.predict_action(b["obs"])["action"][:,0].cpu().numpy())
                targets.append(b["action"][:,1].cpu().numpy())
        pred=np.concatenate(predictions);target=np.concatenate(targets)
        groups=defaultdict(list)
        for j,i in enumerate(chosen):groups[label(dataset.rows[dataset.starts[i]+1])].append(j)
        return dict(**physical_metrics(pred,target),diffusion_loss=float(np.mean(loss)),
                    per_axis_rmse=np.sqrt(np.mean((pred-target)**2,axis=0)).tolist(),
                    by_source_stage_or_time={k:physical_metrics(pred[v],target[v]) for k,v in groups.items()})
    train_cov=coverage(data,list(range(len(data))))
    val_cov=coverage(valid,list(range(len(valid))))
    report=dict(train_sequences=len(data),validation_sequences=len(valid),train_coverage=train_cov,
                validation_coverage=val_cov,fit_probe_coverage=coverage(data,probe_ids),
                held_out_missing_groups=sorted(set(train_cov)-set(val_cov)),
                independent_trajectory_generalization=False,
                probe_is_in_sample=a.scope in FIT_SCOPES,
                native_time_bins_are_not_stage_labels=True,
                native_source="verified 24 physics steps / 100ms",historical_source_used=a.scope=="all")
    (out/"coverage.json").write_text(json.dumps(report,indent=2))
    (out/"config.json").write_text(json.dumps(cfg,indent=2))
    sources=[Path(__file__),Path(__file__).with_name("representation.py"),Path(__file__).with_name("data.py"),
             Path(__file__).with_name("common.py"),Path(__file__).with_name("worker.py"),
             Path(__file__).with_name("diagnostics.py")]
    (out/"source_hashes.json").write_text(json.dumps({str(p.relative_to(ROOT)):digest(p) for p in sources},indent=2))
    def save(name,step,best):
        payload=dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                     step=step,best=best,config=cfg,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        tmp=out/(name+".tmp");torch.save(payload,tmp);tmp.replace(out/(name+".pt"))
        meta=dict(algorithm="diffusion_policy",bc_only=True,network_updates=0,training_steps=step,
                  sha256=digest(out/(name+".pt")),official_revision=REVISION,
                  interface_version=contract["interface_version"],reward_version="task1_stage_once_v001",
                  task_success_verified=False,config=cfg)
        (out/(name+".json")).write_text(json.dumps(meta,indent=2))
    step=0;best=float("inf")
    if cp:
        optimizer.load_state_dict(cp["optimizer"])
        step=cp["step"];best=cp["best"];torch.set_rng_state(cp["torch_rng"]);torch.cuda.set_rng_state_all(cp["cuda_rng"])
    iterator=iter(loader);wall=time.monotonic()
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
                metric=evaluate(valid,val_ids);rec["validation"]=metric
                rec["in_sample_fit_probe"]=evaluate(data,probe_ids)
                if metric["action_mse"]<best:best=metric["action_mse"];save("best",step,best)
                save("last",step,best)
            log.write(json.dumps(rec)+"\n");log.flush()
            if step%100==0 or step==a.steps:print(json.dumps(rec),flush=True)
    (out/"result.json").write_text(json.dumps(dict(status="finished",steps=step,best_action_mse=best,
        wall_s=time.monotonic()-wall,task_success_verified=False,scope=a.scope,**contract),indent=2))

if __name__=="__main__":main()
