"""Materialize verified BC observations as contiguous Diffusion Policy sequences."""
import argparse,json,collections
from pathlib import Path
import numpy as np
from .common import ROOT,resolve
from algos.diffusion_policy.common import CAMERAS,digest
from algos.diffusion_policy.data import sequence_indices
from tasks.task1_dimm_insert.env.bc_dataset import BCDataset
def main():
 p=argparse.ArgumentParser();p.add_argument("--source",default="outputs/datasets/task1/bc_all_demos_001");p.add_argument("--output",required=True);a=p.parse_args()
 out=resolve(a.output).resolve()
 if not out.is_relative_to(ROOT/"outputs"):raise ValueError("Outputs only")
 out.mkdir(parents=True,exist_ok=False)
 data=BCDataset(resolve(a.source));rows=[];selected=[];first={};counts=collections.Counter()
 for i,r in enumerate(data.rows):
  ep=r["episode"];t=float(r["observations"]["sim_time"])
  t0=first.setdefault(ep,t);tick=round((t-t0)*10)
  if abs((t-t0)-tick/10)>1e-4:counts["off_100ms_grid"]+=1;continue
  selected.append(i);rows.append(dict(episode=ep,tick=tick,source_index=r["source_index"],stage=r["source_stage"],bc_index=i))
 n=len(rows)
 state=np.lib.format.open_memmap(out/"state.npy",mode="w+",dtype=np.float32,shape=(n,31))
 action=np.lib.format.open_memmap(out/"action.npy",mode="w+",dtype=np.float32,shape=(n,7))
 images={k:np.lib.format.open_memmap(out/(k+".npy"),mode="w+",dtype=np.uint8,shape=(n,256,256,3)) for k in CAMERAS}
 for j,i in enumerate(selected):
  item=data.get(i);state[j]=item["observations"]["state"][0];action[j]=item["actions"]
  for k in CAMERAS:images[k][j]=item["observations"][k][0]
  if j%1000==0:print(json.dumps(dict(cached=j,total=n)),flush=True)
 state.flush();action.flush()
 for x in images.values():x.flush()
 train,valid=sequence_indices([r["episode"] for r in rows],[r["tick"] for r in rows])
 if not train or not valid:raise ValueError("Empty sequence split")
 train_ids=set(i for k in train for i in range(k,k+16))
 valid_ids=set(i for k in valid for i in range(k,k+16))
 assert not train_ids&valid_ids
 meta=dict(horizon=16,n_obs_steps=2,n_action_steps=1,train=train,validation=valid,rows=n,
  episodes=dict(collections.Counter(Path(r["episode"]).name for r in rows)),counts=dict(counts),
  train_stage_coverage=dict(collections.Counter(rows[i]["stage"] for i in train_ids)),
  validation_stage_coverage=dict(collections.Counter(rows[i]["stage"] for i in valid_ids)),
  source=str(data.path),source_index_sha256=data.manifest["index_sha256"],
  source_image_hashes_verified=False,notes=["Each start is a real timestamp with real image; no cross-gap or cross-episode windows.",
  "Historical targets are first 50ms commands sampled at 100ms observation boundaries; not equivalent SAC transitions.",
  "Same time bins assigned together across duplicated motion groups; validation is temporal, not independent generalization.",
  "Original 256x256 RGB preserved. Force inputs and privileged object states not added."])
 (out/"rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
 meta["file_hashes"]={p.name:digest(p) for p in out.glob("*.npy")}
 (out/"manifest.json").write_text(json.dumps(meta,indent=2))
 print(json.dumps(dict(rows=n,train_sequences=len(train),validation_sequences=len(valid),train_stage_coverage=meta["train_stage_coverage"])),flush=True)
if __name__=="__main__":main()
