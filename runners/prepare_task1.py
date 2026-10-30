"""Prepare the current Task1 data and import only eligible online transitions."""
import argparse,json,shutil
from .common import ROOT,config,resolve
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/experiments/task1_serl.json");a=p.parse_args()
 c=config(a.config)
 from tasks.task1_dimm_insert.env.dataset import prepare
 from algos.serl.replay import Journal
 native=dict(c,dataset=c.get("native_dataset",c["dataset"]),data_sources=dict(c["data_sources"]))
 native["data_sources"]["historical_manifest"]=c["data_sources"].get("historical_source_manifest",c["data_sources"]["historical_manifest"])
 m=prepare(native,ROOT)
 if native["dataset"]!=c["dataset"]:
  from tasks.task1_dimm_insert.env.align_history import align
  m=align(c,ROOT,resolve(c["dataset"]))
 # Existing source journal is read-only; only committed, eligible records are copied.
 source=Path(c["data_sources"]["prior_online"]);out=resolve(c["replay"]);out.mkdir(parents=True,exist_ok=False)
 count=0
 for meta_path in sorted(source.glob("*.json")):
  meta=json.loads(meta_path.read_text())
  if not meta.get("eligible"):continue
  blob=source/meta["npz"]
  from tasks.task1_dimm_insert.env.dataset import digest
  if blob.parent.resolve()!=source.resolve() or digest(blob)!=meta["sha256"]:raise ValueError("Replay hash/path")
  shutil.copy2(blob,out/blob.name);shutil.copy2(meta_path,out/meta_path.name);count+=1
 assert len(list(Journal(out).records()))==count
 summary=dict(transitions=m["transitions"],demo_return=m["return_total"],eligible_online=count,
  dataset=str(resolve(c["dataset"])),replay=str(out),raw_data_modified=False)
 (ROOT/"tasks/task1_dimm_insert/configs/data_manifest.json").write_text(json.dumps(summary,indent=2))
 print(json.dumps(summary,indent=2))
from pathlib import Path
if __name__=="__main__":main()
