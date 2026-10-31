"""Policy inference and optional online mixed updates; no simulator imports."""
import os,sys,time,json,traceback,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from runners.common import config,resolve,setup
cfg=config(os.environ.get("TASK1_CONFIG","configs/experiments/task1_serl.json"));setup(cfg["platform"])
import numpy as np,jax
from flax import serialization
from flax.core import freeze
from algos.serl.agent import make_agent
from algos.serl.inference import sample_policy
from algos.serl.replay import Journal
from tasks.task1_dimm_insert.env.dataset import Dataset
OUT=Path(sys.argv[1]);SMOKE=os.environ.get("TASK1_SMOKE")=="1"
LEARN=os.environ.get("TASK1_ONLINE_LEARN")=="1"
def write(name,obj):
 p=OUT/name;tmp=p.with_suffix(".tmp");tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def obs(i):
 with np.load(OUT/f"obs_{i:03d}.npz",allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
try:
 dataset=Dataset(resolve(cfg["dataset"]));journal=Journal(os.environ["SERL_REPLAY_DIR"])
 online=list(journal.records());initial_online=len(online)
 checkpoint=Path(os.environ["SERL_INPUT_CHECKPOINT"]);meta=json.loads(checkpoint.with_suffix(".json").read_text())
 if meta.get("bc_only") and (LEARN or os.environ.get("TASK1_DETERMINISTIC")!="1"):
  raise ValueError("BC checkpoint requires deterministic evaluation; critic/std calibration pending")
 payload=checkpoint.read_bytes()
 if hashlib.sha256(payload).hexdigest()!=meta["sha256"]:raise ValueError("Checkpoint hash")
 if meta["interface_version"]!=dataset.manifest["interface_version"] or meta["reward_version"]!=dataset.manifest["reward_version"]:raise ValueError("Contract mismatch")
 rng=np.random.default_rng(cfg["seed"]);rng.bit_generator.state=meta["sampler_rng_state"]
 updates=0;agent=None;n=0;last=-1;action_key=jax.random.PRNGKey(int(os.environ.get("TASK1_EPISODE_SEED",cfg["seed"])))
 def insert(i):
  l=json.loads((OUT/f"stage_transition_{i:03d}.json").read_text())
  a=json.loads((OUT/f"action_{i:03d}.json").read_text())
  r0=json.loads((OUT/f"raw_{i:03d}.json").read_text());r1=json.loads((OUT/f"raw_{i+1:03d}.json").read_text())
  dt=r1["sim_time"]-r0["sim_time"]
  tr=dict(observations=obs(i),next_observations=obs(i+1),actions=np.asarray(a["executed"],np.float32),
   rewards=np.float32(l["reward"]),masks=np.float32(l["mask"]),dones=bool(l["terminated"] or l["truncated"]))
  eligible=bool(l.get("valid_for_training",l["evidence_valid"]) and not l.get("injected",False))
  key=hashlib.sha256(str(OUT).encode()).hexdigest()[:16]+"_"+str(i)
  record=journal.append(key,tr,dict(reward_version=l["reward_version"],interface_version=l["interface_version"],
   terminated=l["terminated"],truncated=l["truncated"],dt_s=dt,physics_steps=round(dt*240),
   eligible=eligible,source=str(OUT),index=i,diagnostic_wrapper=SMOKE,reason=l.get("failure")))
  if eligible:online.append((tr,record))
 while True:
  request=OUT/f"request_{n:03d}.json";deadline=time.monotonic()+600
  while not request.exists() and not (OUT/"policy_stop.json").exists():
   if (OUT/"actor_error.json").exists():raise RuntimeError("Actor failed")
   if time.monotonic()>deadline:raise TimeoutError("Policy request")
   time.sleep(.03)
  if not request.exists():break
  o=obs(n)
  if agent is None:
   agent=make_agent(0,o,np.zeros(7,np.float32),("external","wrist_left","wrist_right"),cfg["discount"])
   agent=agent.replace(state=serialization.from_bytes(agent.state,payload))
   write("loaded_checkpoint.json",dict(path=str(checkpoint),sha256=meta["sha256"]))
  if n>0:insert(n-1);last=n-1
  if LEARN and n>0 and n%10==0:
   half=cfg["batch_size"]//2
   samples=[dataset.get(int(rng.integers(len(dataset)))) for _ in range(half)]
   samples += [online[int(rng.integers(len(online)))][0] for _ in range(half)]
   batch=freeze(jax.tree_util.tree_map(lambda *v:np.stack(v),*samples))
   agent,_=agent.update_high_utd(batch,utd_ratio=cfg["utd_ratio"]);jax.block_until_ready(agent.state.params);updates+=1
   if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(agent.state.params)):raise ValueError("Nonfinite weights")
  action_key,key=jax.random.split(action_key)
  raw=np.asarray(sample_policy(agent,o,key=key,deterministic=os.environ.get("TASK1_DETERMINISTIC")=="1"),float)
  if raw.shape!=(7,) or not np.isfinite(raw).all() or np.max(np.abs(raw))>1.00001:raise ValueError("Policy action")
  a=raw.copy()
  if SMOKE:
   a[:3]=np.clip(a[:3],-.002/.32,.002/.32)
   norm=np.linalg.norm(a[:3])
   if norm>.002/.32:a[:3]*=(.002/.32)/norm
   a[3:]=0
  write(f"action_{n:03d}.json",dict(index=n,proposed=raw.tolist(),executed=a.tolist(),network_updates=updates,
   sampling_mode="argmax" if os.environ.get("TASK1_DETERMINISTIC")=="1" else "seeded_sample"))
  n+=1
 stop=json.loads((OUT/"policy_stop.json").read_text());final=stop["final_observation_index"]
 if final-1>last:insert(final-1)
 if agent is not None:
  data=serialization.to_bytes(agent.state);(OUT/"checkpoint.msgpack").write_bytes(data)
  output_meta=dict(meta,updates=meta["updates"]+updates,updates_this_run=updates,
   online_transitions=len(online),demo_transitions=len(dataset),
   scope="online mixed learner" if LEARN else "isolated policy rollout; no learning",
   sampler_rng_state=rng.bit_generator.state,sha256=hashlib.sha256(data).hexdigest())
  (OUT/"checkpoint.json").write_text(json.dumps(output_meta,indent=2))
 write("worker_result.json",dict(passed=True,network_updates=updates,initial_online=initial_online,
  final_online=len(online),new_transitions=final,smoke=SMOKE,learning=LEARN))
except Exception:
 write("worker_error.json",dict(error=traceback.format_exc()));raise
