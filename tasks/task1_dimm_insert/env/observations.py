"""Vision policy boundary. No Isaac, task truth, contact forces or drives."""
from pathlib import Path
import numpy as np
from PIL import Image
CAMERAS=("external","wrist_left","wrist_right")
STATE_DIM=27
STATE_FIELDS=(("joint_positions",9),("joint_velocities",9),("gripper_width_m",1),
              ("goal_position",3),("goal_quaternion_xyzw",4),("gripper_target",1))

def state(o):
    def vector(x,n):
        a=np.asarray(x,dtype=np.float32)
        if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError("Bad robot state")
        return a
    q=vector(o["joint_positions"],9);qd=vector(o["joint_velocities"],9)
    c=o["controller_state"];gq=vector(c["goal_quaternion_xyzw"],4)
    norm=np.linalg.norm(gq)
    if not .99<=norm<=1.01:raise ValueError("Invalid target quaternion")
    gq=gq/norm
    if gq[3]<0:gq=-gq
    width=float(o["gripper_width_m"]);target=float(c["gripper_target"])
    if not (0<=width<=.081 and 0<=target<=.081):raise ValueError("Invalid gripper width")
    result=np.concatenate([q,qd,[width],vector(c["goal_position"],3),gq,[target]]).astype(np.float32)
    assert result.shape==(STATE_DIM,)
    return result[None,:] # (T=1, state), matches EncodingWrapper enable_stacking

def observation(o,rgb):
    if o.get("sensor_valid") is not True:raise ValueError("Invalid source sensors")
    out={"state":state(o)}
    for key in CAMERAS:
        a=np.asarray(rgb[key])
        if a.shape!=(256,256,3) or a.dtype!=np.uint8:raise ValueError("Expected uint8 RGB 256x256")
        out[key]=a[None,...]
    return out

def load_observation(o,episode,max_age_s=.25):
    episode=Path(episode).resolve();rgb={}
    for key in CAMERAS:
        info=o["images"][key]
        age=o["sim_time"]-info["sim_time"]
        if not 0<=age<=max_age_s:raise ValueError("Stale or future RGB")
        if info.get("quality",{}).get("valid") is not True:raise ValueError("Recorded RGB invalid")
        p=(episode/info["file"]).resolve()
        if not p.is_relative_to(episode):raise ValueError("Image outside episode")
        with Image.open(p) as im:
            im.load()
            if im.mode!="RGB":raise ValueError("Expected RGB file, no implicit colour conversion")
            a=np.asarray(im).copy()
        if float(a.mean())<1 and float(np.mean(a.max(axis=-1)>8))<.01:
            raise ValueError("Near-black RGB")
        rgb[key]=a
    return observation(o,rgb)

def make_agent(seed,sample_observation,sample_action):
    from algos.serl.agent import make_agent as factory
    if set(sample_observation)!={"state",*CAMERAS}:raise ValueError("Policy keys")
    return factory(seed,sample_observation,sample_action,CAMERAS)
