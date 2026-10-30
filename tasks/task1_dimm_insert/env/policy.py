"""Explicit 31D revision: original robot state27 plus four reward-history bits."""
import numpy as np
from . import observations as original
from .reward import INTERFACE
def attach_progress(obs, progress):
    p=np.asarray(progress,dtype=np.float32)
    if p.shape!=(4,) or not np.isin(p,[0,1]).all():
        raise ValueError("Expected four binary reward flags")
    s=np.asarray(obs["state"])
    if s.shape!=(1,27):
        raise ValueError("Expected original state (1,27)")
    return {**obs,"state":np.concatenate([s,p[None,:]],axis=-1).astype(np.float32)}
def load_observation(o,episode,progress,max_age_s=.1):
    return attach_progress(original.load_observation(o,episode,max_age_s),progress)
def make_agent(seed,sample_observation,sample_action):
    if np.asarray(sample_observation["state"]).shape!=(1,31):
        raise ValueError(INTERFACE+" requires 31D, old checkpoints incompatible")
    return original.make_agent(seed,sample_observation,sample_action)
