#!/usr/bin/env python3
"""Validate a Task1 SERL demo pickle (no Isaac needed)."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

OBS_DIM = 30
ACT_DIM = 7
REQUIRED = {"observations", "actions", "next_observations", "rewards", "masks", "dones"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl", type=str)
    args = parser.parse_args()
    path = Path(args.pkl)
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list) or not data:
        raise SystemExit(f"expected nonempty list, got {type(data)}")
    missing = REQUIRED - set(data[0])
    if missing:
        raise SystemExit(f"missing keys {missing}")
    n = len(data)
    obs = np.stack([np.asarray(t["observations"], dtype=np.float32) for t in data])
    act = np.stack([np.asarray(t["actions"], dtype=np.float32) for t in data])
    nxt = np.stack([np.asarray(t["next_observations"], dtype=np.float32) for t in data])
    if obs.shape != (n, OBS_DIM) or nxt.shape != (n, OBS_DIM):
        raise SystemExit(f"bad obs shapes {obs.shape} {nxt.shape}")
    if act.shape != (n, ACT_DIM):
        raise SystemExit(f"bad act shape {act.shape}")
    if np.max(np.abs(act)) > 1.0 + 1e-3:
        raise SystemExit(f"actions outside [-1,1]: max|a|={np.max(np.abs(act))}")
    dones = sum(bool(t["dones"]) for t in data)
    print(
        f"OK {path} transitions={n} dones={dones} "
        f"obs={obs.shape} act={act.shape} reward_mean={np.mean([float(t['rewards']) for t in data]):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
