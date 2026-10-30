"""Thin SERL replay helpers for Task1 (online + optional demo mix)."""

from __future__ import annotations

import numpy as np

from serl_launcher.data.replay_buffer import ReplayBuffer


class Task1ReplayBuffer(ReplayBuffer):
    """Standard SERL transition schema; no cost columns (plain task)."""


def sample_half_batch(online: Task1ReplayBuffer, demo: Task1ReplayBuffer, batch_size: int):
    """Exact 50/50 mix. batch_size must be even; both buffers nonempty."""
    if batch_size % 2 != 0:
        raise ValueError("batch_size must be even for an exact 50/50 mix")
    if len(demo) == 0 or len(online) == 0:
        raise RuntimeError("50/50 sampling needs both demo and online buffers")
    half = batch_size // 2
    demo_batch = demo.sample(batch_size=half)
    online_batch = online.sample(batch_size=half)
    keys = demo_batch.keys()
    if keys != online_batch.keys():
        raise RuntimeError(f"buffer schemas differ: {keys} vs {online_batch.keys()}")
    mixed = {
        key: np.concatenate([demo_batch[key], online_batch[key]], axis=0) for key in keys
    }
    return mixed, {"demo_count": half, "online_count": half, "demo_fraction": 0.5}
