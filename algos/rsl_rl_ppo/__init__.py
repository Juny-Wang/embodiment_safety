"""rsl_rl_ppo — Stage 1A PPO package (no t1_rl import)."""

from __future__ import annotations

from pathlib import Path

from algos.rsl_rl_ppo.train import train

PACKAGE_DIR = Path(__file__).resolve().parent
CFG_DIR = PACKAGE_DIR / "cfg"

__all__ = ["PACKAGE_DIR", "CFG_DIR", "train"]
