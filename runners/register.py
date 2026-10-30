"""Task and algo registries.

Scaffold-only: names map to import paths / ids. No Isaac dependency.
"""

from __future__ import annotations

from typing import Dict, List

TASKS: Dict[str, str] = {
    "task1_dimm_insert": "tasks.task1_dimm_insert",
}

ALGOS: Dict[str, str] = {
    "rsl_rl_ppo": "algos.rsl_rl_ppo",
    "rsl_rl_ppo_plain": "algos.rsl_rl_ppo.train_plain",
    "serl": "algos.serl",
    "diffusion_policy": "algos.diffusion_policy",
    "act": "algos.act",
}


def list_tasks() -> List[str]:
    return sorted(TASKS.keys())


def list_algos() -> List[str]:
    return sorted(ALGOS.keys())


def resolve_task(task_id: str) -> str:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task {task_id!r}. Known: {list_tasks()}")
    return TASKS[task_id]


def resolve_algo(algo_id: str) -> str:
    if algo_id not in ALGOS:
        raise KeyError(f"Unknown algo {algo_id!r}. Known: {list_algos()}")
    return ALGOS[algo_id]
