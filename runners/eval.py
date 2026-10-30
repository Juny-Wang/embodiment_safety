"""Eval entry: compose task × algo. Scaffold-only (no Isaac)."""

from __future__ import annotations

import argparse
from pathlib import Path

from runners.register import list_algos, list_tasks, resolve_algo, resolve_task


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Embodied_Safety eval (scaffold)")
    p.add_argument("--task", required=True, choices=list_tasks())
    p.add_argument("--algo", required=True, choices=list_algos())
    p.add_argument("--config", required=True, help="Experiment YAML under configs/experiments/")
    args = p.parse_args(argv)

    task_mod = resolve_task(args.task)
    algo_mod = resolve_algo(args.algo)
    cfg_path = Path(args.config)

    print("[scaffold] eval")
    print(f"  task={args.task} -> {task_mod}")
    print(f"  algo={args.algo} -> {algo_mod}")
    print(f"  config={cfg_path} exists={cfg_path.is_file()}")
    print("  (no Isaac / no checkpoint load yet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
