"""Train entry: compose task × algo. Stage 1A — FakeEnv dry-run or Lab stub."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

from runners.register import list_algos, list_tasks, resolve_algo, resolve_task

ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        data: dict = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or ":" not in s:
                continue
            if s.startswith("notes:"):
                break
            k, _, v = s.partition(":")
            k = k.strip()
            v = v.strip().strip("\"'")
            if k and v and not v.startswith("|"):
                data[k] = v
        return data
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_scene_path(cfg: dict, cfg_path: Path) -> Path | None:
    rel = cfg.get("scene_path")
    if not rel:
        return None
    p = Path(str(rel))
    if not p.is_absolute():
        for c in (ROOT / p, cfg_path.parent.parent.parent / p, Path.cwd() / p):
            if c.exists():
                return c.resolve()
        return (ROOT / p).resolve()
    return p.resolve()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Embodied_Safety train (Stage 1A)")
    p.add_argument("--task", required=True, choices=list_tasks())
    p.add_argument("--algo", required=True, choices=list_algos())
    p.add_argument("--config", required=True, help="Experiment YAML under configs/experiments/")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Force FakeEnv smoke (no Lab / no OnPolicyRunner.learn)",
    )
    p.add_argument(
        "--backend",
        default=None,
        choices=["fake", "isaac_lab", "isaac", "lab"],
        help="Override experiment backend (default: from YAML, or fake if --dry-run)",
    )
    p.add_argument(
        "--do-learn",
        action="store_true",
        help="Allow Runner.learn when a real VecEnv exists (guarded; see WIRING.md)",
    )
    args = p.parse_args(argv)

    task_mod_name = resolve_task(args.task)
    algo_mod_name = resolve_algo(args.algo)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cand = ROOT / cfg_path
        if cand.is_file():
            cfg_path = cand

    print("[train] Embodied_Safety Stage 1A")
    print(f"  task={args.task} -> {task_mod_name}")
    print(f"  algo={args.algo} -> {algo_mod_name}")
    print(f"  config={cfg_path} exists={cfg_path.is_file()}")
    print(f"  dry_run={args.dry_run}")

    cfg: dict[str, Any] = {}
    if cfg_path.is_file():
        cfg = _load_yaml(cfg_path)

    scene = _resolve_scene_path(cfg, cfg_path)
    if scene is not None:
        print(f"  scene_path={scene}")
        print(f"  scene_is_symlink={scene.is_symlink() if scene.exists() else 'MISSING'}")
        print(f"  scene_exists={scene.exists()}")
        usda = scene / "manual_franka_scene.usda"
        safety = scene / "safety_config.json"
        print(f"  scene_usda={usda.exists()} safety_config={safety.exists()}")

    if args.dry_run:
        backend = "fake"
    elif args.backend:
        backend = args.backend
    else:
        backend = str(cfg.get("backend", "fake")).lower()

    task_mod = importlib.import_module(task_mod_name)
    if not hasattr(task_mod, "make_env"):
        raise AttributeError(f"{task_mod_name} must export make_env(cfg)")
    env_cfg = {
        "scene_path": str(scene) if scene else cfg.get("scene_path"),
        "backend": backend,
        "max_episode_steps": int(cfg.get("max_episode_steps", 500)),
        "stage": cfg.get("stage", "stage1a"),
        "reward_mode": cfg.get("reward_mode", cfg.get("stage", "stage1a")),
        "num_envs": int(cfg.get("num_envs", 64)) if str(cfg.get("num_envs", "")).isdigit() or isinstance(cfg.get("num_envs"), int) else 64,
        "seed": int(cfg.get("seed", 42)) if str(cfg.get("seed", "42")).lstrip("-").isdigit() else 42,
        "device": cfg.get("device", "cuda:0"),
    }
    # coerce num_envs/seed if yaml loader left strings
    try:
        env_cfg["num_envs"] = int(cfg.get("num_envs", env_cfg["num_envs"]))
    except (TypeError, ValueError):
        pass
    try:
        env_cfg["seed"] = int(cfg.get("seed", env_cfg["seed"]))
    except (TypeError, ValueError):
        pass

    env = task_mod.make_env(env_cfg)
    print(f"  make_env -> {type(env).__name__} backend={getattr(env, 'backend', '?')}")

    algo_mod = importlib.import_module(algo_mod_name)
    if not hasattr(algo_mod, "train"):
        raise AttributeError(f"{algo_mod_name} must export train(env, cfg)")
    train_cfg = dict(cfg)
    train_cfg["algo_cfg"] = cfg.get("algo_cfg", "algos/rsl_rl_ppo/cfg/stage1a.yaml")
    train_cfg["scene_path"] = str(scene) if scene else train_cfg.get("scene_path")
    train_cfg["dry_run"] = bool(args.dry_run) or backend == "fake"
    train_cfg["do_learn"] = bool(args.do_learn)
    train_cfg["backend"] = backend
    train_cfg["reward_mode"] = cfg.get("reward_mode", cfg.get("stage", "stage1a"))
    train_cfg["stage"] = cfg.get("stage", train_cfg.get("stage", "stage1a"))
    # Forward numeric fields
    for k in ("num_envs", "seed", "device", "output_dir", "max_iterations"):
        if k in cfg:
            train_cfg[k] = cfg[k]
    result = algo_mod.train(env, train_cfg)
    status = (result or {}).get("status", "ok") if isinstance(result, dict) else "ok"
    print(f"  train_status={status}")
    if status == "scaffold_ok":
        print("[train] OK — FakeEnv dry-run complete")
    elif status == "lab_stub_ok":
        print("[train] OK — Lab stub path (no learn); see WIRING.md for blockers")
    else:
        print(f"[train] OK — status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
