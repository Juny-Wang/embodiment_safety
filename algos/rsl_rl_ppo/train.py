"""PPO train entry — Stage 1A.

- FakeEnv / --dry-run: print plan + one reset/step (no rsl_rl required).
- Lab path: probe stack, optionally construct OnPolicyRunner on FakeVecEnv
  (construct-only), refuse real learn() until DirectRLEnv exists.
Never imports franka_sim/t1_rl.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

from algos.rsl_rl_ppo.runner import Runner, build_rsl_train_cfg, rsl_rl_available

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CFG = PACKAGE_DIR / "cfg" / "stage1a.yaml"
ROOT = PACKAGE_DIR.parents[1]


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
            k, v = k.strip(), v.strip().strip("\"'")
            if not k or not v or v.startswith(("[", "{", "|")):
                continue
            if v.lower() in ("true", "false"):
                data[k] = v.lower() == "true"
            else:
                try:
                    data[k] = int(v) if "." not in v and "e" not in v.lower() else float(v)
                except ValueError:
                    data[k] = v
        return data
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_algo_cfg(cfg: Mapping[str, Any]) -> tuple[Path, dict]:
    algo_cfg_path = Path(str(cfg.get("algo_cfg", DEFAULT_CFG)))
    if not algo_cfg_path.is_absolute():
        for cand in (PACKAGE_DIR / algo_cfg_path, ROOT / algo_cfg_path, Path.cwd() / algo_cfg_path):
            if cand.is_file():
                algo_cfg_path = cand
                break
    algo_cfg: dict = {}
    if algo_cfg_path.is_file():
        algo_cfg = _load_yaml(algo_cfg_path)
    # Overlay experiment-level keys
    for k in (
        "num_envs", "seed", "device", "max_iterations", "experiment_name",
        "output_dir", "run_name",
    ):
        if k in cfg and cfg[k] is not None:
            algo_cfg[k] = cfg[k]
    return algo_cfg_path, algo_cfg


def _resolve_log_dir(cfg: Mapping[str, Any], algo_cfg: Mapping[str, Any]) -> Path:
    out = cfg.get("output_dir") or algo_cfg.get("output_dir")
    if out:
        base = Path(str(out))
        if not base.is_absolute():
            base = ROOT / base
    else:
        exp = str(algo_cfg.get("experiment_name", "task1_dimm_insert_stage1a"))
        base = ROOT / "outputs" / "runs" / "task1_dimm_insert" / "rsl_rl_ppo" / exp
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = base / stamp
    run.mkdir(parents=True, exist_ok=True)
    return run


def _is_fake_env(env: Any) -> bool:
    return getattr(env, "backend", "") == "fake" or type(env).__name__ == "FakeEnv"


def _is_lab_stub(env: Any) -> bool:
    # LabEnvHandle without live vec_env stays on stub path.
    if getattr(env, "backend", "") == "isaac_lab_direct" and getattr(env, "vec_env", None) is not None:
        return False
    return getattr(env, "backend", "") in ("isaac_lab_stub", "isaac_lab_direct") or type(env).__name__ == "LabEnvHandle"


def train(env: Any, cfg: Optional[Mapping[str, Any]] = None) -> MutableMapping[str, Any]:
    cfg = dict(cfg or {})
    dry_run = bool(cfg.get("dry_run", False))
    algo_cfg_path, algo_cfg = _resolve_algo_cfg(cfg)
    train_cfg = build_rsl_train_cfg(algo_cfg)

    print("[rsl_rl_ppo] Stage 1A train()")
    print(f"  dry_run={dry_run}")
    print(f"  algo_cfg={algo_cfg_path} exists={algo_cfg_path.is_file()}")
    print(f"  num_envs={algo_cfg.get('num_envs')} seed={algo_cfg.get('seed')}")
    print(
        f"  learning_rate={train_cfg['algorithm'].get('learning_rate')} "
        f"gamma={train_cfg['algorithm'].get('gamma')} "
        f"clip={train_cfg['algorithm'].get('clip_param')}"
    )
    print(f"  rsl_rl_available={rsl_rl_available()}")
    print(f"  env_type={type(env).__name__} backend={getattr(env, 'backend', '?')}")

    # --- FakeEnv smoke path (also --dry-run) ---------------------------------
    if dry_run or _is_fake_env(env):
        if hasattr(env, "observation_space"):
            obs_sp = env.observation_space
            print(
                f"  obs_space={obs_sp.to_dict() if hasattr(obs_sp, 'to_dict') else obs_sp}"
            )
        if hasattr(env, "action_space"):
            act_sp = env.action_space
            print(
                f"  act_space={act_sp.to_dict() if hasattr(act_sp, 'to_dict') else act_sp}"
            )
        if hasattr(env, "reset") and hasattr(env, "step"):
            obs, info = env.reset()
            act_dim = int(getattr(env.action_space, "shape", [7])[0])
            zeros = [0.0] * act_dim
            _obs2, rew, term, trunc, info2 = env.step(zeros)
            print(
                f"  dry_run reset_obs_len={len(obs)} step_reward={rew} "
                f"terminated={term} truncated={trunc}"
            )
            print(f"  dry_run reward_terms={info2.get('reward_terms')}")
            print(f"  dry_run safety_limits_ref={info.get('safety_limits_ref')}")
        print("  note: FakeEnv path — no OnPolicyRunner.learn()")
        print("  note: franka_sim/t1_rl not used")
        return {
            "status": "scaffold_ok",
            "algo_cfg": algo_cfg,
            "algo_cfg_path": str(algo_cfg_path),
            "train_cfg_keys": list(train_cfg.keys()),
            "rsl_rl_available": rsl_rl_available(),
        }

    # --- Lab stub / real path ------------------------------------------------
    log_dir = _resolve_log_dir(cfg, algo_cfg)
    print(f"  log_dir={log_dir}")

    if _is_lab_stub(env):
        probe = getattr(env, "probe", {}) or {}
        print(f"  lab_probe rsl_rl={probe.get('rsl_rl') is not None} "
              f"isaaclab={probe.get('isaaclab') is not None} "
              f"DirectRLEnv={probe.get('DirectRLEnv') is not None} "
              f"pxr={probe.get('pxr') is not None}")
        if getattr(env, "blocker", None):
            print("  lab_blocker_summary (first line):")
            print("   ", str(env.blocker).splitlines()[0])
        usd = None
        if hasattr(env, "cfg") and hasattr(env.cfg, "resolved_usd"):
            usd = env.cfg.resolved_usd()
            print(f"  usd={usd} exists={usd.is_file()}")

        # Construct-only smoke with FakeVecEnv when rsl_rl importable
        construct_ok = False
        construct_err = None
        if rsl_rl_available() and bool(cfg.get("construct_runner_smoke", True)):
            try:
                from tasks.task1_dimm_insert.env.lab_env import try_make_fake_vec_env

                fake_vec = try_make_fake_vec_env({
                    **algo_cfg,
                    "scene_path": getattr(getattr(env, "cfg", None), "scene_path", cfg.get("scene_path")),
                    "num_envs": int(algo_cfg.get("num_envs", 4)),
                    "fake_vec_device": "cpu",
                    "device": "cpu",
                })
                if fake_vec is not None:
                    runner = Runner(
                        fake_vec,
                        {**algo_cfg, "device": "cpu"},
                        log_dir=str(log_dir / "construct_smoke"),
                        device="cpu",
                    )
                    construct_ok = runner.ready
                    print(f"  OnPolicyRunner construct_smoke ready={construct_ok}")
                else:
                    construct_err = "try_make_fake_vec_env returned None (torch/tensordict?)"
            except Exception as e:  # noqa: BLE001
                construct_err = f"{type(e).__name__}: {e}"
                print(f"  OnPolicyRunner construct_smoke FAILED: {construct_err}")

        status = "lab_stub_ok"
        if getattr(env, "vec_env", None) is None:
            print("  plan: implement DirectRLEnv then call Runner.learn()")
            print("  note: refusing learn() — no real VecEnv yet (see WIRING.md)")
            return {
                "status": status,
                "algo_cfg": algo_cfg,
                "algo_cfg_path": str(algo_cfg_path),
                "log_dir": str(log_dir),
                "rsl_rl_available": rsl_rl_available(),
                "construct_smoke": construct_ok,
                "construct_error": construct_err,
                "blocker": getattr(env, "blocker", None),
                "usd": str(usd) if usd else None,
            }

    # Real vec_env on handle or env itself
    vec = getattr(env, "vec_env", None) or (env if hasattr(env, "get_observations") else None)
    if vec is None:
        raise RuntimeError(
            "No VecEnv available for PPO. Use --dry-run / backend=fake, or "
            "implement Lab DirectRLEnv (see tasks/.../env/lab_env.py)."
        )

    do_learn = bool(cfg.get("do_learn", False))
    max_iters = int(cfg.get("max_iterations", algo_cfg.get("max_iterations", 2000)))
    runner = Runner(vec, algo_cfg, log_dir=str(log_dir), device=str(algo_cfg.get("device", "cuda:0")))
    print(f"  OnPolicyRunner ready={runner.ready} do_learn={do_learn}")
    if do_learn:
        # Guard: never accidental multi-hour run from default config
        if max_iters > 5 and not bool(cfg.get("allow_long_run", False)):
            raise RuntimeError(
                f"Refusing learn(max_iterations={max_iters}). Pass allow_long_run=true "
                "or set max_iterations<=5 for a short smoke learn."
            )
        runner.learn(num_learning_iterations=max_iters)
        return {"status": "learn_finished", "log_dir": str(log_dir), "max_iterations": max_iters}

    print("  note: Runner constructed; learn() skipped (do_learn=false)")
    return {
        "status": "runner_ready",
        "log_dir": str(log_dir),
        "rsl_rl_available": rsl_rl_available(),
        "ready": runner.ready,
    }
