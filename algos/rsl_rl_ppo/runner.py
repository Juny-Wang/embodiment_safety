"""Wrapper around rsl_rl.runners.OnPolicyRunner (import-guarded)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional


def rsl_rl_available() -> bool:
    try:
        from rsl_rl.runners import OnPolicyRunner  # noqa: F401

        return True
    except Exception:
        return False


def build_rsl_train_cfg(algo_cfg: Mapping[str, Any]) -> dict:
    """Map stage1a.yaml → rsl_rl **5.x** OnPolicyRunner train_cfg.

    rsl-rl >= 4/5 uses separate ``actor`` / ``critic`` MLPModel blocks (not the
    deprecated ``policy: ActorCritic`` shape).
    """
    actor_hidden = algo_cfg.get("actor_hidden_dims", [256, 256, 128])
    critic_hidden = algo_cfg.get("critic_hidden_dims", [256, 256, 128])
    if isinstance(actor_hidden, str):
        actor_hidden = [int(x) for x in actor_hidden.strip("[]").split(",") if x.strip()]
    if isinstance(critic_hidden, str):
        critic_hidden = [int(x) for x in critic_hidden.strip("[]").split(",") if x.strip()]

    # Already in new nested shape
    if "actor" in algo_cfg and "critic" in algo_cfg and "algorithm" in algo_cfg:
        train_cfg = {k: v for k, v in algo_cfg.items()}
        train_cfg.setdefault(
            "obs_groups",
            {"actor": ["policy"], "critic": ["policy"]},
        )
        train_cfg.setdefault("multi_gpu", None)
        return train_cfg

    init_std = float(algo_cfg.get("init_noise_std", algo_cfg.get("init_std", 1.0)))
    activation = str(algo_cfg.get("activation", "elu"))

    train_cfg: dict = {
        "seed": int(algo_cfg.get("seed", 42)),
        "device": str(algo_cfg.get("device", "cuda:0")),
        "num_steps_per_env": int(algo_cfg.get("num_steps_per_env", 24)),
        "max_iterations": int(algo_cfg.get("max_iterations", 2000)),
        "save_interval": int(algo_cfg.get("save_interval", 50)),
        "experiment_name": str(algo_cfg.get("experiment_name", "task1_dimm_insert_stage1a")),
        "run_name": str(algo_cfg.get("run_name", "")),
        "logger": str(algo_cfg.get("logger", "tensorboard")),
        "neptune_project": algo_cfg.get("neptune_project", "embodied_safety"),
        "wandb_project": algo_cfg.get("wandb_project", "embodied_safety"),
        "resume": bool(algo_cfg.get("resume", False)),
        "load_run": algo_cfg.get("load_run", ".*"),
        "load_checkpoint": algo_cfg.get("load_checkpoint", "model_.*.pt"),
        "obs_groups": algo_cfg.get("obs_groups")
        or {"actor": ["policy"], "critic": ["policy"]},
        "multi_gpu": algo_cfg.get("multi_gpu", None),
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": list(actor_hidden),
            "activation": activation,
            "obs_normalization": bool(algo_cfg.get("actor_obs_normalization", False)),
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": init_std,
                "std_type": str(algo_cfg.get("noise_std_type", "scalar")),
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": list(critic_hidden),
            "activation": activation,
            "obs_normalization": bool(algo_cfg.get("critic_obs_normalization", False)),
            "distribution_cfg": None,
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": float(algo_cfg.get("value_loss_coef", 1.0)),
            "use_clipped_value_loss": bool(algo_cfg.get("use_clipped_value_loss", True)),
            "clip_param": float(algo_cfg.get("clip_param", 0.2)),
            "entropy_coef": float(algo_cfg.get("entropy_coef", 0.01)),
            "num_learning_epochs": int(algo_cfg.get("num_learning_epochs", 5)),
            "num_mini_batches": int(algo_cfg.get("num_mini_batches", 4)),
            "learning_rate": float(algo_cfg.get("learning_rate", 1.0e-3)),
            "schedule": str(algo_cfg.get("schedule", "adaptive")),
            "gamma": float(algo_cfg.get("gamma", 0.99)),
            "lam": float(algo_cfg.get("lam", 0.95)),
            "desired_kl": float(algo_cfg.get("desired_kl", 0.01)),
            "max_grad_norm": float(algo_cfg.get("max_grad_norm", 1.0)),
        },
    }
    return train_cfg


class Runner:
    """Wraps rsl_rl OnPolicyRunner when importable and a VecEnv is provided."""

    def __init__(
        self,
        env: Any,
        cfg: Optional[Mapping[str, Any]] = None,
        *,
        log_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.env = env
        self.cfg = dict(cfg or {})
        self.log_dir = log_dir
        self.device = device or str(self.cfg.get("device", "cpu"))
        self._impl = None
        self._train_cfg = build_rsl_train_cfg(self.cfg)

        if not rsl_rl_available():
            return
        # Only construct if env looks like VecEnv (has get_observations)
        if env is None or not hasattr(env, "get_observations"):
            return
        import copy

        from rsl_rl.runners import OnPolicyRunner

        # construct_algorithm pops class_name from nested dicts — pass a copy
        self._impl = OnPolicyRunner(
            env=env,
            train_cfg=copy.deepcopy(self._train_cfg),
            log_dir=log_dir,
            device=self.device,
        )

    @property
    def ready(self) -> bool:
        return self._impl is not None

    def learn(self, num_learning_iterations: Optional[int] = None) -> None:
        if self._impl is None:
            raise RuntimeError(
                "Runner.learn: OnPolicyRunner not constructed. Need importable "
                "rsl_rl AND a VecEnv with get_observations(). "
                "See algos/rsl_rl_ppo/README.md and WIRING.md. "
                f"rsl_rl_available={rsl_rl_available()} env_type={type(self.env).__name__}"
            )
        iters = (
            int(num_learning_iterations)
            if num_learning_iterations is not None
            else int(self._train_cfg.get("max_iterations", 2000))
        )
        self._impl.learn(num_learning_iterations=iters)

    def save(self, path: str) -> None:
        if self._impl is None:
            raise RuntimeError("no OnPolicyRunner to save")
        self._impl.save(path)

    def load(self, path: str) -> None:
        if self._impl is None:
            raise RuntimeError("no OnPolicyRunner to load")
        self._impl.load(path)
