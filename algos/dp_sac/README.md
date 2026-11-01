# DP-BC + SAC

The trained Diffusion Policy supplies the base action. A Gaussian residual actor
learns seven-dimensional corrections with SAC. The DP EMA weights, ResNet
encoders and normalizers remain frozen throughout training.

Observation: two frames of three RGB cameras and rotation6d33 state, using the
same adapter as DP-BC. Each frame supplies 384 visual features and 33 normalized
state values; two frames plus the base action form an 841-dimensional SAC input.
The residual actor and twin Q networks each use two 256-unit hidden layers.

The policy executes `clip(base_action + residual_scale * residual, -1, 1)` at
10 Hz. SAC treats the unscaled residual as its action: its Q networks, replay
and entropy term all use that coordinate system. The default residual scale is
0.05 on each axis. The first 128 transitions use DP alone; subsequent transitions
receive one SAC update with batch size 64. Early residual exploration starts with
zero mean and standard deviation exp(-3) before tanh.

Training uses newly collected, valid online transitions. The existing SERL replay
and BC-only observation/action pairs are not used. Time limits retain bootstrap
mask 1; true terminal failures and success use mask 0. Evaluation never appends
to the training replay. Original task reward, control period and stop thresholds
apply.

Run from `/mnt/data2/wjy_SERL/Embodied_Safety`:

```bash
# Use the completed 30,000-step DP EMA checkpoint and collect one training episode.
python3 -B -m runners.train_dp_sac \
  --output outputs/runs/task1/dp_sac/train_001

# Continue with saved residual weights, optimizers, target Q, temperature and replay.
python3 -B -m runners.train_dp_sac \
  --checkpoint outputs/runs/task1/dp_sac/train_001/episode_001/checkpoint.pt \
  --episodes 10 --output outputs/runs/task1/dp_sac/train_002

# Evaluate the residual mean; no learning or training replay writes.
python3 -B -m runners.train_dp_sac --eval \
  --checkpoint outputs/runs/task1/dp_sac/train_001/episode_001/checkpoint.pt \
  --output outputs/runs/task1/dp_sac/eval_001

# DP-only comparison through the same observation/action path.
python3 -B -m runners.train_dp_sac --baseline \
  --output outputs/runs/task1/dp_sac/baseline_001
```

Use matching `--seed` and `--max-steps` for baseline and residual comparisons.
DP sampling uses its own seeded random generator, independent of SAC updates.
Each episode starts a fresh Isaac process. `--max-steps` defaults to 2400;
`--wall-timeout-s` defaults to 3600. Output directories must be new.
`--config` selects SAC hyperparameters for a new run; resuming retains the saved
configuration and base checkpoint. `--base-checkpoint` selects another trained
DP with the same observation and action contract for a new run.

The worker uses `algos/diffusion_policy/.venv/bin/python`. Checkpoints contain the
residual networks, optimizers, replay and sampling state; they reference the
original DP checkpoint without copying it. A checkpoint is saved every 100
accepted transitions and at episode completion. `metrics.jsonl` records losses,
temperature and update counts. Action records include base action, sampled
residual, applied correction and final executed action.

## Limitations

The base checkpoint has completed BC training; full-task success has not been
established. Short rollout verification establishes integration and learning
updates, not policy convergence. DP remains stochastic under seeded DDIM initial
noise; residual evaluation uses its mean. Residual bounds limit how far the
learned controller can depart from DP. Cached replay features require the base
checkpoint and normalizers to remain fixed.

## Policy Decorator

The `policy_decorator` backend adds progressive exploration, an observation-only
residual actor and critics over the summed action. See
[Policy Decorator for Task1](POLICY_DECORATOR.md) for the method, configuration
and matched-seed pilot.
