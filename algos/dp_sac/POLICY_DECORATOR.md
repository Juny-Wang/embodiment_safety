# Policy Decorator for Task1

This backend adapts [Policy Decorator](https://arxiv.org/abs/2412.13630)
and its [official visual implementation](https://github.com/tongzhoumu/policy_decorator/blob/main/online/pi_dec_diffusion_maniskill2_rgbd.py)
to the existing Task1 environment and trained DP EMA checkpoint.

The DP, its image encoders and its normalizers stay frozen. DP receives two
frames of three RGB views and rotation6d33 state. The residual actor uses only
the latest frame's 417-dimensional embedding, without the base action. Actor
and twin critics use three 256-unit ReLU layers. Residuals are tanh Gaussian
actions, scaled by 0.05 on each axis. The critics consume the observation and
the proposed sum of DP action and scaled residual. The environment clips that
sum to [-1, 1]. The replay retains the sampled base action with its observation
and the actual selected residual, including zero when exploration is disabled.

During training, residual corrections are enabled independently at each step
with probability min(environment_steps / 30000, 1). Before the first learning
update, enabled corrections are uniform samples; afterward they come from the
residual actor. Evaluation always enables the residual mean. Checkpoints retain
the exploration step count and RNG alongside networks, optimizers and replay.

The initial Task1 pilot keeps the baseline's discount, update frequency,
learning rate, temperature initialization, batch size, replay size and
128-transition learning start. It also retains Task1 stage rewards, stop
conditions, 10 Hz control and the DP's single-action execution horizon.
These are Task1 adaptations; this run does not reproduce the paper's benchmark
settings, sample budget or reported success rates. The official visual method
also freezes the base visual embedding. No additional dependencies are needed.

From the repository root:

```bash
python3 -B -m runners.train_dp_sac \
  --config configs/experiments/task1_policy_decorator.json \
  --episodes 10 --max-steps 2400 --wall-timeout-s 7200 \
  --seed 20261102 --output outputs/runs/task1/policy_decorator/train_001

python3 -B -m runners.train_dp_sac \
  --checkpoint outputs/runs/task1/policy_decorator/train_001/episode_010/checkpoint.pt \
  --eval --episodes 3 --seed 20261201 \
  --output outputs/runs/task1/policy_decorator/eval_001
```

Use the same evaluation seeds, episode limits and stop conditions for DP-only,
the original residual-SAC baseline and Policy Decorator. Compare complete task
success, completed stages, reward, termination reason and transitions used for
training. Equal episode counts need not imply equal training transition counts.

## Pilot batch

`outputs/runs/task1/policy_decorator/pilot_001/pipeline.py` queues behind
`dp_sac/train_batch_001`, then runs a fresh 140-step training episode and a
short evaluation to verify the native environment integration. It continues
from that method's own checkpoint for ten training episodes and evaluates all
three methods on the same three seeds. Its `pipeline_status.json` distinguishes
queueing, training, evaluation, completion and failure. `comparison.json` holds
the three evaluation summaries and their training budgets.

## Limitations

This is an initial single-training-seed pilot. Three evaluation episodes cannot
establish reliable method rankings. The base policy's full-task success rate
is not established. Ten episodes may end before the 30000-step exploration
schedule completes; results must include the final exploration probability.
