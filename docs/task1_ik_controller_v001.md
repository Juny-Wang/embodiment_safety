# Task1 continuous IK controller comparison

`dogbox_trf` completed 600 policy steps and 1,200 accepted IK solves without an IK stop. The TRF control stopped at policy step 164 with the same IK residual failure as the original evaluation. Median solve time was 4.71 ms for dogbox and 511.53 ms for TRF in these runs.

Both evaluations used the final `train_batch_001/train/episode_010/checkpoint.pt`, deterministic residual means, DP seed `20261201`, a 600-step limit, and a fresh Isaac reset. They ran sequentially. Neither evaluation updated network weights or added replay transitions.

## Results

| Measurement | TRF | Dogbox, TRF fallback |
|---|---:|---:|
| Policy steps | 164 | 600 |
| Accepted / attempted IK | 326 / 327 | 1200 / 1200 |
| Stop | IK residual error | 600-step time limit |
| Solve time, median / p95 (ms) | 511.53 / 536.86 | 4.71 / 6.12 |
| Controller tick, median / p95 (ms) | 512.30 / 537.41 | 5.19 / 6.73 |
| First 10 simulation seconds: tick median (ms) | 443.61 | 5.22 |
| Controller ticks exceeding 50 ms | 327 | 1 |
| Episode wall time (s, including startup) | 277.72 | 392.34 |
| Largest commanded arm step (rad) | 0.006869 | 0.009341 |
| Largest consecutive IK candidate step (rad) | 0.006869 | 0.009341 |
| Commanded arm rate violations | 0 | 0 |
| Smallest candidate joint-limit margin (rad) | 0.341941 | 0.342135 |
| Task reward / assembly success | 0 / false | 0 / false |

Dogbox solved all 1,200 targets directly; the TRF fallback was never needed in this rollout. Maximum accepted FK position and orientation residuals were below 5e-16 m and 9e-16 rad. These are numerical IK residuals, not measured physical tracking errors. The largest target step was 0.009341 rad (0.535 degrees), below the existing 0.05 rad per-tick limit.

## Execution

`env/ik.py` runs bounded dogbox first and continues from its candidate with TRF only if the candidate fails the physical pose or joint-limit checks. Both methods use the existing weighted residual, central-difference Jacobian, termination tolerances and 400-evaluation limit per attempt. A previous solution is reused only after its pose error is checked against the current target.

`env/ik_controller.py` supplies the policy tick installed by `env/execution.py`. The frozen teleoperation controller and scene manifest remain unchanged. Position tolerance is 0.05 mm, orientation tolerance is 0.0005 rad, IK period is 0.05 s, and arm target speed is capped at 1 rad/s. Tracking, contact stops and task acceptance are retained. DP/SAC checkpoints, network architecture and inference steps are unchanged.

`runners.train_dp_sac --ik-solver dogbox_trf` selects the independent evaluation configuration. The default remains `trf`; the alternative currently requires evaluation mode. Each episode records its solver in `execution_configuration.json` and logs solver attempts, accepted poses, target changes and timings in `ik_trace.jsonl`.

## Validation

All 21 tests passed across `test_ik`, `test_dp_sac`, `test_unlimited_evaluation` and `test_partial_terminal`. IK tests cover reachable and unreachable targets, fallback based on pose accuracy rather than the optimizer success flag, verified cache reuse, arm rate limiting, rejection stops, and matching the frozen controller's default TRF outputs.

The deployed solver also passed all five recorded failure targets from both the measured-joint and failed-candidate seeds (10/10 trials).

## Reproduce

```bash
python3 -B -m runners.train_dp_sac --eval \
  --checkpoint outputs/runs/task1/dp_sac/train_batch_001/train/episode_010/checkpoint.pt \
  --output outputs/reports/task1_ik_controller_recheck/dogbox_eval \
  --episodes 1 --max-steps 600 --wall-timeout-s 3600 \
  --seed 20261201 --ik-solver dogbox_trf
```

Use `--ik-solver trf` and a new output directory for the control.

Artifacts on `isaac-b` / `ufdwaa`, relative to `/mnt/data2/wjy_SERL/Embodied_Safety/outputs/reports/task1_ik_controller_v001/`:

- `comparison.json`: timing and continuity statistics; `summarize.py` computes them from the traces.
- `recorded_targets.json`: ten frozen-target trials; `check_recorded_targets.py` reproduces them.
- `dogbox_eval/` and `trf_eval/`: evaluation summaries and per-tick evidence.
- `dogbox_eval_three_views.mp4` and `trf_eval_three_views.mp4`: external, left-wrist and right-wrist views at 10 fps of simulation time (60.1 s and 16.5 s including the final observation).
- `videos.json`: frame counts and playback clock; `export_videos.py` exports the recorded observations.

## Limitations

This is one seed and one 60-second continuous rollout budget. Neither method completed assembly or earned a stage reward. The result supports faster IK and removal of the observed early numerical stop; it does not establish successful SAC learning or performance beyond 600 steps. The two closed-loop trajectories diverge, so solve-time statistics are observations of their respective trajectories, not identical-target microbenchmarks. Dogbox's first controller tick took 190 ms; all subsequent ticks stayed below 50 ms. The full 60-second simulation still took 392 seconds of wall time, so the complete system is not real-time.

