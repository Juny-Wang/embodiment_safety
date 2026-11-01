# Task1 IK failure diagnosis

All five recorded failure targets have joint-limit-compliant solutions at the original 0.05 mm position and 0.0005 rad orientation tolerances. SciPy dogbox passes all five from both the measured joints and the failed candidate. This supports a solver/convergence explanation for these stops rather than target unreachability.

The existing TRF method passes four of five cases when restarted from measured joints, and five of five when continuing the failed candidate. Continuing TRF takes 330–352 ms in these trials; dogbox takes 2.7–3.9 ms from the failed candidate. Timings are individual offline measurements, not end-to-end training speedups.

Recommended next change: use bounded dogbox as a local fallback when the existing solve fails the physical tolerances. Retain the same residual definition, limits, joint speed cap, contact checks and task acceptance. Verify in a separate closed-loop run before changing the ongoing training configuration.

## Offline comparison

All candidates use the recorded targets, joint limits and original tolerances. The logged failed-candidate errors are reproduced before each comparison. Lula is limited to one supplied seed, with no additional random restarts. Acceptance uses the original rotation-vector norm, regardless of solver success flags.

| Episode | Method | Seed | Accepted | Position error (mm) | Rotation error (deg) | Solve time (ms) | Max joint change from measured (deg) |
|---|---|---|---|---|---|---|---|
| 1 | scipy_trf | measured | True | 0.000000 | 0.000000 | 16.765 | 1.040 |
| 1 | scipy_trf | failed_candidate | True | 0.020484 | 0.015461 | 349.142 | 1.027 |
| 1 | scipy_dogbox | measured | True | 0.000000 | 0.000000 | 19.242 | 1.087 |
| 1 | scipy_dogbox | failed_candidate | True | 0.000000 | 0.000000 | 2.901 | 1.044 |
| 1 | lula_ccd_bfgs | measured | False | 0.000315 | 0.028812 | 2.146 | 1.057 |
| 1 | lula_ccd_bfgs | failed_candidate | False | 0.003770 | 0.078379 | 0.014 | 1.008 |
| 2 | scipy_trf | measured | True | 0.000000 | 0.000000 | 13.501 | 1.117 |
| 2 | scipy_trf | failed_candidate | True | 0.011773 | 0.011215 | 329.945 | 1.108 |
| 2 | scipy_dogbox | measured | True | 0.000000 | 0.000000 | 3.712 | 1.133 |
| 2 | scipy_dogbox | failed_candidate | True | 0.000000 | 0.000000 | 2.739 | 1.133 |
| 2 | lula_ccd_bfgs | measured | False | 0.001252 | 0.043933 | 0.045 | 1.123 |
| 2 | lula_ccd_bfgs | failed_candidate | False | 0.000196 | 0.044554 | 0.011 | 1.064 |
| 3 | scipy_trf | measured | False | 0.006007 | 0.057864 | 343.408 | 0.958 |
| 3 | scipy_trf | failed_candidate | True | 0.000284 | 0.003770 | 345.194 | 0.959 |
| 3 | scipy_dogbox | measured | True | 0.000000 | 0.000000 | 3.529 | 0.975 |
| 3 | scipy_dogbox | failed_candidate | True | 0.000000 | 0.000000 | 3.917 | 0.959 |
| 3 | lula_ccd_bfgs | measured | False | 0.000476 | 0.075860 | 0.115 | 0.956 |
| 3 | lula_ccd_bfgs | failed_candidate | True | 0.001109 | 0.026653 | 0.021 | 0.959 |
| 4 | scipy_trf | measured | True | 0.000000 | 0.000000 | 10.324 | 5.837 |
| 4 | scipy_trf | failed_candidate | True | 0.004598 | 0.004661 | 351.915 | 4.708 |
| 4 | scipy_dogbox | measured | True | 0.000000 | 0.000000 | 2.774 | 4.436 |
| 4 | scipy_dogbox | failed_candidate | True | 0.000000 | 0.000000 | 2.806 | 4.707 |
| 4 | lula_ccd_bfgs | measured | False | 0.000404 | 0.067741 | 0.078 | 4.743 |
| 4 | lula_ccd_bfgs | failed_candidate | False | 0.000307 | 0.053662 | 0.012 | 4.725 |
| 5 | scipy_trf | measured | True | 0.001827 | 0.007621 | 347.601 | 0.860 |
| 5 | scipy_trf | failed_candidate | True | 0.000298 | 0.003787 | 341.601 | 0.859 |
| 5 | scipy_dogbox | measured | True | 0.000000 | 0.000000 | 3.499 | 0.862 |
| 5 | scipy_dogbox | failed_candidate | True | 0.000000 | 0.000000 | 2.773 | 0.859 |
| 5 | lula_ccd_bfgs | measured | False | 0.000719 | 0.125258 | 0.038 | 0.862 |
| 5 | lula_ccd_bfgs | failed_candidate | True | 0.001786 | 0.027212 | 0.009 | 0.859 |

## Limits

These are five frozen failure snapshots, not closed-loop rollouts or a performance benchmark. The last successful IK solution was not recorded; a failed-candidate restart is additional optimization, not a replay of the original solve. Valid kinematic solutions do not establish collision-free motion, physical tracking or assembly success. No training or controller settings were changed.

## Reproduce

```bash
/mnt/data2/isaacsim-5.0.0/python.sh runners/analyze_ik_failures.py --input outputs/runs/task1/dp_sac/train_batch_001/train --output <new-output-directory>
```

