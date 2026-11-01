# Task1 Diffusion Policy behavioral cloning

Uses the unmodified official real-stanford/diffusion_policy implementation at
5ba07ac6661db573af695b419a7947ecb704690f, MIT license.
Upstream: https://github.com/real-stanford/diffusion_policy
Class: diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy.

Three actual 256x256 cameras, state31, observation history2, action horizon16,
one action executed per100ms. DDIM16 inference steps, cosine100 training schedule.
ResNet18 per camera (random initialization, GroupNorm), U-Net widths128/256/512.
No force input, scene, reward, or controller-threshold changes. GPU BC, not SAC.

Environment: .venv with system torch2.7.0+cu128/torchvision0.22.0;
install requirements only into this project-local venv. No user-site packages.
Reproduce source: git clone https://github.com/real-stanford/diffusion_policy.git vendor
then git -C vendor checkout 5ba07ac6661db573af695b419a7947ecb704690f.

Prepare: .venv/bin/python -B -m runners.prepare_diffusion --output outputs/datasets/task1/diffusion_sequences_v001
Train: .venv/bin/python -B -m algos.diffusion_policy.train --data outputs/datasets/task1/diffusion_sequences_v001 --output outputs/runs/task1/diffusion_policy/train_v001 --steps 20000
Eval: python3 -B -m runners.eval_diffusion --checkpoint outputs/runs/task1/diffusion_policy/train_v001/best.pt --output outputs/runs/task1/diffusion_policy/eval10_v001

Commands above run from repository root; use algos/diffusion_policy/.venv/bin/python.
Training exports best and last EMA/model/optimizer checkpoints every500steps.
Validation is fixed-noise action reconstruction over correlated temporal blocks,
not proof of full-task success. Full task evaluation uses fresh-process reset.
Raw historic commands were50ms feedback commands; first command at each100ms
observation boundary is the BC label. No averaged action, fabricated substep
observation, cross-gap sequence, or SAC replay eligibility is claimed.

## Versioned orientation correction (v002 diagnostic)
Raw Task1 recordings and actor output remain state31. DP alone can encode the
goal XYZW quaternion as the first two columns of its rotation matrix:
state33 = raw[:22] + R[:,0] + R[:,1] + raw[26:].
This is sign-invariant and continuous at the downward orientation (w=0).
Training and worker share representation.py. New checkpoints explicitly declare
task1_rgb_state33_rot6d_reward_history_v002 and cannot load into the old network.
Legacy checkpoints keep their original state31 preprocessing.

For an isolated same-interface opening fit:
algos/diffusion_policy/.venv/bin/python -B -m algos.diffusion_policy.train --data outputs/datasets/task1/diffusion_sequences_v001 --output outputs/runs/task1/diffusion_policy/native_opening_rot6d_v002/train --state-representation rotation6d33_v002 --scope native-opening --steps 2000

This selects only windows wholly inside ticks 0..509 of the verified native
100ms demonstration. Existing disjoint temporal validation blocks are preserved.
Historical 50ms subcommands are excluded from this diagnostic, not deleted.
coverage.json explicitly lists missing validation groups. In-sample fit probes
cover native 5-second time bins; these are time bins, not fabricated stage labels.
Reports include mm/s, deg/s, gripper mm/s and wrong-direction counts.
No independent held-out trajectory or complete-task success is claimed.
Use a fixed-budget last.pt for the subsequent bounded opening rollout.
Do not compare this short native-only fit to the old full-data training as a
controlled orientation-only ablation; data scope and training budget also differ.

## Direct-action target and complete-fit diagnostic
`--prediction-type sample` uses the official DDIM/DP clean-trajectory target;
`epsilon` remains the default for old checkpoints. Resume rejects target changes.
Physical action errors and closed-loop results must be compared; diffusion losses
from different targets are not comparable.

`--scope native-opening-fit-all` includes every contiguous 16-step window in
native opening ticks 0..509. Both named splits deliberately contain training
samples: this is a functional fitting diagnostic, NOT held-out validation.
Its coverage/config reports mark that fact. It does not establish generalization
and must not be used as the final training/test split.

During evaluation only, TASK1_DIAGNOSTIC_STOP_STAGE=both_open stops after the
existing measured both-open event (including cleared contact dwell). The report
still marks full_task_success=false. It cannot be enabled for online learning.
No geometry, physical gains, force thresholds, or reward conditions are changed.

### Small action ranges
`--action-range-eps 1e-8` fits the official action normalizer without treating
the observed 2.07e-5 / 3.92e-5 roll/pitch command ranges as unit-scale channels.
The old default (1e-4) remains checkpoint-compatible. State normalization,
7D velocity units, controller limits and physics are unchanged. New normalizer
contracts require independent training; resume refuses a different epsilon.
The checkpoint stores the normalizer; worker execution does not add a deadband,
orientation clamp, angle write, trajectory lookup or corrective scripted action.

## Complete Task1 BC fit with opening checks retained
The experiment config is configs/experiments/task1_diffusion_full_flow.json.
Scope native-full-fit-all selects every contiguous window from the verified
native 100ms trajectory, including stage handoffs and final release/hold.
Both fit probes contain training samples, explicitly not held-out validation.
Historical 50ms action labels are not silently treated as native transitions.
This trains one policy across the entire task; no scripted stage controller.

Full-task evaluation leaves TASK1_DIAGNOSTIC_STOP_STAGE empty so a verified
opening event does not end the episode. It retains per-transition opening
diagnostics, the insertion-path-clear condition and terminal opening history.
No scene, reward, safety threshold or controller change is made.
A failed opening is not relabeled as success. Full success still requires
measured seating, contact-driven closure, release, retraction and stable hold.
