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
