# Task template

Copy this folder to `tasks/<task_id>/`.

## Required

- `scene/` — USD/USDA and asset references
- `env/` — Gym/Isaac env: obs/act, reward, done, reset (`make_env`)
- `configs/` — task-only: reward weights, success criteria, safety margins
- `teleop/` — optional human teleop for acceptance

## Do not put here

Training loops, PPO/IL code, or generic Franka drivers (those go in `algos/` / `lib/`).
