# Algo template

Copy to `algos/<algo_name>/`.

Expose a stable entry used by `runners` (e.g. `train(env, cfg)` / `Runner` class).
Do not import a concrete task module; take the env from the runner.
