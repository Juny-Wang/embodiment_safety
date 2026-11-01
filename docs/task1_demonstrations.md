# Task1 successful demonstrations

Task1 demonstrations execute the full latch-opening, grasp, transfer, insertion,
release, and withdrawal sequence in the native Isaac Sim scene. The feedback
expert uses measured task geometry and contacts to select actions. BC inputs
remain the three RGB views and robot proprioception; task truth is teacher-only.

Each attempt starts a new simulator process. Only physically successful,
data-valid, smooth, distinct episodes enter `manifest.json`.

## Timing and motion

- Physics: 240 Hz. Actions and recorded transitions: 10 Hz. IK: 20 Hz.
- Translational actions, rotational actions, and gripper commands retain the
  existing normalized seven-dimensional action interface.
- Free-space paths use a minimum-jerk interpolation plus one smooth seeded bend.
  Horizontal bend offsets are within 15 mm; vertical offsets are 3–12 mm.
  Path endpoint velocity and acceleration are zero.
- The `smooth_fast_v002` free-space speed limits are 0.10 m/s for approaches and
  0.08 m/s for transfer. Translational acceleration remains limited to 0.04 m/s².
- During grasp closure without detected contact, the aperture closes at 4 mm/s
  above 4 mm, 1.5 mm/s above 2.5 mm, and 0.5 mm/s below that. Detected contact
  uses the existing force feedback and approximately 3 N target.
- Contact motions, alignment, insertion, and post-release stability criteria use
  the existing qualified expert settings.

## Collect

From `/mnt/data2/wjy_SERL/Embodied_Safety` on `isaac-b`:

```bash
algos/diffusion_policy/.venv/bin/python -B -m runners.collect_task1 \
  --output outputs/datasets/task1/smooth_success100_v002 \
  --seed 2026110204 --workers 2 --successes 3 --max-attempts 3
```

Compare the pilot against the slow reference:

```bash
algos/diffusion_policy/.venv/bin/python -B -m runners.compare_task1_collections \
  --baseline outputs/datasets/task1/smooth_success100_v001 \
  --candidate outputs/datasets/task1/smooth_success100_v002
```

The pilot passes when all three attempts are accepted, all three seeds have
successful slow references, mean duration decreases by at least 10%, and each
contact channel's mean peak is at most 1.2 times the reference plus 0.1 N.
This comparison measures demonstration quality and speed, not BC policy success.

Continue an accepted pilot to 100 successful episodes:

```bash
algos/diffusion_policy/.venv/bin/python -B -m runners.collect_task1 \
  --output outputs/datasets/task1/smooth_success100_v002 \
  --seed 2026110204 --workers 2 --resume --successes 100 --max-attempts 200
```

## Dataset and acceptance

`summary.json` records attempts and progress. `manifest.json` contains accepted
episodes only. Each entry links to raw `transitions.jsonl`, RGB `images/`,
`metadata.json`, and `validation.json`. The attempt's `acceptance.json` records
task success, reward stages, smoothness, and distance from previously accepted
paths. Failed attempts are retained outside the accepted manifest.

Acceptance requires all five task reward stages, a successful native terminal
result, valid data, and the existing contact limits. Within each sufficiently
sampled phase, the RMS residual from a centered five-sample velocity average
must not exceed 5 mm/s translationally or 0.05 rad/s rotationally. Free-space
path length divided by endpoint chord length must not exceed 1.5, using a 1 cm
denominator floor. Four free-space phases are sampled at equal arc-length
intervals; a new path must differ from every accepted path by at least 0.5 mm
RMS across those samples.

Render a complete three-view video with `runners.render_demonstration --episode
<episode-directory> --output <video.mp4>`. Video playback follows simulation time
at 10 fps. Real collection wall time can be longer than the video.
