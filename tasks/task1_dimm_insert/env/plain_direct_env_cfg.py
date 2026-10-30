"""DirectRLEnvCfg for Task1 **plain** PPO (task completion only).

Inherits Stage 1A scene / assets / obs. Differs only in reward weights and
flags consumed by ``Task1PlainInsertEnv``:

- ``w_force_penalty = 0`` (no soft force cost)
- ``w_success`` sparse bonus on insert success
- ``disable_safety_terminate = True`` (over-force does NOT end the episode)

Safety sensors may still fill OBS contact channels; they do not drive reward
or termination in this trainer.

GUI / demo collection: viewer is framed on env_0 DIMM+socket workspace so the
confirmed Task1 assets are visible (not the default 7.5 m world eye that only
shows an arm tip). Scene authority for teleop visuals/safety JSON:
``tasks/task1_dimm_insert/scene/manual_franka_scene.usda``.
"""

from __future__ import annotations

from isaaclab.envs import ViewerCfg
from isaaclab.utils import configclass

from tasks.task1_dimm_insert.env.direct_env_cfg import Task1DimmInsertEnvCfg


@configclass
class Task1PlainInsertEnvCfg(Task1DimmInsertEnvCfg):
    """Plain task-completion cfg — Stage1A defaults left untouched."""

    # Zero safety soft cost; keep geometric shaping for learning.
    w_force_penalty: float = 0.0
    # Sparse bonus when seated + xy_ok + dual latch closed.
    w_success: float = 20.0
    # Stronger early XY/tilt shaping so policy reaches near-align / success.
    w_align: float = 2.5
    w_approach: float = 1.5
    near_align_xy_m: float = 0.04
    # Hard over-force / latch-break terminate OFF for this trainer.
    disable_safety_terminate: bool = True

    # GUI: look at socket/DIMM in env_0 (matches teleop workspace framing).
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.05, 0.85, 0.70),
        lookat=(0.50, 0.0, 0.24),
        origin_type="env",
        env_index=0,
    )

    asset_notes: str = (
        "PLAIN PPO / SERL demos: same Lab assets as Stage1A = teleop MatchedDIMM "
        "physics proxy (+ visual mesh) and lab_socket_articulation (v008 hinges). "
        "Scene authority: tasks/task1_dimm_insert/scene/manual_franka_scene.usda. "
        "Reward = approach/align/depth/latch + success bonus. NO force_penalty. "
        "NO overforce terminate. Contact sensors may still appear in OBS but are "
        "not training constraints."
    )
