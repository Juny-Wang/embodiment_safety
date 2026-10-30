"""DirectRLEnvCfg for task1_dimm_insert Stage 1A.

Scene strategy:
- Franka Panda from Nucleus OR local robot_cache USD (see resolve_franka_usd_path).
- DIMM from teleop physics proxy via lab_dimm_dynamic.usda.
- Socket as Articulation via lab_socket_articulation.usda → teleop
  socket_franka_install_v008/socket_print_v007_installed.usda
  (FixedAssembly + LeftHinge/RightHinge ±40° hard stop).
- Latch angles read from articulation joint_pos (OBS dim stays 30).
- Contact: DIMM→Socket (axial/lateral), Latch→Robot/DIMM (press), Finger→DIMM.

Requires Isaac Sim Kit (pxr). Do not import this module from plain conda python.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from tasks.task1_dimm_insert.env.make_env import ACT_DIM, OBS_DIM

# Local Franka cache kept after Phase A/B cleanup (absolute path on isaac-b).
_LOCAL_FRANKA_USD = Path(
    "/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/"
    "franka_latch_open_test_v1/robot_cache/Isaac/IsaacLab/Robots/FrankaEmika/"
    "panda_instanceable.usd"
)

# Teleop / train-validated physics proxies (source of truth).
_ASSET_DIMM_PROXY = Path(
    "/mnt/data2/wjy_SERL/franka_sim/assets/task1_pcb_insert/usd/DIMM/"
    "2308107_4_matched_work/physics_contact_bench_v1/assets/"
    "dimm_2308107_matched_physics_proxy_pending.usda"
)
_ASSET_SOCKET_PROXY = Path(
    "/mnt/data2/wjy_SERL/franka_sim/assets/task1_pcb_insert/usd/DIMM/"
    "2308107_4_matched_work/physics_contact_bench_v1/assets/"
    "socket_2308107_4_physics_proxy_pending.usda"
)

# Lab wrappers (single defaultPrim Asset).
_ENV_ASSETS = Path(__file__).resolve().parent / "assets"
_LAB_DIMM_USD = _ENV_ASSETS / "lab_dimm_dynamic.usda"
# Prefer articulation wrap (latches); keep kinematic Base-only as fallback.
_LAB_SOCKET_ARTIC_USD = _ENV_ASSETS / "lab_socket_articulation.usda"
_LAB_SOCKET_USD = _ENV_ASSETS / "lab_socket_kinematic.usda"  # legacy Base-only
# Teleop MatchedSocket source (print_v007 / v008, ±40° hinges).
_ASSET_SOCKET_V008 = Path(
    "/mnt/data2/wjy_SERL/franka_sim/outputs/T1_rl/socket_franka_install_v008/"
    "socket_print_v007_installed.usda"
)


def resolve_franka_usd_path() -> str:
    """Prefer local cached USD; fall back to Nucleus token path."""
    if _LOCAL_FRANKA_USD.is_file():
        return str(_LOCAL_FRANKA_USD)
    try:
        from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

        return f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    except Exception:
        return str(_LOCAL_FRANKA_USD)


def resolve_dimm_usd_path() -> str:
    """Lab wrapper if present, else raw physics proxy (defaultPrim Asset)."""
    if _LAB_DIMM_USD.is_file():
        return str(_LAB_DIMM_USD)
    return str(_ASSET_DIMM_PROXY)


def resolve_socket_usd_path() -> str:
    """Lab articulation wrap (FixedAssembly + Left/Right hinges) preferred.

    Falls back to Base-only kinematic wrap only if articulation USD missing.
    Raw physics_proxy / v008 must not be spawned as RigidObject.
    """
    if _LAB_SOCKET_ARTIC_USD.is_file():
        return str(_LAB_SOCKET_ARTIC_USD)
    if _LAB_SOCKET_USD.is_file():
        return str(_LAB_SOCKET_USD)
    raise FileNotFoundError(
        f"Missing Lab socket wraps {_LAB_SOCKET_ARTIC_USD} / {_LAB_SOCKET_USD}; "
        f"v008 at {_ASSET_SOCKET_V008}; raw proxy at {_ASSET_SOCKET_PROXY}."
    )


def make_franka_cfg(prim_path: str = "/World/envs/env_.*/Robot") -> ArticulationCfg:
    """Franka with stiffer PD (IK-friendly) for Stage 1A skeleton."""
    usd = resolve_franka_usd_path()
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd,
            activate_contact_sensors=True,  # needed for finger ContactSensors
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=400.0,
                damping=80.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


def make_socket_cfg(prim_path: str = "/World/envs/env_.*/Socket") -> ArticulationCfg:
    """Socket articulation = teleop v008 FixedAssembly + LeftHinge/RightHinge.

    Hard limits from USD: Left -40..0 deg, Right 0..40 deg (PhysX USD degrees).
    Lab joint_pos is radians. Init at open15 (±15°) so dual_latch is not free at t=0.
    Actuators: viscous damping only (stiffness=0), matching teleop drive note.
    """
    # open15 in radians
    open15 = 0.2617993877991494
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=resolve_socket_usd_path(),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
                # Do NOT use fix_root_link=True: Lab moves ArticulationRoot to env_0,
                # nesting with Franka. World pin is USD PhysicsFixedJoint RootFixed.
                fix_root_link=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.20),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "LeftHinge": -open15,
                "RightHinge": open15,
            },
        ),
        actuators={
            "latches": ImplicitActuatorCfg(
                joint_names_expr=["LeftHinge", "RightHinge"],
                effort_limit_sim=0.05,
                stiffness=0.0,
                damping=0.001,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )


def make_dimm_cfg(prim_path: str = "/World/envs/env_.*/DIMM") -> RigidObjectCfg:
    """Dynamic DIMM = teleop physics proxy (gold fingers + PCB Main)."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=resolve_dimm_usd_path(),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=1.0,
                enable_gyroscopic_forces=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, 0.0, 0.28), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )


@configclass
class Task1DimmInsertSceneCfg(InteractiveSceneCfg):
    """Minimal interactive scene — Franka + real DIMM/socket physics proxies."""

    num_envs: int = 4
    env_spacing: float = 2.5
    replicate_physics: bool = True


@configclass
class Task1DimmInsertEnvCfg(DirectRLEnvCfg):
    """Stage 1A DirectRLEnv config — OBS 30 / ACT 7."""

    # env timing: dt=1/120, decimation=2 → policy dt ~1/60; 500 steps ≈ 8.33 s
    decimation = 2
    episode_length_s = 8.3333
    action_space = ACT_DIM  # 7
    observation_space = OBS_DIM  # 30
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    scene: Task1DimmInsertSceneCfg = Task1DimmInsertSceneCfg(
        num_envs=4, env_spacing=2.5, replicate_physics=True
    )

    robot: ArticulationCfg = make_franka_cfg()
    socket: ArticulationCfg = make_socket_cfg()
    dimm: RigidObjectCfg = make_dimm_cfg()

    # Contact sensors → OBS contact_forces[6]:
    #   [axial, lateral, left_press, right_press, finger_L, finger_R]
    # thresh-2: axial/lateral = DIMM↔Socket; press = external↔Latch; finger = finger↔DIMM.
    dimm_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/DIMM",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        force_threshold=0.01,
        filter_prim_paths_expr=["/World/envs/env_.*/Socket"],
    )
    latch_l_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Socket/LeftLatch",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        force_threshold=0.01,
        # external only (exclude Base↔Latch); match thresh-2 left_press_N
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot",
            "/World/envs/env_.*/DIMM",
        ],
    )
    latch_r_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Socket/RightLatch",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        force_threshold=0.01,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot",
            "/World/envs/env_.*/DIMM",
        ],
    )
    finger_l_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_leftfinger",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        force_threshold=0.01,
        filter_prim_paths_expr=["/World/envs/env_.*/DIMM"],
    )
    finger_r_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/panda_rightfinger",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        force_threshold=0.01,
        filter_prim_paths_expr=["/World/envs/env_.*/DIMM"],
    )

    # Control mode: "joint" (default, smoke-safe) or "diff_ik" (EE delta)
    control_mode: str = "joint"
    # Joint-position scales (normalized action [-1,1] → rad)
    joint_pos_scale: float = 0.05
    # EE delta scales (normalized action [-1,1] → meters / rad) — used when control_mode=diff_ik
    ee_pos_scale: float = 0.02
    ee_rot_scale: float = 0.05
    gripper_open: float = 0.04
    gripper_close: float = 0.0

    # Reward weights (REWARD_SPEC.md)
    w_approach: float = 1.0
    w_align: float = 1.0
    w_insert_depth: float = 2.0
    w_dual_latch: float = 5.0
    w_force_penalty: float = 0.1

    # Dense align scales (REWARD_DIAG fix#1) — NOT the success/termination tols.
    # Spawn Δxy~0.10m → xy/0.05 = 2; tilt~π → π/0.5 ≈ 6.3 then clip to [-5, 0].
    align_xy_scale_m: float = 0.05
    align_tilt_scale_rad: float = 0.5
    align_clip_min: float = -5.0
    align_clip_max: float = 0.0
    # Looser near-align band gating insert_depth / dual_latch dense bonuses.
    near_align_xy_m: float = 0.03
    near_align_tilt_rad: float = 0.25

    # Geometry tolerances — success / termination / Plan-D clearance only.
    xy_tol_m: float = 0.0025
    tilt_tol_rad: float = 0.035
    seat_depth_m: float = 0.01
    success_depth_m: float = 0.012

    # Curriculum prep (REWARD_DIAG fix#2) — defaults keep current free-DIMM smoke path.
    # When True in a future train cfg: disable DIMM gravity at reset and/or hold pose
    # near socket mouth. Not applied in _reset_idx yet (safe default = False).
    dimm_gravity_off: bool = False
    dimm_init_in_gripper: bool = False  # TODO: attach / spawn-in-hand curriculum

    # Safety limits (overridden from safety_config.json when present)
    latch_overforce_N: float = 3.0
    grip_force_limit_N: float = 15.0
    axial_contact_N: float = 5.0
    lateral_contact_N: float = 2.0

    # Observability of latch/force sources (updated by env at runtime)
    latch_source: str = "articulation_LeftHinge_RightHinge"
    contact_source: str = "contact_sensor_or_zero"

    franka_usd_resolved: str = ""
    dimm_usd_resolved: str = ""
    socket_usd_resolved: str = ""
    asset_notes: str = (
        "Franka: local robot_cache or Nucleus panda_instanceable.usd. "
        "DIMM: physics proxy dimm_2308107_matched_physics_proxy_pending (via lab_dimm_dynamic). "
        "Socket: lab_socket_articulation — Asset(ArticulationRoot only) + Base/LeftLatch/RightLatch; "
        "RootFixed pins Base (not Lab fix_root_link); LeftHinge/RightHinge ±40deg from v008; init open15. "
        "Latch angles: socket articulation joint_pos [rad]. "
        "Contact: DIMM→Socket axial/lateral; Latch→Robot/DIMM press; Finger→DIMM (panda_left/rightfinger). Soft-fail per sensor."
    )
