"""DirectRLEnv for task1_dimm_insert Stage 1A (OBS 30 / ACT 7).

Implements:
- Franka + real DIMM physics proxy + socket Articulation (v008 latches)
- Observations matching OBS_SPEC.md (dim 30 unchanged)
- Reward terms matching REWARD_SPEC.md
- ACT 7 = joint deltas (control_mode=joint, default) OR EE Δxyz+rpy + gripper (diff_ik)

Latch angles: LeftHinge / RightHinge joint_pos from socket Articulation [rad].
Contact forces[6]: DIMM→Socket axial/lateral; Latch→Robot/DIMM press; Finger→DIMM.
Requires Kit (pxr) — import only after AppLauncher.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import subtract_frame_transforms

from tasks.task1_dimm_insert.env.direct_env_cfg import (
    Task1DimmInsertEnvCfg,
    resolve_dimm_usd_path,
    resolve_franka_usd_path,
    resolve_socket_usd_path,
)
from tasks.task1_dimm_insert.env.lab_env import load_safety_limits
from tasks.task1_dimm_insert.env.make_env import ACT_DIM, OBS_DIM


def _flog(msg: str) -> None:
    """Flushing debug log for hang diagnosis (stdout + fsynced crumb file)."""
    ts = time.strftime("%H:%M:%S")
    line = f"[Task1EnvDBG {ts}] {msg}"
    print(line, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        with open("/tmp/embodied_kit_smoke/crumbs.txt", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:
        pass


class Task1DimmInsertEnv(DirectRLEnv):
    """Stage 1A DIMM insert DirectRLEnv with real physics-proxy assets."""

    cfg: Task1DimmInsertEnvCfg

    def __init__(self, cfg: Task1DimmInsertEnvCfg, render_mode: str | None = None, **kwargs):
        # Resolve paths before super() so spawn uses them; also stash for logs.
        cfg.franka_usd_resolved = resolve_franka_usd_path()
        cfg.dimm_usd_resolved = resolve_dimm_usd_path()
        cfg.socket_usd_resolved = resolve_socket_usd_path()
        cfg.robot.spawn.usd_path = cfg.franka_usd_resolved  # type: ignore[union-attr]
        cfg.dimm.spawn.usd_path = cfg.dimm_usd_resolved  # type: ignore[union-attr]
        cfg.socket.spawn.usd_path = cfg.socket_usd_resolved  # type: ignore[union-attr]

        super().__init__(cfg, render_mode, **kwargs)

        # Arm + finger joint ids
        self._arm_joint_ids, _ = self._robot.find_joints("panda_joint.*")
        self._finger_joint_ids, _ = self._robot.find_joints("panda_finger_joint.*")
        self._ee_body_ids, _ = self._robot.find_bodies("panda_hand")
        self._ee_body_idx = int(self._ee_body_ids[0])
        # Jacobian body index for Franka hand (fixed joint offset convention)
        self._ee_jacobi_idx = self._ee_body_idx - 1

        # Socket latch hinge joint ids (LeftHinge, RightHinge) — order L then R for OBS.
        self._latch_joint_ids: list[int] = []
        self._latch_joint_names: list[str] = []
        try:
            ids, names = self._socket.find_joints(["LeftHinge", "RightHinge"])
            # find_joints may return in discovery order; force L then R.
            name_to_id = {str(n): int(i) for i, n in zip(ids, names)}
            ordered = []
            ordered_names = []
            for want in ("LeftHinge", "RightHinge"):
                # names may be bare or path-suffixed
                hit = None
                for n, i in name_to_id.items():
                    if n == want or n.endswith(want) or want in n:
                        hit = (i, n)
                        break
                if hit is None:
                    raise KeyError(want)
                ordered.append(hit[0])
                ordered_names.append(hit[1])
            self._latch_joint_ids = ordered
            self._latch_joint_names = ordered_names
        except Exception as exc:
            _flog(f"latch joint resolve failed: {type(exc).__name__}: {exc}")
            self._latch_joint_ids = []
            self._latch_joint_names = []

        self._control_mode = str(getattr(cfg, "control_mode", "joint")).lower()
        self._ik = None
        if self._control_mode == "diff_ik":
            ik_cfg = DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            )
            self._ik = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)

        self._actions = torch.zeros(self.num_envs, ACT_DIM, device=self.device)
        self._joint_pos_des = torch.zeros(self.num_envs, self._robot.num_joints, device=self.device)
        self._dbg_step = 0
        # _contact_ok already set in _setup_scene during super().__init__
        if not hasattr(self, "_contact_ok"):
            self._contact_ok = False

        # Buffers for reward / obs intermediates
        self._dimm_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._dimm_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._socket_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._socket_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._rel_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._rel_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._latch_angles = torch.zeros(self.num_envs, 2, device=self.device)
        self._contact_forces = torch.zeros(self.num_envs, 6, device=self.device)
        self._insert_depth = torch.zeros(self.num_envs, device=self.device)

        # Apply safety JSON overrides when scene_path provided via cfg extras
        scene_path = getattr(cfg, "_scene_path_override", None)
        if scene_path:
            limits = load_safety_limits(Path(scene_path))
            for k in (
                "latch_overforce_N",
                "grip_force_limit_N",
                "axial_contact_N",
                "lateral_contact_N",
            ):
                if k in limits:
                    setattr(self.cfg, k, float(limits[k]))

        if len(self._latch_joint_ids) == 2:
            self.cfg.latch_source = "articulation_LeftHinge_RightHinge"
        else:
            self.cfg.latch_source = "heuristic_insert_depth_fallback"

        self.cfg.contact_source = "contact_sensor" if self._contact_ok else "zeros_pending"

        _flog(
            f"init done num_envs={self.num_envs} obs={OBS_DIM} act={ACT_DIM} "
            f"control_mode={self._control_mode} contact_ok={self._contact_ok} "
            f"arm_ids={list(self._arm_joint_ids)} finger_ids={list(self._finger_joint_ids)} "
            f"ee_body={self._ee_body_idx} jacobi={self._ee_jacobi_idx} "
            f"latch_ids={self._latch_joint_ids} latch_names={self._latch_joint_names}"
        )
        print(f"[Task1DimmInsertEnv] franka={self.cfg.franka_usd_resolved}", flush=True)
        print(f"[Task1DimmInsertEnv] dimm={self.cfg.dimm_usd_resolved}", flush=True)
        print(f"[Task1DimmInsertEnv] socket={self.cfg.socket_usd_resolved}", flush=True)
        print(f"[Task1DimmInsertEnv] asset_notes={self.cfg.asset_notes}", flush=True)
        print(
            f"[Task1DimmInsertEnv] latch_source={self.cfg.latch_source} "
            f"contact_source={self.cfg.contact_source} "
            f"contact_flags={getattr(self, '_contact_flags', {})}",
            flush=True,
        )

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._socket = Articulation(self.cfg.socket)
        self._dimm = RigidObject(self.cfg.dimm)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["socket"] = self._socket
        self.scene.rigid_objects["dimm"] = self._dimm

        # Contact sensors (soft-fail individually). See OBS contact_forces[6].
        self._dimm_contact: ContactSensor | None = None
        self._latch_l_contact: ContactSensor | None = None
        self._latch_r_contact: ContactSensor | None = None
        self._finger_l_contact: ContactSensor | None = None
        self._finger_r_contact: ContactSensor | None = None
        self._contact_flags: dict[str, bool] = {}

        def _reg(name: str, cfg_attr: str) -> ContactSensor | None:
            try:
                sens = ContactSensor(getattr(self.cfg, cfg_attr))
                self.scene.sensors[name] = sens
                self._contact_flags[name] = True
                _flog(f"ContactSensor {name} registered")
                return sens
            except Exception as exc:
                self._contact_flags[name] = False
                _flog(f"ContactSensor {name} skipped: {type(exc).__name__}: {exc}")
                return None

        self._dimm_contact = _reg("dimm_contact", "dimm_contact")
        self._latch_l_contact = _reg("latch_l_contact", "latch_l_contact")
        self._latch_r_contact = _reg("latch_r_contact", "latch_r_contact")
        self._finger_l_contact = _reg("finger_l_contact", "finger_l_contact")
        self._finger_r_contact = _reg("finger_r_contact", "finger_r_contact")
        self._contact_ok = bool(self._contact_flags.get("dimm_contact"))

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ----------------------------------------------------------- pre-physics
    def _pre_physics_step(self, actions: torch.Tensor):
        t0 = time.perf_counter()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        if self._dbg_step < 3:
            _flog(
                f"_pre_physics_step #{self._dbg_step} "
                f"actions_shape={tuple(self._actions.shape)} "
                f"dt={(time.perf_counter()-t0)*1e3:.2f}ms"
            )

    def _apply_action(self):
        t0 = time.perf_counter()
        dbg = self._dbg_step < 3
        if dbg:
            _flog(f"_apply_action ENTER #{self._dbg_step} mode={self._control_mode}")

        self._joint_pos_des[:] = self._robot.data.joint_pos

        if self._control_mode == "diff_ik":
            self._apply_action_diff_ik(dbg=dbg)
        else:
            self._apply_action_joint(dbg=dbg)

        t1 = time.perf_counter()
        if dbg:
            _flog("_apply_action set_joint_position_target …")
        self._robot.set_joint_position_target(self._joint_pos_des)
        if dbg:
            _flog(
                f"_apply_action EXIT #{self._dbg_step} "
                f"compute={(t1-t0)*1e3:.2f}ms total={(time.perf_counter()-t0)*1e3:.2f}ms"
            )
        self._dbg_step += 1

    def _apply_action_joint(self, dbg: bool = False):
        """Map ACT7 → 7 arm joint offsets from default (smoke-safe, no Jacobian).

        Gripper held open in joint mode for Stage1A smoke.
        """
        t0 = time.perf_counter()
        scale = float(getattr(self.cfg, "joint_pos_scale", 0.05))
        default = self._robot.data.default_joint_pos
        arm_tgt = default[:, self._arm_joint_ids] + self._actions[:, 0:7] * scale
        self._joint_pos_des[:, self._arm_joint_ids] = arm_tgt
        for fid in self._finger_joint_ids:
            self._joint_pos_des[:, fid] = self.cfg.gripper_open
        if dbg:
            _flog(
                f"  joint mode arm_tgt[0]={arm_tgt[0].detach().cpu().tolist()} "
                f"dt={(time.perf_counter()-t0)*1e3:.2f}ms"
            )

    def _apply_action_diff_ik(self, dbg: bool = False):
        """Original EE relative DifferentialIK path (may hang on jacobian)."""
        assert self._ik is not None
        t0 = time.perf_counter()
        ee_cmd = torch.zeros(self.num_envs, 6, device=self.device)
        ee_cmd[:, 0:3] = self._actions[:, 0:3] * self.cfg.ee_pos_scale
        ee_cmd[:, 3:6] = self._actions[:, 3:6] * self.cfg.ee_rot_scale
        if dbg:
            _flog(f"  diff_ik ee_cmd ready dt={(time.perf_counter()-t0)*1e3:.2f}ms")

        ee_pose_w = self._robot.data.body_pose_w[:, self._ee_body_idx]
        root_pose_w = self._robot.data.root_pose_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )
        if dbg:
            _flog(f"  diff_ik ee_pose_b done dt={(time.perf_counter()-t0)*1e3:.2f}ms")

        self._ik.set_command(ee_cmd, ee_pos_b, ee_quat_b)
        if dbg:
            _flog("  diff_ik set_command done; fetching jacobians …")

        jac = self._robot.root_physx_view.get_jacobians()[
            :, self._ee_jacobi_idx, :, self._arm_joint_ids
        ]
        if dbg:
            _flog(
                f"  diff_ik jacobians shape={tuple(jac.shape)} "
                f"dt={(time.perf_counter()-t0)*1e3:.2f}ms"
            )

        joint_pos = self._robot.data.joint_pos[:, self._arm_joint_ids]
        joint_pos_des = self._ik.compute(ee_pos_b, ee_quat_b, jac, joint_pos)
        if dbg:
            _flog(f"  diff_ik compute done dt={(time.perf_counter()-t0)*1e3:.2f}ms")

        self._joint_pos_des[:, self._arm_joint_ids] = joint_pos_des

        g = (self._actions[:, 6:7] + 1.0) * 0.5
        finger = self.cfg.gripper_open * (1.0 - g) + self.cfg.gripper_close * g
        for fid in self._finger_joint_ids:
            self._joint_pos_des[:, fid] = finger.squeeze(-1)

    # ---------------------------------------------------------- post-physics
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_intermediates()
        seated = self._insert_depth >= self.cfg.success_depth_m
        xy_ok = torch.linalg.norm(self._rel_pos[:, 0:2], dim=-1) < self.cfg.xy_tol_m * 2.0
        latched = (self._latch_angles.abs() < 0.05).all(dim=-1)
        success = seated & xy_ok & latched

        axial = self._contact_forces[:, 0]
        lateral = self._contact_forces[:, 1]
        latch_f = self._contact_forces[:, 2:4].abs().max(dim=-1).values
        grip_f = self._contact_forces[:, 4:6].abs().max(dim=-1).values
        overforce = (
            (axial.abs() > self.cfg.axial_contact_N)
            | (lateral.abs() > self.cfg.lateral_contact_N)
            | (latch_f > self.cfg.latch_overforce_N)
            | (grip_f > self.cfg.grip_force_limit_N)
        )

        terminated = success | overforce
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        """Dense reward — REWARD_DIAG fix#1: cm-scale align + near-align gates.

        Success/termination still use ``xy_tol_m`` / ``tilt_tol_rad`` (mm/deg gates).
        Dense align uses ``align_xy_scale_m`` / ``align_tilt_scale_rad`` then clips.
        """
        self._refresh_intermediates()
        dist = torch.linalg.norm(self._rel_pos, dim=-1)
        approach = -dist

        xy_err = torch.linalg.norm(self._rel_pos[:, 0:2], dim=-1)
        # Lab quat is wxyz; relative tilt from |w|.
        w = self._rel_quat[:, 0].abs().clamp(0.0, 1.0)
        tilt = 2.0 * torch.acos(w)

        # Dense align: cm-scale / ~soft-rad scale (NOT success tols), then clip.
        xy_scale = max(float(self.cfg.align_xy_scale_m), 1e-6)
        tilt_scale = max(float(self.cfg.align_tilt_scale_rad), 1e-6)
        align_raw = -(xy_err / xy_scale) - (tilt / tilt_scale)
        align = align_raw.clamp(
            min=float(self.cfg.align_clip_min),
            max=float(self.cfg.align_clip_max),
        )

        # Near-align band (looser than success tols) gates depth / latch bonuses.
        near = (xy_err < float(self.cfg.near_align_xy_m)) & (
            tilt < float(self.cfg.near_align_tilt_rad)
        )
        near_f = near.to(dtype=torch.float32)
        insert_depth = self._insert_depth * near_f
        dual_latch = (
            (self._latch_angles.abs() < 0.05).to(dtype=torch.float32).sum(dim=-1) * near_f
        )

        force_mag = self._contact_forces.abs().sum(dim=-1)
        force_penalty = -force_mag

        reward = (
            self.cfg.w_approach * approach
            + self.cfg.w_align * align
            + self.cfg.w_insert_depth * insert_depth
            + self.cfg.w_dual_latch * dual_latch
            + self.cfg.w_force_penalty * force_penalty
        )

        self.extras["log"] = {
            "approach": (self.cfg.w_approach * approach).mean(),
            "align": (self.cfg.w_align * align).mean(),
            "insert_depth": (self.cfg.w_insert_depth * insert_depth).mean(),
            "dual_latch": (self.cfg.w_dual_latch * dual_latch).mean(),
            "force_penalty": (self.cfg.w_force_penalty * force_penalty).mean(),
            "reward": reward.mean(),
            # Raw geometry for TB / next-train diagnosis (unweighted means).
            "xy_err": xy_err.mean(),
            "tilt": tilt.mean(),
            "align_raw": align_raw.mean(),
            "near_align_frac": near_f.mean(),
        }
        self.extras["reward_terms"] = {
            "approach": approach,
            "align": align,
            "insert_depth": insert_depth,
            "dual_latch": dual_latch,
            "force_penalty": force_penalty,
            "xy_err": xy_err,
            "tilt": tilt,
            "near_align": near,
        }
        return reward

    def _get_observations(self) -> dict:
        self._refresh_intermediates()

        jp = self._robot.data.joint_pos[:, self._arm_joint_ids]
        jv = self._robot.data.joint_vel[:, self._arm_joint_ids]

        fingers = self._robot.data.joint_pos[:, self._finger_joint_ids]
        grip = fingers.sum(dim=-1, keepdim=True)

        obs = torch.cat(
            (
                jp,
                jv,
                grip,
                self._rel_pos,
                self._rel_quat,
                self._latch_angles,
                self._contact_forces,
            ),
            dim=-1,
        )
        assert obs.shape[-1] == OBS_DIM, f"obs dim {obs.shape[-1]} != {OBS_DIM}"
        return {"policy": obs}

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        sock_pose = self._socket.data.default_root_state[env_ids].clone()
        sock_pose[:, 0:3] += self.scene.env_origins[env_ids]
        self._socket.write_root_pose_to_sim(sock_pose[:, 0:7], env_ids=env_ids)
        self._socket.write_root_velocity_to_sim(
            torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids
        )
        # Reset latch hinges to default (open15) joint state.
        sock_jpos = self._socket.data.default_joint_pos[env_ids].clone()
        sock_jvel = torch.zeros_like(sock_jpos)
        self._socket.write_joint_state_to_sim(sock_jpos, sock_jvel, env_ids=env_ids)

        dimm_pose = self._dimm.data.default_root_state[env_ids].clone()
        noise = (torch.rand(len(env_ids), 2, device=self.device) - 0.5) * 0.02
        dimm_pose[:, 0] += noise[:, 0]
        dimm_pose[:, 1] += noise[:, 1]
        dimm_pose[:, 0:3] += self.scene.env_origins[env_ids]
        self._dimm.write_root_pose_to_sim(dimm_pose[:, 0:7], env_ids=env_ids)
        self._dimm.write_root_velocity_to_sim(
            torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids
        )

        if self._ik is not None:
            self._ik.reset(env_ids)
        self._refresh_intermediates(env_ids)

    # --------------------------------------------------------------- helpers
    def _refresh_intermediates(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self._dimm_pos_w[env_ids] = self._dimm.data.root_pos_w[env_ids]
        self._dimm_quat_w[env_ids] = self._dimm.data.root_quat_w[env_ids]
        self._socket_pos_w[env_ids] = self._socket.data.root_pos_w[env_ids]
        self._socket_quat_w[env_ids] = self._socket.data.root_quat_w[env_ids]

        rel_pos, rel_quat = subtract_frame_transforms(
            self._socket_pos_w[env_ids],
            self._socket_quat_w[env_ids],
            self._dimm_pos_w[env_ids],
            self._dimm_quat_w[env_ids],
        )
        self._rel_pos[env_ids] = rel_pos
        self._rel_quat[env_ids] = rel_quat

        self._insert_depth[env_ids] = (-rel_pos[:, 2]).clamp(min=0.0)

        # Latch angles from socket Articulation hinges (rad). Fallback = heuristic.
        if len(self._latch_joint_ids) == 2:
            jp = self._socket.data.joint_pos[env_ids][:, self._latch_joint_ids]
            self._latch_angles[env_ids] = jp
            self.cfg.latch_source = "articulation_LeftHinge_RightHinge"
        else:
            depth = self._insert_depth[env_ids]
            closed_frac = (depth / max(self.cfg.seat_depth_m, 1e-6)).clamp(0.0, 1.0)
            ang = 0.4 * (1.0 - closed_frac)
            self._latch_angles[env_ids, 0] = ang
            self._latch_angles[env_ids, 1] = ang
            self.cfg.latch_source = "heuristic_insert_depth_fallback"

        self._fill_contact_forces(env_ids)

    def _fill_contact_forces(self, env_ids: torch.Tensor) -> None:
        """Map ContactSensors into OBS contact block [6].

        Layout: [axial_N, lateral_N, left_press_N, right_press_N, finger_L_N, finger_R_N]
          axial/lateral — DIMM→Socket net force (world z / xy hypot)
          left/right press — ||F|| Latch→{Robot,DIMM} (thresh-2 external↔Latch)
          finger L/R — ||F|| panda_{left,right}finger→DIMM
        """
        self._contact_forces[env_ids] = 0.0
        parts: list[str] = []

        # --- axial / lateral from DIMM→Socket ---
        mag_ax = self._sensor_force_vec(self._dimm_contact, env_ids)
        if mag_ax is not None:
            axial = mag_ax[:, 2]
            lateral = torch.linalg.norm(mag_ax[:, 0:2], dim=-1)
            self._contact_forces[env_ids, 0] = axial
            self._contact_forces[env_ids, 1] = lateral
            parts.append("dimm_socket")

        # --- latch press ---
        lp = self._sensor_force_mag(self._latch_l_contact, env_ids)
        if lp is not None:
            self._contact_forces[env_ids, 2] = lp
            parts.append("latch_L")
        rp = self._sensor_force_mag(self._latch_r_contact, env_ids)
        if rp is not None:
            self._contact_forces[env_ids, 3] = rp
            parts.append("latch_R")

        # --- finger vs DIMM ---
        fl = self._sensor_force_mag(self._finger_l_contact, env_ids)
        if fl is not None:
            self._contact_forces[env_ids, 4] = fl
            parts.append("finger_L")
        fr = self._sensor_force_mag(self._finger_r_contact, env_ids)
        if fr is not None:
            self._contact_forces[env_ids, 5] = fr
            parts.append("finger_R")

        if parts:
            self.cfg.contact_source = "+".join(parts)
            self._contact_ok = True
        else:
            self.cfg.contact_source = "zeros_no_sensor"

    def _sensor_force_vec(
        self, sensor: ContactSensor | None, env_ids: torch.Tensor
    ) -> torch.Tensor | None:
        """Net contact force vector (N, 3) for selected envs, or None."""
        if sensor is None:
            return None
        try:
            data = sensor.data
            # Prefer filtered force matrix (sum over bodies × filters) when present.
            fm = getattr(data, "force_matrix_w", None)
            if fm is not None:
                # (N, B, M, 3) → (N, 3)
                return fm[env_ids].sum(dim=(1, 2))
            forces = data.net_forces_w
            if forces is None:
                return None
            if forces.ndim == 3:
                return forces[env_ids].sum(dim=1)
            return forces[env_ids]
        except Exception as exc:
            if self._dbg_step < 3:
                _flog(f"sensor force_vec failed: {type(exc).__name__}: {exc}")
            return None

    def _sensor_force_mag(
        self, sensor: ContactSensor | None, env_ids: torch.Tensor
    ) -> torch.Tensor | None:
        """Scalar ||F|| (N,) for selected envs, or None."""
        vec = self._sensor_force_vec(sensor, env_ids)
        if vec is None:
            return None
        return torch.linalg.norm(vec, dim=-1)

