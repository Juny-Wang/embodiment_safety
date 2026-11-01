"""Feedback demonstrator with seeded smooth variation in free-space transit."""
import numpy as np
from scipy.spatial.transform import Rotation
from .smooth_path import SmoothPath


class Task1Expert:
    PROFILE = "smooth_slow_v001"
    TRANSIT = {"LEFT_ABOVE": .05, "RIGHT_ABOVE": .05, "GRASP_ABOVE": .05, "TRANSFER": .04}

    def __init__(self, panel, seed, variation=1.):
        from pxr import UsdGeom
        self.panel = panel
        self.seed = seed
        self.variation = variation
        self.rng = np.random.default_rng(seed)
        self.done = False
        self.error = None
        self.start = panel.t
        self.enter = panel.t
        self.state = "LEFT_ABOVE"
        self.target = np.array([-.010437, .055, .94])
        self.initialR = panel.controller.pose()[1].copy()
        self.graspR = Rotation.from_rotvec([0, 0, np.pi/2]).as_matrix() @ self.initialR
        self.trace = []
        self.plans = []
        self.stable = None
        self.relative = None
        self.velocity = np.zeros(3)
        self.last_action_t = panel.t
        self.release_reference = None
        mat = UsdGeom.XformCache().GetLocalToWorldTransform(panel.stage.GetPrimAtPath("/World/MatchedSocket/Asset"))
        self.sp = np.array(mat.ExtractTranslation())
        self.sR = np.array(mat)[:3, :3].T
        self.entry = self.sp[2]+.0068
        self.seated = self.sp[2]+.0003963
        self.dimm0 = panel.bodies["dimm"].get_transforms()[0, :3].copy()
        self.path = None
        self.previous_velocity = np.zeros(3)
        self.change("LEFT_ABOVE", self.target)

    def change(self, state, target=None):
        self.state = state
        self.enter = self.panel.t
        self.stable = None
        if target is not None:
            self.target = np.array(target, dtype=float)
        self.trace.append(dict(time=self.panel.t, state=state))
        self.path = None
        if state in self.TRANSIT:
            start = np.array(self.panel.controller.pose()[0])
            offset = self.rng.uniform([-.015, -.015, .003], [.015, .015, .012])*self.variation
            self.path = SmoothPath(start, self.target, offset, self.TRANSIT[state])
            self.plans.append(dict(stage=state, start_time=self.enter, **self.path.description()))

    def action(self, cfg):
        stage = self.state
        a = self._contact_action(None, cfg)
        if self.error or self.done:
            return np.zeros(7)
        c = self.panel.controller
        pos, matrix = c.pose()
        if stage == self.state and self.path is not None:
            target, feedforward = self.path.sample(self.panel.t-self.enter)
            velocity = feedforward+2*(target-np.array(pos))
            speed = np.linalg.norm(velocity)
            if speed > self.TRANSIT[stage]:
                velocity *= self.TRANSIT[stage]/speed
            delta = velocity-self.previous_velocity
            change = np.linalg.norm(delta)
            if change > .004:
                velocity = self.previous_velocity+delta*(.004/change)
            a[:3] = velocity/cfg["translation_m_s"]
        dimm = self.panel.bodies["dimm"].get_transforms()[0, :3]
        if self.state in ("INSERT", "SEATED_REVIEW") and dimm[2] <= self.sp[2]+.0005:
            if self.state == "SEATED_REVIEW":
                a[:] = 0.
            velocity = a[:3]*cfg["translation_m_s"]-(np.asarray(c.goal[0])-pos)/.1
            angular = a[3:6]*cfg["rotation_rad_s"]+(Rotation.from_matrix(matrix)*c.goal[1].inv()).as_rotvec()/.1
            velocity *= min(1., .001/max(np.linalg.norm(velocity), 1e-12))
            angular *= min(1., .01/max(np.linalg.norm(angular), 1e-12))
            a[:3] = velocity/cfg["translation_m_s"]
            a[3:6] = angular/cfg["rotation_rad_s"]
        self.previous_velocity = a[:3]*cfg["translation_m_s"]
        return a

    def setup_button(self, side):
        from pxr import UsdGeom, Gf
        self.side = side
        finger = 'right' if side == 'Left' else 'left'
        b = UsdGeom.BBoxCache(0, ['default', 'render', 'proxy']).ComputeWorldBound(self.panel.stage.GetPrimAtPath('/World/Robot/panda_' + finger + 'finger')).ComputeAlignedRange()
        self.offset = np.array(self.panel.controller.pose()[0]) - np.array([(b.GetMin()[0] + b.GetMax()[0]) / 2, (b.GetMin()[1] + b.GetMax()[1]) / 2, b.GetMin()[2]])
        self.lp = self.panel.stage.GetPrimAtPath('/World/MatchedSocket/Asset/' + side + 'Latch')
        self.button = np.array([-0.0104365, 0.110199 if side == 'Left' else -0.041941, 0.785974])
        self.bl = UsdGeom.XformCache().GetLocalToWorldTransform(self.lp).GetInverse().Transform(Gf.Vec3d(*map(float, self.button)))
        self.contact_target = self.button + self.offset + [0, 0, 0.002]
        self.change(side.upper() + '_APPROACH', self.contact_target)

    def _contact_action(self, keys, cfg):
        from pxr import UsdGeom
        a = np.zeros(7)
        if self.error or self.done:
            return a
        (pos, R) = self.panel.controller.pose()
        pos = np.array(pos)
        tf = self.panel.bodies['dimm'].get_transforms()[0].copy()
        d = tf[:3].astype(float)
        dR = Rotation.from_quat(tf[3:]).as_matrix()
        metrics = self.panel.task_metrics({k: v.get_transforms()[0].copy() for (k, v) in self.panel.bodies.items()})
        if self.panel.t - self.enter > 100:
            self.error = 'stage_timeout:' + self.state
            return a
        v = (self.target - pos) * 2
        speed = 0.05
        rot = np.zeros(3)
        vals = [float(self.panel.contact_metrics.get('finger_' + s + '_N', 0)) for s in ('left', 'right')]
        width = float(sum(self.panel.controller.get_joint_positions()[self.panel.controller.fingers]))
        holding = self.state in ('GRIP', 'LIFT', 'TRANSFER', 'ALIGN', 'INSERT', 'SEATED_REVIEW')
        if holding:
            if self.panel.contact_metrics.get('forbidden_contact', 0):
                self.error = 'forbidden_grasp'
                return a
            if min(vals) < 0.2:
                if max(vals) > 3.5:
                    self.error = 'unilateral_grasp'
                    return a
                if width < 0.0005:
                    self.error = 'empty_grasp'
                    return a
                a[6] = (-0.0015 if width > 0.004 and max(vals) < 0.1 else -0.0005) / cfg['gripper_m_s']
            else:
                a[6] = float(np.clip(-(3 - np.mean(vals)) * 2e-05, -0.0001, 0.0001)) / cfg['gripper_m_s']
            if self.relative is not None and np.linalg.norm(R.T @ (d - pos) - self.relative) > 0.002:
                self.error = 'grasp_slip_over_2mm'
                return np.zeros(7)
        if self.state == 'LEFT_ABOVE':
            if np.linalg.norm(self.target - pos) < 0.001:
                self.setup_button('Left')
        elif self.state in ('LEFT_APPROACH', 'RIGHT_APPROACH'):
            speed = 0.01
            if np.linalg.norm(self.target - pos) < 0.0008:
                self.change(self.side.upper() + '_PRESS')
        elif self.state in ('LEFT_PRESS', 'RIGHT_PRESS'):
            side = self.side.lower()
            angle = metrics[side + '_angle_deg']
            if angle <= -15 if side == 'left' else angle >= 15:
                self.change(self.side.upper() + '_RETRACT', pos + [0, 0, 0.035])
                v = np.zeros(3)
            else:
                button = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(self.lp).Transform(self.bl))
                v = np.array([2 * (button[0] + self.offset[0] - pos[0]), 2 * (button[1] + self.offset[1] - pos[1]), -0.001])
                speed = 0.0015
                if pos[2] < self.contact_target[2] - 0.012:
                    self.error = 'press_travel_limit'
        elif self.state == 'LEFT_RETRACT':
            if np.linalg.norm(self.target - pos) < 0.001:
                self.setup_button('Right')
                t = self.target.copy()
                t[2] = 0.94
                self.change('RIGHT_ABOVE', t)
        elif self.state == 'RIGHT_ABOVE':
            if np.linalg.norm(self.target - pos) < 0.001:
                self.change('RIGHT_APPROACH', self.contact_target)
        elif self.state == 'RIGHT_RETRACT':
            if np.linalg.norm(self.target - pos) < 0.001:
                self.change('GRASP_ABOVE', [d[0], d[1], 0.97])
        elif self.state in ('GRASP_ABOVE', 'GRASP_LOWER'):
            if self.state == 'GRASP_ABOVE' and width > 0.01:
                a[6] = -0.006 / cfg['gripper_m_s']
                if np.linalg.norm(self.target - pos) < 0.001:
                    v[:] = 0
            rot = Rotation.from_matrix(self.graspR @ R.T).as_rotvec()
            if np.linalg.norm(rot) > 0.2:
                rot *= 0.2 / np.linalg.norm(rot)
            if self.state == 'GRASP_ABOVE' and np.linalg.norm(self.target - pos) < 0.0005 and (np.linalg.norm(rot) < 0.025) and (width <= 0.01):
                self.change('GRASP_LOWER', [d[0], d[1], d[2] + 0.12873])
            if self.state == 'GRASP_LOWER':
                speed = 0.02 if pos[2] - self.target[2] > 0.005 else 0.006
                if np.linalg.norm(self.target - pos) < 0.00025:
                    self.change('GRIP', pos)
        elif self.state == 'GRIP':
            v[:] = 0
            if min(vals) >= 1 and 2.5 <= np.mean(vals) <= 3.5:
                if self.stable is None:
                    self.stable = self.panel.t
                if self.panel.t - self.stable >= 0.5:
                    self.relative = R.T @ (d - pos)
                    self.change('LIFT', pos + [0, 0, 0.06])
            else:
                self.stable = None
        elif self.state == 'LIFT':
            speed = 0.002 if d[2] - self.dimm0[2] < 0.005 else 0.01
            if np.linalg.norm(self.target - pos) < 0.0003:
                if d[2] - self.dimm0[2] < 0.055:
                    self.error = 'insufficient_rack_clearance'
                else:
                    self.change('TRANSFER', pos + np.array([self.sp[0] - d[0], self.sp[1] - d[1], 0]))
        elif self.state == 'TRANSFER':
            speed = 0.04
            if np.linalg.norm(self.target - pos) < 0.0003:
                self.change('ALIGN', pos)
                self.alignz = d[2]
        elif self.state in ('ALIGN', 'INSERT'):
            rot = Rotation.from_matrix(self.sR @ dR.T).as_rotvec()
            ang = np.linalg.norm(rot)
            xy = np.linalg.norm(d[:2] - self.sp[:2])
            if ang > 0.005:
                rot *= 0.005 / ang
            v = np.array([*0.8 * (self.sp[:2] - d[:2]), 0.8 * (self.alignz - d[2])])
            speed = 0.001
            if self.state == 'ALIGN':
                if xy < 0.00015 and ang < np.deg2rad(0.15):
                    if self.stable is None:
                        self.stable = self.panel.t
                    if self.panel.t - self.stable >= 0.5:
                        self.change('INSERT')
                else:
                    self.stable = None
            else:
                if xy > 0.0004 or ang > np.deg2rad(0.3):
                    self.error = 'insertion_misalignment'
                force = float(self.panel.contact_metrics.get('dimm_environment_N', 0))
                v[2] = -0.003 if d[2] > self.entry + 0.003 else -float(np.clip((4.0 - force) * 0.00025, 0.0001, 0.001))
                speed = 0.003 if d[2] > self.entry + 0.003 else 0.001
                if d[2] <= self.sp[2] + 0.0001 and max((abs(metrics[k + '_angle_deg']) for k in ('left', 'right'))) <= 3.5:
                    self.change('SEATED_REVIEW', pos)
                    v[:] = 0
                if d[2] < self.sp[2] - 8e-05:
                    self.error = 'insertion_depth_limit'
        elif self.state == 'SEATED_REVIEW':
            v[:] = 0
            if self.panel.t - self.enter >= 0.5:
                self.release_reference = tf.copy()
                self.change('RELEASE', pos)
                self.release_hand = pos.copy()
        elif self.state in ('RELEASE', 'FREE_HOLD', 'RETRACT', 'FINAL_HOLD'):
            v[:] = 0
            displacement = float(np.linalg.norm(d - self.release_reference[:3]))
            angle_change = float(np.degrees((Rotation.from_quat(tf[3:]) * Rotation.from_quat(self.release_reference[3:]).inv()).magnitude()))
            if max((abs(metrics[k + '_angle_deg']) for k in ('left', 'right'))) > 3.5:
                self.error = 'release_angle_outside_3p5deg'
            if displacement > 0.0005 or angle_change > 0.5:
                self.error = 'release_instability'
            if self.state == 'RELEASE':
                if width < 0.0198:
                    a[6] = 0.004 / cfg['gripper_m_s']
                elif max(vals) < 0.1:
                    self.change('FREE_HOLD', pos)
            elif self.state == 'FREE_HOLD' and self.panel.t - self.enter >= 2:
                self.change('RETRACT', self.release_hand + [0, 0, 0.03])
            elif self.state == 'RETRACT':
                v = (self.target - pos) * 2
                speed = 0.02
                if np.linalg.norm(self.target - pos) < 0.0005:
                    self.change('FINAL_HOLD', pos)
            elif self.state == 'FINAL_HOLD' and self.panel.t - self.enter >= 2:
                self.change('COMPLETE_PENDING_AUDIT', pos)
        elif self.state == 'COMPLETE_PENDING_AUDIT':
            v[:] = 0
        norm = np.linalg.norm(v)
        if norm > speed:
            v *= speed / norm
        if self.state in ('LIFT', 'TRANSFER'):
            dt = max(0.001, self.panel.t - self.last_action_t)
            dv = v - self.velocity
            n = np.linalg.norm(dv)
            if n > 0.04 * dt:
                v = self.velocity + dv * (0.04 * dt / n)
        self.velocity = v.copy()
        self.last_action_t = self.panel.t
        a[:3] = v / cfg['translation_m_s']
        a[3:6] = rot / cfg['rotation_rad_s']
        return a
