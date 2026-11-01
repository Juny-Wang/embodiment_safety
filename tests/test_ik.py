import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from tasks.task1_dimm_insert.env.ik import solve
from tasks.task1_dimm_insert.env.ik_controller import tick
from tasks.task1_dimm_insert.teleop.vendor.franka_teleop import FrankaTeleop


class IKTests(unittest.TestCase):
    def setUp(self):
        self.lower = np.full(7, -1.)
        self.upper = np.full(7, 1.)
        self.target = np.array([.1, -.1, .2, .03, 0., 0., 0.])

    def errors(self, q):
        return q[:3] - self.target[:3], q[3:6] - self.target[3:6]

    def test_reachable(self):
        q, info = solve(self.errors, np.zeros(7), self.lower, self.upper, 5e-5, 5e-4, 400, "dogbox_trf")
        self.assertTrue(info["accepted"])
        self.assertEqual([x["method"] for x in info["attempts"]], ["dogbox"])
        np.testing.assert_allclose(q[:6], self.target[:6], atol=5e-5)

    def test_fallback_uses_pose_error_and_previous_candidate(self):
        bad = np.zeros(7)
        # A solver's success flag does not establish pose accuracy.
        with patch("tasks.task1_dimm_insert.env.ik.least_squares", side_effect=[
            SimpleNamespace(x=bad, success=True, nfev=2),
            SimpleNamespace(x=self.target.copy(), success=False, nfev=3),
        ]) as optimizer:
            _, info = solve(self.errors, np.ones(7)*.2, self.lower, self.upper, 5e-5, 5e-4, 400, "dogbox_trf")
        self.assertTrue(info["accepted"])
        self.assertFalse(info["local_solver_success"])
        self.assertEqual([x["method"] for x in info["attempts"]], ["dogbox", "trf"])
        np.testing.assert_array_equal(optimizer.call_args_list[1].args[1], bad)

    def test_unreachable_target_rejected_by_both_solvers(self):
        self.target[0] = 2.
        _, info = solve(self.errors, np.zeros(7), self.lower, self.upper, 5e-5, 5e-4, 40, "dogbox_trf")
        self.assertFalse(info["accepted"])
        self.assertEqual(len(info["attempts"]), 2)

    def controller(self, limit=1.):
        control = dict(ik_solver="dogbox_trf", ik_period_s=.05, ee_frame="panda_hand",
                       tracking_error_stop_m=2., translation_speed_m_s=10., rotation_speed_rad_s=10.,
                       arm_joint_speed_rad_s=1., gripper_speed_m_s=.06,
                       ik_position_tolerance_m=5e-5, ik_rotation_tolerance_rad=5e-4, ik_max_evaluations=40)
        c = FrankaTeleop(dict(control=control))
        c.tick = MethodType(tick, c)
        c.ready = c.armed = True
        c.arm_indices = np.arange(7)
        c.fingers = np.array([7, 8])
        c.lower, c.upper = np.full(7, -limit), np.full(7, limit)
        c.hold_bias = c.last_action = np.zeros(9)
        c.width_goal = 0.
        c.last_solution = None
        c.fallback_count = 0
        c.robot = c
        c.get_joint_positions = lambda: np.zeros(9)
        c.pose = lambda: (np.zeros(3), np.eye(3))
        c.apply_targets = lambda q: setattr(c, "applied", q.copy())
        c.goal = (np.array([.1, 0., 0.]), Rotation.from_rotvec([.2, 0., 0.]))
        c.cart_command = (np.zeros(3), Rotation.identity())
        c.solver = SimpleNamespace(compute_forward_kinematics=lambda frame, q:
                                  (q[:3], Rotation.from_rotvec(q[3:6]).as_matrix()))
        return c

    def test_controller_rate_limit_and_verified_cache(self):
        c = self.controller()
        c.tick(.05)
        self.assertTrue(c.armed)
        self.assertTrue(c.ik_diagnostics["accepted"])
        self.assertLessEqual(np.abs(c.applied[:7]).max(), .05)
        self.assertGreater(np.abs(c.last_solution).max(), .05)
        with patch("tasks.task1_dimm_insert.env.ik.solve", side_effect=AssertionError("Cache should avoid solving")):
            c.tick(.05)
        self.assertTrue(c.ik_diagnostics["used_verified_cache"])
        self.assertLessEqual(np.abs(c.ik_diagnostics["arm_target_step_rad"]).max(), .05)

    def test_controller_stops_on_rejected_pose(self):
        c = self.controller(limit=.01)
        c.tick(.05)
        self.assertFalse(c.armed)
        self.assertIn("IK RESIDUAL/LIMIT FAILURE", c.error)
        self.assertIsNone(c.last_solution)
        np.testing.assert_array_equal(c.applied, np.zeros(9))

    def test_default_trf_matches_frozen_controller(self):
        original, current = self.controller(), self.controller()
        original.tick = MethodType(FrankaTeleop.tick, original)
        current.cfg["control"]["ik_solver"] = "trf"
        for target in ([.1, 0., 0.], [.1, .03, 0.], [.08, .02, -.01]):
            for c in (original, current):
                c.goal = (np.array(target), Rotation.from_rotvec([.02, .01, 0.]))
                c.tick(.05)
            self.assertEqual(original.armed, current.armed)
            np.testing.assert_allclose(current.applied, original.applied, atol=1e-12)
            np.testing.assert_allclose(current.last_solution, original.last_solution, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
