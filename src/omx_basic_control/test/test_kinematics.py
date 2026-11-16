import math
import unittest

import numpy as np

from omx_basic_control.forward import forward_kinematics
from omx_basic_control.inverse import inverse_kinematics


def tool_quaternion(q):
    # Revised theta2/theta3 offsets sum to 2*pi, so cancel in orientation.
    a = q[0] / 2
    b = (q[1] + q[2] + q[3]) / 2
    h = math.sqrt(0.5)
    return [h * math.cos(a-b), h * math.sin(a-b),
            h * math.sin(a+b), h * math.cos(a+b)]


class KinematicsTests(unittest.TestCase):
    def test_revised_zero_pose(self):
        # Offset link contributes [-24, 0, -128] mm at zero joint angles.
        expected = [[1, 0, 0, -24+124+133.4], [0, 0, -1, 0],
                    [0, 1, 0, 96.326-128], [0, 0, 0, 1]]
        actual = forward_kinematics([0] * 4)
        actual[:3, 3] *= 1000
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_revised_teammate_dh_chain(self):
        # Independently retain the teammate's millimeter formula for comparison.
        rng = np.random.default_rng(42)
        for q in rng.uniform(-math.pi, math.pi, (100, 4)):
            expected = np.eye(4)
            r = (128**2 + 24**2)**0.5
            for a, theta, d, alpha in zip(
                [0, r, 124, 133.4],
                [q[0], math.pi+math.asin(128/r)+q[1],
                 math.pi-math.asin(128/r)+q[2], q[3]],
                [96.326, 0, 0, 0], [math.pi/2, 0, 0, 0]
            ):
                c, s, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
                expected = expected @ np.array([[c, -s*ca, s*sa, a*c],
                    [s, c*ca, -c*sa, a*s], [0, sa, ca, d], [0, 0, 0, 1]])
            actual = forward_kinematics(q)
            actual[:3, 3] *= 1000
            np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_base_rotation(self):
        actual = forward_kinematics([math.pi/2, 0, 0, 0])
        np.testing.assert_allclose(actual[:3, 3], [0, 0.2334, -0.031674], atol=1e-12)

    def test_elbow_singularities(self):
        beta = math.asin(128 / math.hypot(128, 24))
        offset = math.pi - beta
        for gamma in (0, math.pi, -math.pi, 1e-8):
            q = [0.3, -math.pi-beta, gamma-offset, -gamma]
            target = forward_kinematics(q)
            result = inverse_kinematics(target[:3, 3], tool_quaternion(q))
            np.testing.assert_allclose(forward_kinematics(result), target, atol=1e-8)

    def test_round_trip(self):
        rng = np.random.default_rng(500)
        presets = [[-0.0015, -0.3114, 0.2546, 1.4757],
                   [-0.0015, -0.0537, 0.3344, 1.3162],
                   [-0.0015, -0.4801, 0.1089, 1.4665]]
        for q in presets + rng.uniform(-math.pi, math.pi, (200, 4)).tolist():
            target = forward_kinematics(q)
            # Original IK assumes a positive radial distance from the base axis.
            radial = math.cos(q[0])*target[0, 3] + math.sin(q[0])*target[1, 3]
            if radial <= 1e-8:
                continue
            result = inverse_kinematics(target[:3, 3], tool_quaternion(q))
            np.testing.assert_allclose(forward_kinematics(result), target, atol=1e-8)

    def test_pose_recovery_for_measured_pose_and_pick_waypoints(self):
        configurations = [
            [0, 0, 0, 0],
            [-0.30219421521367806, 0.2592427531524666,
             0.24236896448572454, 1.0293011086710582],
            [-0.0015, -0.3114, 0.2546, 1.4757],
            [-0.0015, -0.0537, 0.3344, 1.3162],
            [-0.0015, -0.4801, 0.1089, 1.4665],
        ]
        for q in configurations:
            with self.subTest(q=q):
                target = forward_kinematics(q)
                result = inverse_kinematics(target[:3, 3], tool_quaternion(q))
                np.testing.assert_allclose(forward_kinematics(result), target, atol=1e-8)

    def test_other_elbow_branch_preserves_pose_but_not_original_angles(self):
        # Document the current solver's branch-selection limitation explicitly.
        q = [0, 0, 0, 0]
        target = forward_kinematics(q)
        result = inverse_kinematics(target[:3, 3], tool_quaternion(q))
        delta = np.asarray(result) - q
        wrapped_error = np.arctan2(np.sin(delta), np.cos(delta))
        self.assertGreater(np.max(np.abs(wrapped_error)), 0.5)
        np.testing.assert_allclose(forward_kinematics(result), target, atol=1e-8)

    def test_unreachable_and_incompatible_orientation(self):
        with self.assertRaisesRegex(ValueError, 'out of reach'):
            inverse_kinematics([10, 0, 0], tool_quaternion([0]*4))
        with self.assertRaisesRegex(ValueError, 'orientation'):
            inverse_kinematics([0.1, 0, 0.1], [0, 0, 0, 1])

    def test_invalid_values(self):
        for q in ([], [0]*3, [0]*5, [0, 0, float('nan'), 0]):
            with self.assertRaises(ValueError):
                forward_kinematics(q)
        for p, quat in (([0, 0, float('inf')], [1, 0, 0, 0]),
                        ([0, 0, 0], [0]*4)):
            with self.assertRaises(ValueError):
                inverse_kinematics(p, quat)


if __name__ == '__main__':
    unittest.main()
