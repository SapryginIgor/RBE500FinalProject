import math
import unittest

import numpy as np
from builtin_interfaces.msg import Time

from omx_basic_control.forward import forward_kinematics, compute_fk, ros_to_dh_angles, pose_message
from omx_basic_control.inverse import quaternion_rotation, compute_ik, dh_to_ros_angles


def tool_quaternion(q):
    # DH orientation: Rz(q1) Rx(pi/2) Rz(-q2-q3-q4).
    a, b = q[0]/2, -sum(q[1:])/2
    h = math.sqrt(0.5)
    return [h*math.cos(a-b), h*math.sin(a-b),
            h*math.sin(a+b), h*math.cos(a+b)]


class RosKinematicsTests(unittest.TestCase):
    def test_ros_zero_pose(self):
        expected = np.eye(4)
        expected[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        expected[:3, 3] = [0.024+0.124+0.1334, 0, 0.096326+0.128]
        np.testing.assert_allclose(compute_fk([0]*4), expected, atol=1e-12)

    def test_recorded_robot_pose_in_dh_frame(self):
        q = [-0.30219421521367806, 0.2592427531524666,
             0.24236896448572454, 1.0293011086710582]
        actual = compute_fk(q)
        q1, q2, q3, q4 = q
        r = 0.024*math.cos(q2) + 0.128*math.sin(q2) + 0.124*math.cos(q2+q3) + 0.1334*math.cos(q2+q3+q4)
        z = 0.096326 - 0.024*math.sin(q2) + 0.128*math.cos(q2) - 0.124*math.sin(q2+q3) - 0.1334*math.sin(q2+q3+q4)
        np.testing.assert_allclose(actual[:3, 3], [r*math.cos(q1), r*math.sin(q1), z], atol=1e-12)
        np.testing.assert_allclose(actual[:3, :3], quaternion_rotation(tool_quaternion(q)), atol=1e-12)

    def test_ros_joint_recovery(self):
        for q in ([0]*4, [0, math.pi/6, -math.pi/6, 0],
                  [-0.30219421521367806, 0.2592427531524666, 0.24236896448572454, 1.0293011086710582],
                  [-0.0015, -0.3114, 0.2546, 1.4757],
                  [-0.0015, -0.0537, 0.3344, 1.3162],
                  [-0.0015, -0.4801, 0.1089, 1.4665]):
            with self.subTest(q=q):
                target = compute_fk(q)
                result = compute_ik(target[:3, 3], tool_quaternion(q))
                np.testing.assert_allclose(result, q, atol=1e-8)

    def test_ros_random_pose_round_trips(self):
        for q in np.random.default_rng(500).uniform(-math.pi, math.pi, (200, 4)):
            target = compute_fk(q)
            dh_target = forward_kinematics(ros_to_dh_angles(q))
            radial = math.cos(q[0])*dh_target[0, 3] + math.sin(q[0])*dh_target[1, 3]
            if radial <= 1e-8:
                continue
            result = compute_ik(target[:3, 3], tool_quaternion(q))
            np.testing.assert_allclose(compute_fk(result), target, atol=1e-8)

    def test_pose_message_matches_fk(self):
        for q in np.random.default_rng(7).uniform(-math.pi, math.pi, (100, 4)):
            T = compute_fk(q)
            pose = pose_message(T, Time(), 'dh_base').pose
            p, o = pose.position, pose.orientation
            np.testing.assert_allclose([p.x, p.y, p.z], T[:3, 3], atol=1e-12)
            np.testing.assert_allclose(
                quaternion_rotation([o.x, o.y, o.z, o.w]), T[:3, :3], atol=1e-10)

    def test_target_behind_base_axis_is_rejected(self):
        q = [0, math.pi, 0, 0]
        target = compute_fk(q)
        with self.assertRaisesRegex(ValueError, 'orientation'):
            compute_ik(target[:3, 3], tool_quaternion(q))

    def test_joint_mapping_is_reversible_modulo_full_turns(self):
        for q in np.random.default_rng(123).uniform(-math.pi, math.pi, (20, 4)):
            result = dh_to_ros_angles(ros_to_dh_angles(q))
            delta = result-q
            np.testing.assert_allclose(np.arctan2(np.sin(delta), np.cos(delta)), 0, atol=1e-12)

    def test_ros_invalid_and_unreachable_targets(self):
        for position, quaternion in (([10, 0, 0], [0, 0, 0, 1]),
                                     ([0.2814, 0, 0.224326], [0]*4),
                                     ([0, 0, float('nan')], [0, 0, 0, 1])):
            with self.assertRaises(ValueError):
                compute_ik(position, quaternion)


if __name__ == '__main__':
    unittest.main()
