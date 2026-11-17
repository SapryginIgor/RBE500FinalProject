"""FK from ROS joint angles."""

import math
from math import sin, cos

from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState

JOINT_NAMES = ('joint1', 'joint2', 'joint3', 'joint4')


def ros_to_dh_angles(joint_angles):
    q = np.asarray(joint_angles, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError('Expected four finite joint angles')
    return np.array([q[0], math.pi-q[1], math.pi-q[2], -q[3]])


def dh_transform(a, theta, d, alpha):
    An = np.array([
        [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
        [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
        [0, sin(alpha), cos(alpha), d],
        [0, 0, 0, 1],
    ], dtype=float)
    return An


def forward_kinematics(joint_angles):
    if len(joint_angles) != 4 or not np.all(np.isfinite(joint_angles)):
        raise ValueError('Expected four finite joint angles')
    q1, q2, q3, q4 = joint_angles

    l1 = 0.096326
    l2 = 0.128
    l3 = 0.124
    l4 = 0.1334
    a2 = 0.024
    r = (l2*l2 + a2*a2)**0.5

    a_l = [0, r, l3, l4]
    theta_l = [q1, math.pi + math.asin(l2/r) + q2,
               math.pi - math.asin(l2/r) + q3, q4]
    d_l = [l1, 0, 0, 0]
    alpha_l = [math.pi/2, 0, 0, 0]

    A1 = dh_transform(a_l[0], theta_l[0], d_l[0], alpha_l[0])
    A2 = dh_transform(a_l[1], theta_l[1], d_l[1], alpha_l[1])
    A3 = dh_transform(a_l[2], theta_l[2], d_l[2], alpha_l[2])
    A4 = dh_transform(a_l[3], theta_l[3], d_l[3], alpha_l[3])
    Tf = A1 @ A2 @ A3 @ A4
    return Tf


def compute_fk(joint_angles):
    # Input: ROS radians. Output: base-to-tip DH pose, in meters.
    return forward_kinematics(ros_to_dh_angles(joint_angles))


def pose_message(transform, stamp, frame_id):
    """End-effector pose in the DH base frame; position in meters (ROS convention)."""
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (
        float(v) for v in transform[:3, 3])
    x, y, z, w = Rotation.from_matrix(transform[:3, :3]).as_quat()
    msg.pose.orientation.x = float(x)
    msg.pose.orientation.y = float(y)
    msg.pose.orientation.z = float(z)
    msg.pose.orientation.w = float(w)
    return msg


def joint_angles_from_state(msg):
    """Extract a complete named sample, ignoring gripper and message order."""
    positions = dict(zip(msg.name, msg.position))
    if not all(name in positions for name in JOINT_NAMES):
        raise ValueError('JointState must contain positions for joint1 through joint4')
    angles = [positions[name] for name in JOINT_NAMES]
    if not np.all(np.isfinite(angles)):
        raise ValueError('JointState contains non-finite arm positions')
    return angles


class ForwardKin(Node):
    def __init__(self):
        super().__init__('forward_kin')
        self.pose_publisher = self.create_publisher(PoseStamped, 'fk_pose', 10)
        self.declare_parameter('frame_id', 'dh_base')
        self.joint_subscription = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

    def joint_state_callback(self, msg):
        try:
            angles = joint_angles_from_state(msg)
            self.publish_transform(compute_fk(angles), angles)
        except ValueError as error:
            self.get_logger().warning(str(error))

    def publish_transform(self, transform, angles):
        self.pose_publisher.publish(pose_message(
            transform, self.get_clock().now().to_msg(),
            self.get_parameter('frame_id').value))
        Tf = transform.copy()
        Tf[:3, 3] *= 1000
        Tf[np.abs(Tf) < 0.000005] = 0
        tf_str = np.array2string(Tf, formatter={'float_kind': lambda x: f'{x:12.5f}'})
        degrees = np.array2string(np.degrees(angles), precision=5, suppress_small=True)
        report = (
            f'\nJoint angles (degrees): {degrees}\n'
            f'Tip position (mm): {Tf[:3, 3]}\n'
            f'T Matrix (base -> tool, mm):\n{tf_str}'
        )
        # Joint states arrive at ~100 Hz; log once per second, publish every sample
        self.get_logger().info(report, throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = ForwardKin()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
