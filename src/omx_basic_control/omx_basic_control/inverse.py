"""ComputeIK service; computes angles without commanding hardware."""

import math

import rclpy
import numpy as np
from rclpy.node import Node
from omx_interfaces.srv import ComputeIK


def quaternion_rotation(quaternion):
    """Convert an [x, y, z, w] quaternion to a rotation matrix."""
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError('Expected a finite quaternion [x, y, z, w]')
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError('Quaternion must be nonzero')
    x, y, z, w = q / norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def inverse_kinematics(position, quaternion):
    p = np.asarray(position, dtype=float)
    if p.shape != (3,) or not np.all(np.isfinite(p)):
        raise ValueError('Expected a finite position [x, y, z] in meters')
    x, y, z = p
    rotation = quaternion_rotation(quaternion)

    theta1 = math.atan2(y, x)
    r = math.hypot(x, y)
    if not np.allclose(rotation[:, 2], [math.sin(theta1), -math.cos(theta1), 0],
                       atol=1e-6, rtol=0):
        raise ValueError('Target orientation is incompatible with the arm')

    # phi is tool elevation;
    phi = math.atan2(rotation[2, 0],
                     math.cos(theta1)*rotation[0, 0] + math.sin(theta1)*rotation[1, 0])

    d1 = 0.096326
    a2 = (0.128**2 + 0.024**2)**0.5
    a3 = 0.124
    a4 = 0.1334
    offset2 = math.pi + math.asin(0.128/a2)
    offset3 = math.pi - math.asin(0.128/a2)

    dz = z - d1
    rw = r - a4*math.cos(phi)
    zw = dz - a4*math.sin(phi)

    D = (rw**2 + zw**2 - a2**2 - a3**2) / (2*a2*a3)
    if abs(D) > 1 + 1e-9:
        raise ValueError('Target pose is out of reach')
    D = float(np.clip(D, -1, 1))
    gamma = math.atan2(-math.sqrt(max(0, 1 - D**2)), D)
    theta3 = gamma - offset3

    alpha = math.atan2(zw, rw)
    psi = math.atan2(a3*math.sin(gamma), a2 + a3*math.cos(gamma))
    theta2 = (alpha - psi) - offset2
    theta4 = phi - (theta2 + theta3)

    return [theta1, theta2, theta3, theta4]


def dh_to_ros_angles(joint_angles):
    q = np.asarray(joint_angles, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError('Expected four finite joint angles')
    angles = np.array([q[0], math.pi-q[1], math.pi-q[2], -q[3]])
    return np.arctan2(np.sin(angles), np.cos(angles)).tolist()


def compute_ik(position, quaternion):
    # Input: DH pose, in meters. Output: ROS radians.
    return dh_to_ros_angles(inverse_kinematics(position, quaternion))


class InverseKinematicsNode(Node):
    def __init__(self):
        super().__init__('inverse_kinematics_node')
        self.srv = self.create_service(ComputeIK, 'compute_ik', self.compute_ik_callback)

    def compute_ik_callback(self, request, response):
        pose = request.target_pose
        try:
            response.joint_angles = compute_ik(
                [pose.position.x, pose.position.y, pose.position.z],
                [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
            )
            response.success = True
            angles = np.array(response.joint_angles)
            self.get_logger().info(
                '\nIK solution [joint1, joint2, joint3, joint4]:\n'
                f'Radians: {np.array2string(angles, precision=5, suppress_small=True)}\n'
                f'Degrees: {np.array2string(np.degrees(angles), precision=5, suppress_small=True)}'
            )
        except ValueError as error:
            self.get_logger().warning(str(error))
            response.joint_angles = []
            response.success = False
        return response


def main(args=None):
    rclpy.init(args=args)
    node = InverseKinematicsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
