import math
import time

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, GripperCommand
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from omx_interfaces.srv import ComputeIK

from .forward import joint_angles_from_state


GRIPPER_OPEN = 0.019
GRIPPER_CLOSED = -0.008
JOINT_LIMITS = [(-math.pi, math.pi), (-1.5, 1.5), (-1.5, 1.4), (-1.7, 1.97)]


class RobotController(Node):

    def __init__(self):
        super().__init__('robot_controller')
        self.arm_client = ActionClient(
            self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory'
        )
        self.gripper_client = ActionClient(
            self, GripperCommand, '/gripper_controller/gripper_cmd'
        )
        self.ik_client = self.create_client(ComputeIK, 'compute_ik')
        self.subscription = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.arm_joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        self.arm_joint_positions = [0.0]*4
        self.joint_received = False
        self.active_goal = None

        self.declare_parameter('grasp_x', 0.13276)
        self.declare_parameter('grasp_y', -0.000199)
        self.declare_parameter('grasp_z', 0.057724)
        self.declare_parameter('lift_height', 0.04)
        self.declare_parameter('tool_angle_deg', -90.0)
        self.declare_parameter('move_duration', 3.0)
        self.declare_parameter('dry_run', False)

    def joint_state_callback(self, msg):
        try:
            self.arm_joint_positions = joint_angles_from_state(msg)
        except ValueError:
            return
        self.joint_received = True

    def request_ik(self, position, orientation):
        request = ComputeIK.Request()
        pose = request.target_pose
        pose.position.x, pose.position.y, pose.position.z = position
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = orientation
        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done():
            future.cancel()
            raise RuntimeError('IK request timed out')
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(f'IK failed for position {position}')
        angles = list(response.joint_angles)
        if len(angles) != 4 or not all(math.isfinite(q) for q in angles):
            raise RuntimeError('IK returned invalid joint angles')
        for name, angle, (lower, upper) in zip(self.arm_joint_names, angles, JOINT_LIMITS):
            if not lower <= angle <= upper:
                raise RuntimeError(f'IK angle for {name} is outside its limits: {angle}')
        return angles

    def plan_pick(self):
        x = self.get_parameter('grasp_x').value
        y = self.get_parameter('grasp_y').value
        z = self.get_parameter('grasp_z').value
        height = self.get_parameter('lift_height').value
        phi = math.radians(self.get_parameter('tool_angle_deg').value)
        duration = self.get_parameter('move_duration').value
        if not all(math.isfinite(v) for v in (x, y, z, height, phi, duration)):
            raise ValueError('Pose and movement parameters must be finite')
        if height <= 0 or duration <= 0:
            raise ValueError('lift_height and move_duration must be positive')

        # DH tool orientation: Rz(base angle) Rx(90 degrees) Rz(tool elevation).
        a = math.atan2(y, x)/2
        b = phi/2
        h = math.sqrt(0.5)
        orientation = [h*math.cos(a-b), h*math.sin(a-b),
                       h*math.sin(a+b), h*math.cos(a+b)]

        pregrasp = self.request_ik([x, y, z+height], orientation)
        grasp = self.request_ik([x, y, z], orientation)
        self.get_logger().info(f'Grasp position (m): {[x, y, z]}')
        self.get_logger().info(f'Pregrasp position (m): {[x, y, z+height]}')
        self.get_logger().info(f'Pregrasp joints (rad): {pregrasp}')
        self.get_logger().info(f'Grasp joints (rad): {grasp}')
        return pregrasp, grasp, duration

    def send_goal(self, client, goal, timeout):
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done():
            raise RuntimeError('Controller did not acknowledge the goal')
        handle = future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError('Controller rejected the goal')
        self.active_goal = handle
        future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done():
            self.cancel_active_goal()
            raise RuntimeError('Movement timed out')
        self.active_goal = None
        result = future.result()
        if result is None or result.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError('Controller failed to complete the movement')
        return result.result

    def cancel_active_goal(self):
        if self.active_goal is not None and rclpy.ok():
            future = self.active_goal.cancel_goal_async()
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            self.active_goal = None

    def send_arm_command(self, angles, duration):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.arm_joint_names
        point = JointTrajectoryPoint()
        point.positions = angles
        point.time_from_start = Duration(seconds=duration).to_msg()
        goal.trajectory.points.append(point)
        result = self.send_goal(self.arm_client, goal, duration+5.0)
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise RuntimeError(f'Arm movement failed: {result.error_string}')

    def send_gripper_command(self, position):
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = 10.0
        self.send_goal(self.gripper_client, goal, 5.0)

    def run(self):
        if not self.ik_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('Start the inverse_kin node: /compute_ik is unavailable')
        pregrasp, grasp, duration = self.plan_pick()
        if self.get_parameter('dry_run').value:
            self.get_logger().info('Dry run complete; no movement requests sent')
            return

        if not self.arm_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError('Arm controller is unavailable')
        if not self.gripper_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError('Gripper controller is unavailable')
        deadline = time.monotonic()+10.0
        while not self.joint_received and rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not self.joint_received:
            raise RuntimeError('No complete /joint_states sample received')

        self.get_logger().info('Opening gripper')
        self.send_gripper_command(GRIPPER_OPEN)
        self.get_logger().info('Moving to pregrasp')
        self.send_arm_command(pregrasp, duration)
        self.get_logger().info('Moving down to grasp')
        self.send_arm_command(grasp, duration)
        self.get_logger().info('Closing gripper')
        self.send_gripper_command(GRIPPER_CLOSED)
        self.get_logger().info('Lifting back to pregrasp')
        self.send_arm_command(pregrasp, duration)
        self.get_logger().info('Lowering object back to grasp position')
        self.send_arm_command(grasp, duration)
        self.get_logger().info('Releasing object')
        self.send_gripper_command(GRIPPER_OPEN)
        self.get_logger().info('Retreating to pregrasp')
        self.send_arm_command(pregrasp, duration)
        self.get_logger().info('Pick and place sequence done')


def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    try:
        node.run()
    except KeyboardInterrupt:
        node.cancel_active_goal()
    except (RuntimeError, ValueError) as error:
        node.get_logger().error(str(error))
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
