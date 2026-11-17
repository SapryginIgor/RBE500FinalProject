import unittest
from unittest.mock import Mock, patch

import numpy as np
import rclpy
from rclpy.parameter import Parameter
from rclpy.task import Future
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, GripperCommand
from omx_interfaces.srv import ComputeIK

from omx_basic_control.forward import compute_fk
from omx_basic_control.inverse import compute_ik
from omx_basic_control.omx_basic_control import RobotController, GRIPPER_OPEN, GRIPPER_CLOSED


class PickSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = RobotController()
        self.node.ik_client.wait_for_service = Mock(return_value=True)
        # These tests never send requests to hardware or other running nodes.
        self.node.request_ik = Mock(side_effect=compute_ik)
        self.node.send_arm_command = Mock()
        self.node.send_gripper_command = Mock()

    def tearDown(self):
        self.node.destroy_node()

    def test_requests_ik_only_for_pregrasp_and_grasp(self):
        pregrasp, grasp, duration = self.node.plan_pick()
        requests = self.node.request_ik.call_args_list
        self.assertEqual(len(requests), 2)
        self.assertEqual(duration, 3.0)
        first, last = requests[0].args[0], requests[-1].args[0]
        self.assertAlmostEqual(first[2]-last[2], 0.04)
        for angles, request in zip((pregrasp, grasp), requests):
            position, _ = request.args
            np.testing.assert_allclose(position[:2], first[:2])
            np.testing.assert_allclose(compute_fk(angles)[:3, 3], position, atol=1e-8)

    def test_pick_and_place_reuses_poses_and_releases_before_retreat(self):
        events = []
        self.node.joint_received = True
        self.node.arm_client.wait_for_server = Mock(return_value=True)
        self.node.gripper_client.wait_for_server = Mock(return_value=True)
        self.node.send_arm_command.side_effect = lambda points, duration: events.append(('arm', points))
        self.node.send_gripper_command.side_effect = lambda position: events.append(('gripper', position))
        self.node.run()
        self.assertEqual([event[0] for event in events],
                         ['gripper', 'arm', 'arm', 'gripper', 'arm', 'arm', 'gripper', 'arm'])
        self.assertEqual(events[0][1], GRIPPER_OPEN)
        self.assertEqual(events[3][1], GRIPPER_CLOSED)
        self.assertEqual(events[4][1], events[1][1])
        self.assertEqual(events[5][1], events[2][1])
        self.assertEqual(events[6][1], GRIPPER_OPEN)
        self.assertEqual(events[7][1], events[1][1])
        self.assertNotEqual(events[2][1], events[1][1])
        self.assertEqual(self.node.request_ik.call_count, 2)

    def test_dry_run_does_not_contact_movement_servers(self):
        self.node.set_parameters([Parameter('dry_run', value=True)])
        self.node.arm_client.wait_for_server = Mock()
        self.node.gripper_client.wait_for_server = Mock()
        self.node.run()
        self.node.send_arm_command.assert_not_called()
        self.node.send_gripper_command.assert_not_called()
        self.node.arm_client.wait_for_server.assert_not_called()
        self.node.gripper_client.wait_for_server.assert_not_called()

    def test_unreachable_target_stops_before_any_movement(self):
        self.node.set_parameters([Parameter('grasp_x', value=10.0)])
        with self.assertRaises(ValueError):
            self.node.run()
        self.node.send_arm_command.assert_not_called()
        self.node.send_gripper_command.assert_not_called()

    def test_ik_failure_and_out_of_limit_result_are_rejected(self):
        for success, angles in ((False, []), (True, [0.0, 2.0, 0.0, 0.0])):
            response = ComputeIK.Response(success=success, joint_angles=angles)
            future = Future()
            future.set_result(response)
            self.node.ik_client.call_async = Mock(return_value=future)
            with patch('omx_basic_control.omx_basic_control.rclpy.spin_until_future_complete'):
                with self.assertRaises(RuntimeError):
                    RobotController.request_ik(self.node, [0.2, 0.0, 0.1], [0.5]*4)

    def test_arm_request_contains_the_ik_solution(self):
        positions = [0.0, 0.1, 0.2, 0.3]
        self.node.send_goal = Mock(return_value=FollowJointTrajectory.Result(error_code=0))
        RobotController.send_arm_command(self.node, positions, 3.0)
        client, goal, timeout = self.node.send_goal.call_args.args
        self.assertIs(client, self.node.arm_client)
        self.assertEqual(timeout, 8.0)
        self.assertEqual(goal.trajectory.joint_names, self.node.arm_joint_names)
        self.assertEqual(len(goal.trajectory.points), 1)
        self.assertEqual(list(goal.trajectory.points[0].positions), positions)
        times = [p.time_from_start.sec+p.time_from_start.nanosec/1e9 for p in goal.trajectory.points]
        self.assertEqual(times, [3.0])

    def gripper_response(self, status, stalled, position=0.002):
        result = GripperCommand.Result(position=position, stalled=stalled)
        return Mock(status=status, result=result)

    def test_gripper_stalled_on_object_counts_as_grasp(self):
        self.node.execute_goal = Mock(
            return_value=self.gripper_response(GoalStatus.STATUS_ABORTED, stalled=True))
        RobotController.send_gripper_command(self.node, GRIPPER_CLOSED)
        goal = self.node.execute_goal.call_args.args[1]
        self.assertEqual(goal.command.position, GRIPPER_CLOSED)

    def test_gripper_reaching_target_succeeds(self):
        self.node.execute_goal = Mock(
            return_value=self.gripper_response(GoalStatus.STATUS_SUCCEEDED, stalled=False))
        RobotController.send_gripper_command(self.node, GRIPPER_OPEN)

    def test_gripper_abort_without_stall_is_an_error(self):
        self.node.execute_goal = Mock(
            return_value=self.gripper_response(GoalStatus.STATUS_ABORTED, stalled=False))
        with self.assertRaisesRegex(RuntimeError, 'Gripper'):
            RobotController.send_gripper_command(self.node, GRIPPER_CLOSED)

if __name__ == '__main__':
    unittest.main()
