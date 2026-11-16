# OpenManipulator FK and IK

`omx_basic_control/forward.py` contains FK, its ROS joint-angle conversion,
and the FK node. `omx_basic_control/inverse.py` contains IK, conversion back to
ROS angles, and the IK service node.
FK output and IK requests use the DH base/tool frames.
The pick controller requests IK for Cartesian grasp/pregrasp poses and sends
the returned ROS joint angles to the arm controller.

## Build

```bash
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select omx_interfaces omx_basic_control --symlink-install
source install/setup.bash
```

Restart FK and IK nodes after rebuilding. Source the workspace in each terminal.

## FK

Start the node:

```bash
ros2 run omx_basic_control forward_kin
```

Send four ROS joint angles in degrees from another terminal:

```bash
ros2 topic pub --once /fk_input std_msgs/msg/Float32MultiArray \
  '{data: [0.0, 0.0, 0.0, 0.0]}'
```

The node prints the matrix and position. `/fk_output` carries the row-major
4x4 base-to-tool DH transform, with translation in mm.
Zero ROS angles give `[281.4, 0, 224.326]` mm. The tool orientation is a +90-degree
rotation about x, with quaternion `[0.7071067811865476, 0, 0, 0.7071067811865476]`.
These coordinates use the drawing's base height and 133.4 mm tool length.
To print the ROS message, run `ros2 topic echo /fk_output --once` before publishing.

To use measured joint angles instead:

```bash
ros2 run omx_basic_control forward_kin --ros-args -p use_joint_states:=true
```

Joint states are read in radians and matched by name. Feedback logging is
limited to once per second; all valid samples are published.
The pick controller publishes the same mapped FK on `/forward_kinematics/output`.

## IK

```bash
ros2 run omx_basic_control inverse_kin
```

From another terminal, request the zero-joint pose:

```bash
ros2 service call /compute_ik omx_interfaces/srv/ComputeIK \
  '{target_pose: {position: {x: 0.2814, y: 0.0, z: 0.224326}, orientation: {x: 0.7071067811865476, y: 0.0, z: 0.0, w: 0.7071067811865476}}}'
```

The request describes the tool relative to the base,
in meters with an orientation quaternion. The response contains ROS joint angles in radians
and `success`. An all-zero quaternion is invalid. Neither node commands motion.
IK assumes the tool lies in front of the base joint's facing direction and uses
one elbow solution, as in the original code. Targets behind the base axis or
with incompatible orientations are rejected. Joint-angle recovery is only
expected for that elbow solution. IK does not check joint limits or collisions.

For `[0, 30, -30, 0]` degrees, FK gives `[342.18461, 0, 195.17725]` mm and
the same orientation as the zero-joint example. The corresponding IK request is:

```bash
ros2 service call /compute_ik omx_interfaces/srv/ComputeIK \
  '{target_pose: {position: {x: 0.3421846096908265, y: 0.0, z: 0.1951772516844082}, orientation: {x: 0.7071067811865476, y: 0.0, z: 0.0, w: 0.7071067811865476}}}'
```

Expected angles: `[0, 0.523598776, -0.523598776, 0]` radians.

## Mapping and equation changes

The revised teammate DH table is retained. Lengths are expressed in meters
inside Python, and the same `a_l`, `theta_l`, `d_l`, `alpha_l` table is used.
IK keeps the cosine-law calculation with `D`, `gamma`, `alpha`, and `psi`.
Changes from the original IK are exact lengths/offsets and using DH tool
elevation instead of quaternion pitch. The original `atan2(y, x)`, `hypot(x, y)`,
and negative square root for the elbow solution are retained. Invalid inputs,
unreachable positions, and incompatible orientations are rejected.

The joint conversion used by FK/IK is:

```text
q_DH = [q1, pi-q2, pi-q3, -q4]       (radians)
```

FK converts ROS angles to DH and returns the DH pose. IK solves the
DH pose and converts the angles back to ROS. DH angles must not be sent
to ROS motor controllers. Part 3 can use these same poses and IK results without
converting poses to URDF coordinates.

## Pick and lift

Keep hardware bringup running, then start `inverse_kin` in another terminal.
The pick controller uses the same DH base/tool coordinates as FK and IK.
Position and lift height parameters are in meters; tool angle is the DH tool
elevation in degrees (`-90` points down).

Preview the IK requests without sending arm or gripper commands:

```bash
ros2 run omx_basic_control omx_basic_control --ros-args -p dry_run:=true
```

The default grasp position `[0.13276, -0.000199, 0.057724]` m was calculated
from the previous grasp waypoint. Set these parameters to your object's actual
grasp location. This example shows the defaults explicitly:

```bash
ros2 run omx_basic_control omx_basic_control --ros-args \
  -p grasp_x:=0.13276 -p grasp_y:=-0.000199 -p grasp_z:=0.057724 \
  -p lift_height:=0.04 -p tool_angle_deg:=-90.0 -p move_duration:=3.0
```

Without `dry_run:=true`, this command moves the real robot. The sequence opens
the gripper, moves to pregrasp, descends, closes the gripper, and lifts back to
the same pregrasp pose. It then lowers the object back to the grasp position,
opens the gripper, and retreats to pregrasp. There are only two IK requests:
pregrasp and grasp; their solutions are reused throughout the sequence.
These targets have the same x/y
and orientation; the controller interpolates joint angles between them, so the
tool path between the endpoints is not guaranteed to be exactly vertical.

All IK requests and joint-limit checks finish before any movement is sent.
Arm requests use `/arm_controller/follow_joint_trajectory`; gripper requests use
`/gripper_controller/gripper_cmd`. Each action must complete successfully before
the next starts. This code does not perform collision checking.

The comparison script alone applies the following pose conversion to check TF:

```text
T_URDF = B @ T_DH @ C
B = translation(12, 0, -36.826)      (mm)
C = translation(-7.4, 0, 0) @ rotation_x(-90 degrees)
```

Do not compare the published DH pose directly to `link1` ->
`end_effector_link` or use previous URDF-frame IK requests. Restart the nodes
for the updated DH frame convention.

## Checks

```bash
PYTHONPATH=src/omx_basic_control python3 -m unittest discover \
  -s src/omx_basic_control/test -v

python3 src/omx_basic_control/tools/check_model_mapping.py \
  --joints -0.30219421521367806 0.2592427531524666 \
  0.24236896448572454 1.0293011086710582
```

The script compares the mapped model with the local URDF at the supplied pose
and 1000 random configurations. Agreement checks the two mathematical models;
it does not independently verify physical calibration.
