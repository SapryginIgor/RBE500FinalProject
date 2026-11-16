"""Compare the teammate DH model with URDF using fixed frame conversions.

Read-only diagnostic: does not initialize ROS or command robot motion.
Run from the workspace root after sourcing install/setup.bash.
"""

import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from omx_basic_control.forward import compute_fk


def translation(x, y, z):
    transform = np.eye(4)
    transform[:3, 3] = [x, y, z]
    return transform


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    transform = np.eye(4)
    transform[:3, :3] = (
        math.cos(angle) * np.eye(3)
        + (1 - math.cos(angle)) * np.outer(axis, axis)
        + math.sin(angle) * cross
    )
    return transform


def mapped_dh(joints):
    # Only this comparison converts DH poses to the URDF frames.
    base = translation(0.012, 0, 0.0595-0.096326)
    tool = translation(0.126-0.1334, 0, 0) @ rotation([1, 0, 0], -math.pi/2)
    return base @ compute_fk(joints) @ tool


def load_urdf_chain(path):
    root = ET.parse(path).getroot()
    joints = []
    for name in ('joint1', 'joint2', 'joint3', 'joint4', 'end_effector_joint'):
        joint = root.find(f"joint[@name='{name}']")
        if joint is None:
            raise ValueError(f'Missing joint {name} in {path}')
        origin = joint.find('origin')
        xyz = [float(value) for value in origin.get('xyz', '0 0 0').split()]
        rpy = [float(value) for value in origin.get('rpy', '0 0 0').split()]
        transform = translation(*xyz)
        for axis, angle in zip(([0, 0, 1], [0, 1, 0], [1, 0, 0]), reversed(rpy)):
            transform = transform @ rotation(axis, angle)
        axis = joint.find('axis')
        joints.append((transform, None if axis is None else [float(v) for v in axis.get('xyz').split()]))
    return joints


def urdf_forward(chain, angles):
    transform = np.eye(4)
    for (origin, axis), angle in zip(chain, [*angles, 0]):
        transform = transform @ origin
        if axis is not None:
            transform = transform @ rotation(axis, angle)
    return transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joints', nargs=4, type=float, default=[0, 0, 0, 0],
                        metavar=('Q1', 'Q2', 'Q3', 'Q4'), help='ROS joint angles in radians')
    parser.add_argument('--urdf', type=Path, default=(
        Path(__file__).resolve().parents[2] / 'open_manipulator'
        / 'open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x.urdf'
    ))
    args = parser.parse_args()
    chain = load_urdf_chain(args.urdf)
    np.set_printoptions(precision=9, suppress=True)
    mapped = mapped_dh(args.joints)
    expected = urdf_forward(chain, args.joints)
    print(f'URDF source: {args.urdf}')
    print(f'ROS joint angles (radians): {args.joints}')
    print('Mapped DH transform (translation in meters):\n', mapped)
    print('URDF transform (translation in meters):\n', expected)
    print('Mapped position (mm):', mapped[:3, 3] * 1000)
    configurations = np.random.default_rng(500).uniform(-math.pi, math.pi, (1000, 4))
    max_position_error = max_rotation_error = 0.0
    for joints in [args.joints, *configurations]:
        actual, target = mapped_dh(joints), urdf_forward(chain, joints)
        np.testing.assert_allclose(actual, target, atol=1e-12, rtol=0)
        max_position_error = max(max_position_error, np.linalg.norm(actual[:3, 3] - target[:3, 3]))
        max_rotation_error = max(max_rotation_error, np.max(np.abs(actual[:3, :3] - target[:3, :3])))
    print('PASS: requested pose and 1000 random configurations agree.')
    print(f'Max position error: {max_position_error:.3e} m')
    print(f'Max rotation-matrix element error: {max_rotation_error:.3e}')


if __name__ == '__main__':
    main()
