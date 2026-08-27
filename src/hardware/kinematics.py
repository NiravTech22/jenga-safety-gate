"""Forward kinematics for the SO-101, straight from its URDF.

Extracted from scripts/render_urdf_preview.py, where the same chain walk was
already verified against the URDF by hand: at zero pose it puts the gripper at
x = 0.399 m, which is the arm's stated reach.

This exists so "jog down to contact and read FK Z" is a computation rather
than an estimate. That measurement is what replaces the z_floor workaround
recorded in docs/PROGRESS.md, where a real safety constraint was disabled to
paper over a miscalibrated ground plane.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_URDF = Path(__file__).resolve().parents[2] / "data/assets/so101/so101_new_calib.urdf"

# Order matters: this is the order joint vectors are given in everywhere else,
# and it matches config.yaml's arm.joints.
JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper")
TOOL_LINK = "gripper_frame_link"


def rpy_to_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx           # URDF fixed-axis convention


def axis_angle_to_matrix(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    x, y, z = axis / n
    c, s, C = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    return np.array([
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ])


def _transform(R, t) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: tuple
    lower: Optional[float]
    upper: Optional[float]


class ArmChain:
    """Kinematic chain read from the URDF, with the joint limits it declares."""

    def __init__(self, urdf_path: Path = DEFAULT_URDF):
        self.urdf_path = Path(urdf_path)
        root = ET.parse(self.urdf_path).getroot()
        self.joints: list[Joint] = []
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in ((o.get("xyz") if o is not None else None) or "0 0 0").split()]
            rpy = [float(v) for v in ((o.get("rpy") if o is not None else None) or "0 0 0").split()]
            axis = j.find("axis")
            limit = j.find("limit")
            self.joints.append(Joint(
                name=j.get("name"), type=j.get("type"),
                parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                origin=_transform(rpy_to_matrix(rpy), xyz),
                axis=tuple(float(v) for v in
                           ((axis.get("xyz") if axis is not None else None) or "0 0 1").split()),
                lower=float(limit.get("lower")) if limit is not None and limit.get("lower") else None,
                upper=float(limit.get("upper")) if limit is not None and limit.get("upper") else None,
            ))

    def limits(self) -> dict:
        """Declared limits per actuated joint, in radians."""
        return {j.name: (j.lower, j.upper) for j in self.joints
                if j.type in ("revolute", "continuous") and j.lower is not None}

    def link_poses(self, angles: dict, root_link: str = "base_link") -> dict:
        poses = {root_link: np.eye(4)}
        remaining = list(self.joints)
        while remaining:
            progressed = False
            for j in list(remaining):
                if j.parent not in poses:
                    continue
                local = j.origin
                if j.type in ("revolute", "continuous"):
                    local = local @ _transform(axis_angle_to_matrix(j.axis, angles.get(j.name, 0.0)),
                                               np.zeros(3))
                poses[j.child] = poses[j.parent] @ local
                remaining.remove(j)
                progressed = True
            if not progressed:
                break
        return poses

    def fk(self, joint_vector, tool_link: str = TOOL_LINK) -> np.ndarray:
        """4x4 pose of the tool frame for a joint vector in JOINT_NAMES order."""
        angles = dict(zip(JOINT_NAMES, np.asarray(joint_vector, float)))
        poses = self.link_poses(angles)
        if tool_link not in poses:
            raise KeyError(f"{tool_link} not reachable from base_link in {self.urdf_path.name}")
        return poses[tool_link]

    def tool_position(self, joint_vector, tool_link: str = TOOL_LINK) -> np.ndarray:
        return self.fk(joint_vector, tool_link)[:3, 3]
