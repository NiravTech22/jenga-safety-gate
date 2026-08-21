"""Pinhole camera model for the simulation backend.

The renderer projects through a real intrinsics/extrinsics pair rather than
drawing to a fixed 2D layout, which matters for one specific reason: it means
the perception code in Step 2 has to recover pose from ArUco markers exactly
the way it will on hardware. A renderer that drew a convenient top-down
layout would let the occupancy detector cheat, and the cheat would only
surface when the real rig arrived.

Intrinsics here stand in for what `cv2.calibrateCamera` produces from a
checkerboard sweep on the physical rig. Nothing downstream reads these
objects; perception gets the image plus the marker layout and works it out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @staticmethod
    def from_fov(width: int, height: int, horizontal_fov_deg: float = 60.0) -> "CameraIntrinsics":
        f = (width / 2.0) / np.tan(np.radians(horizontal_fov_deg) / 2.0)
        return CameraIntrinsics(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0,
                                width=width, height=height)

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def distortion(self) -> np.ndarray:
        """Zero distortion. The sim renders a pinhole image, so perception
        should undistort with zeros here and swap in the real coefficients
        once the physical cameras are calibrated."""
        return np.zeros((5, 1), dtype=np.float64)


class PinholeCamera:
    """World-to-pixel projection with an OpenCV-convention pose.

    The rotation/translation pair stored here is exactly what `cv2.solvePnP`
    returns for the same scene, so the perception module's recovered pose can
    be diffed against ground truth in tests.
    """

    def __init__(self, intrinsics: CameraIntrinsics, R: np.ndarray, t: np.ndarray):
        self.intrinsics = intrinsics
        self.R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(t, dtype=np.float64).reshape(3, 1)

    @staticmethod
    def look_at(intrinsics: CameraIntrinsics, eye, target, up=(0.0, 0.0, 1.0)) -> "PinholeCamera":
        """Build a camera at `eye` pointing at `target`.

        Uses the OpenCV convention: +z forward along the optical axis, +y
        down in the image. `up` is the world direction that should appear
        upward in the image.
        """
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)

        forward = target - eye
        norm = np.linalg.norm(forward)
        if norm == 0:
            raise ValueError("Camera eye and target coincide; no viewing direction.")
        forward /= norm

        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-9:
            raise ValueError("Camera up vector is parallel to the viewing direction.")
        right /= right_norm

        down = np.cross(forward, right)

        # Rows are the camera basis expressed in world coordinates.
        R = np.stack([right, down, forward], axis=0)
        t = -R @ eye.reshape(3, 1)
        return PinholeCamera(intrinsics, R, t)

    @property
    def position(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return (-self.R.T @ self.t).ravel()

    def rvec_tvec(self) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        rvec, _ = cv2.Rodrigues(self.R)
        return rvec, self.t.copy()

    def to_camera_frame(self, points_world: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3).T
        return (self.R @ pts + self.t).T

    def project(self, points_world: np.ndarray) -> np.ndarray:
        """Project Nx3 world points to Nx2 pixel coordinates.

        Points at or behind the image plane are returned as NaN rather than
        wrapped around, so callers can drop them instead of drawing garbage
        geometry at the far edge of the frame.
        """
        cam = self.to_camera_frame(points_world)
        z = cam[:, 2]
        out = np.full((cam.shape[0], 2), np.nan, dtype=np.float64)
        valid = z > 1e-6
        k = self.intrinsics
        out[valid, 0] = k.fx * cam[valid, 0] / z[valid] + k.cx
        out[valid, 1] = k.fy * cam[valid, 1] / z[valid] + k.cy
        return out

    def depth_of(self, point_world) -> float:
        """Distance along the optical axis. Used for painter's-algorithm
        ordering in the renderer."""
        return float(self.to_camera_frame(np.asarray(point_world).reshape(1, 3))[0, 2])
