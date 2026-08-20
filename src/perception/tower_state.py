"""
Step 2: Tower-state detection via classical CV (no learned model).

This is deliberately classical: the anomaly detector and policy in later
steps both get scored against ground truth from here, so it can't itself be
a learned system or the whole evaluation becomes circular.

Implemented: ArUco-based coordinate frame calibration, the TowerState data
structure, and the lean-angle math (pure geometry, testable without a camera
— see tests/test_tower_state.py).
Stubbed (needs the physical rig from Step 1 + real block-position ground
truth to tune): block occupancy detection via edge/contour analysis. The
interface is real; the thresholds inside are placeholders that MUST be
tuned against the 20-trial validation set the plan requires before Step 4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TowerState:
    """Structured tower state extracted from one synchronized frame pair."""
    layer_occupancy: list[list[bool]]   # [layer][slot] -> occupied
    lean_angle_deg: float               # 0 = perfectly vertical
    num_layers_detected: int
    confidence: float                   # 0-1, from edge-detection quality, not a learned score


class ArucoFrameCalibrator:
    """Maps pixel coordinates to real-world coordinates using ArUco markers
    fixed at the tower base. Run once per camera-mount session (Step 1), not
    per frame — the whole point of a fixed rig is that this doesn't change.
    """

    def __init__(self, dictionary_name: str = "DICT_4X4_50", marker_length_m: float = 0.03):
        import cv2  # local import

        self.marker_length_m = marker_length_m
        self.dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dictionary_name)
        )
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)
        self._homography = None  # populated by calibrate()

    def calibrate(self, reference_frame: np.ndarray) -> bool:
        """Detect markers in a reference frame and compute the pixel ->
        world homography. Returns False if markers weren't found — caller
        should treat this as a hard stop, not a warning, since everything
        downstream assumes a valid calibration.
        """
        corners, ids, _ = self.detector.detectMarkers(reference_frame)
        if ids is None or len(ids) < 4:
            return False
        # TODO(Step 1, on real hardware): compute homography from the 4
        # base-corner markers' known real-world positions vs detected pixel
        # positions. Left unimplemented here because it depends on the
        # physical marker layout you choose when you mount the rig.
        raise NotImplementedError(
            "Homography computation needs the physical marker layout from "
            "Step 1. Detected markers OK; wire up cv2.findHomography with "
            "your chosen real-world marker coordinates here."
        )

    def is_calibrated(self) -> bool:
        return self._homography is not None


def estimate_lean_angle(side_frame_edges: np.ndarray) -> float:
    """Pure-geometry piece: given already-detected tower edge line segments
    from the side camera (as (x1,y1,x2,y2) arrays), return the lean angle in
    degrees from vertical. Kept separate from edge detection itself so this
    is independently testable without any camera or CV dependency issues.
    """
    if side_frame_edges is None or len(side_frame_edges) == 0:
        raise ValueError("No edges provided — cannot estimate lean angle")

    angles = []
    for x1, y1, x2, y2 in side_frame_edges:
        dx = x2 - x1
        dy = y2 - y1
        if dy == 0:
            continue  # perfectly horizontal edge, not a vertical tower edge
        angle_from_vertical = np.degrees(np.arctan2(dx, dy))
        angles.append(angle_from_vertical)

    if not angles:
        raise ValueError("All provided edges were horizontal — no vertical reference found")

    return float(np.median(angles))  # median, not mean: robust to one bad edge detection


def detect_block_occupancy(
    overhead_frame: np.ndarray,
    calibrator: ArucoFrameCalibrator,
    num_layers: int,
    slots_per_layer: int = 3,
) -> list[list[bool]]:
    """STUB: contour/edge-based occupancy grid.

    Real implementation needs tuning against actual Jenga-block imagery
    (color thresholding or edge density per expected slot region, warped
    into world coordinates via the calibrator's homography). Left as a
    stub with a clear interface rather than a fake implementation, per the
    plan's Step 2 exit condition: this must hit 20/20 on manual validation
    before anything downstream trusts it, and that can't be faked with
    placeholder logic.
    """
    raise NotImplementedError(
        "Implement against real camera frames once Step 1 rig is mounted. "
        "Validate against the 20-trial manual check in PLAN.md Step 2 "
        "before this is used by any other module."
    )


def get_tower_state(
    overhead_frame: np.ndarray,
    side_frame: np.ndarray,
    calibrator: ArucoFrameCalibrator,
    num_layers: int,
) -> TowerState:
    """Top-level entry point other modules should call. Composes the pieces
    above. Will raise NotImplementedError until detect_block_occupancy is
    implemented — that's intentional; nothing downstream should silently
    get fake tower state.
    """
    occupancy = detect_block_occupancy(overhead_frame, calibrator, num_layers)
    # Edge extraction for lean angle also needs real implementation once
    # there's a real side_frame to run cv2.Canny / HoughLinesP against.
    raise NotImplementedError("Wire real edge extraction here once Step 1 is done.")
