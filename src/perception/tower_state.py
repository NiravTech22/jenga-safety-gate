"""
Step 2: Tower-state detection via classical CV (no learned model).

Deliberately classical. The anomaly detector and the policy in later steps are
both scored against ground truth from here, so if this were itself learned the
whole evaluation would be circular: two models trained on overlapping data
agreeing tells you nothing about whether either is right.

Pipeline, per frame pair:
  1. Detect the four table-plane ArUco markers and recover camera pose with
     solvePnP. Pose, not just a homography, because block faces sit above the
     marker plane and a plane-to-plane homography cannot project them.
  2. Estimate lean by fitting the projected tower silhouette against the
     segmented wood mask over a sweep of candidate angles. Lean has to come
     first: at the top of the tower a few degrees moves a block by more than
     its own width, so sampling slot regions without it would read the wrong
     patches.
  3. Sample each slot's outward-facing rectangle and classify occupied or
     empty against a brightness threshold fitted from the Step 1 reference
     photos of a known-full tower.
  4. Cross-check the top layer against the overhead camera and report a
     confidence from the agreement and the classification margins.

Thresholds here are fitted from reference imagery rather than hard-coded, so
the same code path applies on hardware. What must be re-run on hardware is the
fitting (Step 1) and the 20-trial validation (Step 2), not this logic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from src.rig import (
    BLOCK_H_M,
    BLOCK_L_M,
    MarkerLayout,
    SLOTS_PER_LAYER,
    apply_lean,
    slot_face_quad,
    tower_height_m,
)


@dataclass
class TowerState:
    """Structured tower state extracted from one synchronized frame pair."""
    layer_occupancy: list          # [layer][slot] -> bool
    lean_angle_deg: float          # 0 = perfectly vertical
    num_layers_detected: int
    confidence: float              # 0-1, from edge/segmentation quality, not learned

    def occupancy_array(self) -> np.ndarray:
        return np.array(self.layer_occupancy, dtype=bool)


class CalibrationError(RuntimeError):
    pass


MIN_MARKERS_FOR_POSE = 3


class ArucoFrameCalibrator:
    """Recovers camera pose from the four base markers.

    Run once per camera-mount session, not per frame: the whole point of a
    rigidly fixed rig is that this does not change. Re-running it per frame
    would also mask exactly the drift the Step 1 reference photos exist to
    catch.
    """

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: Optional[np.ndarray] = None,
                 layout: Optional[MarkerLayout] = None):
        self.layout = layout or MarkerLayout()
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = (np.zeros((5, 1)) if dist_coeffs is None
                            else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1))
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, self.layout.dictionary_name)
        )
        self.detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        self._rvec: Optional[np.ndarray] = None
        self._tvec: Optional[np.ndarray] = None
        self._homography: Optional[np.ndarray] = None
        self._n_markers_seen = 0

    # -- calibration -------------------------------------------------------
    def calibrate(self, reference_frame: np.ndarray) -> bool:
        """Detect markers and solve for pose.

        Needs three of the four markers. Each contributes four corners, so
        three markers still give twelve coplanar correspondences, which is a
        well-conditioned pose. The fourth is the spare the layout exists to
        provide: from the side camera the tower itself occludes the far
        marker, and on hardware the arm will occlude others mid-episode.

        Returns False below that. Callers should treat it as a hard stop
        rather than a warning, since everything downstream assumes a valid
        calibration.
        """
        corners, ids, _ = self.detector.detectMarkers(reference_frame)
        if ids is None:
            self._n_markers_seen = 0
            return False

        ids = ids.flatten().tolist()
        self._n_markers_seen = len(ids)

        obj_pts, img_pts = [], []
        for want_index, want_id in enumerate(self.layout.marker_ids):
            if want_id not in ids:
                continue
            det = corners[ids.index(want_id)].reshape(4, 2)
            obj_pts.append(self.layout.corners_world(want_index))
            img_pts.append(det)

        if len(obj_pts) < MIN_MARKERS_FOR_POSE:
            return False

        obj = np.vstack(obj_pts).astype(np.float64)
        img = np.vstack(img_pts).astype(np.float64)

        ok, rvec, tvec = cv2.solvePnP(obj, img, self.camera_matrix, self.dist_coeffs,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return False

        self._rvec, self._tvec = rvec, tvec
        self._obj_px = (obj, img)
        # Table-plane homography, kept for callers that only need z=0 mapping.
        self._homography, _ = cv2.findHomography(obj[:, :2], img)
        return True

    def is_calibrated(self) -> bool:
        return self._rvec is not None

    @property
    def n_markers_seen(self) -> int:
        return self._n_markers_seen

    def _require(self) -> None:
        if not self.is_calibrated():
            raise CalibrationError(
                "Calibrator has no pose. Call calibrate() on a frame showing "
                "all four base markers before projecting anything."
            )

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        self._require()
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.ravel().tolist(),
            "rvec": self._rvec.ravel().tolist(),
            "tvec": self._tvec.ravel().tolist(),
            "n_markers_seen": self._n_markers_seen,
            "reprojection_error_px": self.reprojection_error_px(),
            "marker_length_m": self.layout.marker_length_m,
            "marker_offset_m": self.layout.offset_m,
            "dictionary": self.layout.dictionary_name,
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @staticmethod
    def load(path: Path) -> "ArucoFrameCalibrator":
        raw = json.loads(Path(path).read_text())
        layout = MarkerLayout(marker_length_m=raw["marker_length_m"],
                              offset_m=raw["marker_offset_m"],
                              dictionary_name=raw["dictionary"])
        cal = ArucoFrameCalibrator(np.array(raw["camera_matrix"]),
                                   np.array(raw["dist_coeffs"]), layout)
        cal._rvec = np.array(raw["rvec"], dtype=np.float64).reshape(3, 1)
        cal._tvec = np.array(raw["tvec"], dtype=np.float64).reshape(3, 1)
        cal._n_markers_seen = raw["n_markers_seen"]
        return cal

    # -- projection --------------------------------------------------------
    def project(self, points_world: np.ndarray) -> np.ndarray:
        self._require()
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 1, 3)
        px, _ = cv2.projectPoints(pts, self._rvec, self._tvec,
                                  self.camera_matrix, self.dist_coeffs)
        return px.reshape(-1, 2)

    def reprojection_error_px(self) -> float:
        """RMS reprojection error over the marker corners. The honest health
        check on a calibration: a pose that fits the markers badly will
        mis-place block regions everywhere else."""
        self._require()
        pair = getattr(self, "_obj_px", None)
        if pair is None:
            return float("nan")      # pose was loaded from disk, not solved here
        obj, img = pair
        return float(np.sqrt(np.mean(np.sum((self.project(obj) - img) ** 2, axis=1))))


# --------------------------------------------------------------------------- #
# Segmentation and lean
# --------------------------------------------------------------------------- #
WOOD_SATURATION_MIN = 60
WOOD_VALUE_MIN = 60


def segment_wood_mask(frame: np.ndarray) -> np.ndarray:
    """Binary mask of block faces.

    Keyed on saturation: the blocks are the only strongly coloured thing in
    the scene, with the table and backdrop both near-neutral. On hardware
    this is the one place that will need re-tuning for the actual wood and
    lighting, which is what the Step 1 reference photos are for.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] >= WOOD_SATURATION_MIN) &
            (hsv[:, :, 2] >= WOOD_VALUE_MIN)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def largest_component_hull(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return cv2.convexHull(max(contours, key=cv2.contourArea))


def _tower_hull_world(num_layers: int, lean_deg: float) -> np.ndarray:
    h = BLOCK_L_M / 2
    top = tower_height_m(num_layers)
    box = np.array([
        [-h, -h, 0.0], [h, -h, 0.0], [h, h, 0.0], [-h, h, 0.0],
        [-h, -h, top], [h, -h, top], [h, h, top], [-h, h, top],
    ])
    return apply_lean(box, lean_deg)


def _mask_from_polygon(shape, polygon: np.ndarray) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(polygon.round().astype(np.int32)), 255)
    return m


def estimate_lean_from_silhouette(frame: np.ndarray, calibrator: ArucoFrameCalibrator,
                                  num_layers: int, search_deg: float = 9.0,
                                  coarse_step: float = 0.5) -> tuple:
    """Sweep candidate lean angles and keep the one whose projected tower
    silhouette best overlaps the segmented wood mask.

    Returns (lean_deg, iou). Template fitting rather than Hough line angles,
    because an image-space edge angle is not a world lean under perspective,
    and converting between them needs the pose anyway.
    """
    mask = segment_wood_mask(frame)
    hull = largest_component_hull(mask)
    if hull is None:
        return 0.0, 0.0
    observed = _mask_from_polygon(frame.shape, hull.reshape(-1, 2))

    def iou_for(lean: float) -> float:
        projected = calibrator.project(_tower_hull_world(num_layers, lean))
        if not np.isfinite(projected).all():
            return 0.0
        candidate = _mask_from_polygon(frame.shape, projected)
        inter = np.count_nonzero(cv2.bitwise_and(observed, candidate))
        union = np.count_nonzero(cv2.bitwise_or(observed, candidate))
        return inter / union if union else 0.0

    coarse = np.arange(-search_deg, search_deg + 1e-9, coarse_step)
    scores = [iou_for(a) for a in coarse]
    best = float(coarse[int(np.argmax(scores))])

    fine = np.arange(best - coarse_step, best + coarse_step + 1e-9, coarse_step / 5)
    fine_scores = [iou_for(a) for a in fine]
    best_fine = float(fine[int(np.argmax(fine_scores))])
    return best_fine, float(max(fine_scores))


def estimate_lean_angle(side_frame_edges: np.ndarray) -> float:
    """Pure-geometry helper: given detected tower edge segments as
    (x1,y1,x2,y2) rows, return the median angle from vertical in degrees.

    Kept separate from edge detection so it is testable without a camera.
    Note this is an *image-space* angle; use estimate_lean_from_silhouette
    for a world-frame lean.
    """
    if side_frame_edges is None or len(side_frame_edges) == 0:
        raise ValueError("No edges provided — cannot estimate lean angle")

    angles = []
    for x1, y1, x2, y2 in side_frame_edges:
        dx, dy = x2 - x1, y2 - y1
        if dy == 0:
            continue
        angles.append(np.degrees(np.arctan2(dx, dy)))

    if not angles:
        raise ValueError("All provided edges were horizontal — no vertical reference found")
    return float(np.median(angles))


def extract_tower_edges(side_frame: np.ndarray, min_length_frac: float = 0.12) -> np.ndarray:
    """Near-vertical line segments from the side view, via Canny + Hough."""
    gray = cv2.cvtColor(side_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    min_len = int(side_frame.shape[0] * min_length_frac)
    segments = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=50,
                               minLineLength=min_len, maxLineGap=8)
    if segments is None:
        return np.empty((0, 4))
    segs = segments.reshape(-1, 4)
    keep = []
    for x1, y1, x2, y2 in segs:
        if abs(y2 - y1) > abs(x2 - x1):      # near-vertical only
            keep.append([x1, y1, x2, y2])
    return np.array(keep, dtype=float) if keep else np.empty((0, 4))


# --------------------------------------------------------------------------- #
# Occupancy
# --------------------------------------------------------------------------- #
@dataclass
class OccupancyModel:
    """Brightness threshold separating a block face from an empty recess.

    Fitted from Step 1 reference photos of a known-full tower: every sampled
    patch there is by definition occupied, so the occupied distribution is
    directly measurable and the threshold sits a fixed number of standard
    deviations below its mean.
    """
    threshold: float
    occupied_mean: float
    occupied_std: float
    n_samples: int
    sigma: float = 4.0

    def classify(self, value: float) -> bool:
        return value >= self.threshold

    def margin(self, value: float) -> float:
        scale = max(self.occupied_std, 1.0)
        return abs(value - self.threshold) / scale

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @staticmethod
    def load(path: Path) -> "OccupancyModel":
        return OccupancyModel(**json.loads(Path(path).read_text()))


PATCH_PX = 16
PATCH_SHRINK = 0.72


def _sample_patch_value(frame: np.ndarray, quad_px: np.ndarray) -> Optional[float]:
    """Median grey value of a slot's face rectangle, rectified to a square.

    The quad is shrunk about its centre before sampling so that seam lines and
    a pixel or two of projection error do not drag the statistic toward the
    neighbouring block.
    """
    if not np.isfinite(quad_px).all():
        return None
    centre = quad_px.mean(axis=0)
    shrunk = (centre + (quad_px - centre) * PATCH_SHRINK).astype(np.float32)

    h, w = frame.shape[:2]
    if (shrunk[:, 0] < -w).any() or (shrunk[:, 0] > 2 * w).any():
        return None

    dst = np.array([[0, 0], [PATCH_PX - 1, 0],
                    [PATCH_PX - 1, PATCH_PX - 1], [0, PATCH_PX - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(shrunk, dst)
    patch = cv2.warpPerspective(frame, M, (PATCH_PX, PATCH_PX), flags=cv2.INTER_AREA)
    return float(np.median(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)))


def sample_slot_values(frame: np.ndarray, calibrator: ArucoFrameCalibrator,
                       num_layers: int, lean_deg: float = 0.0) -> np.ndarray:
    """Per-slot patch brightness, NaN where the region is not sampleable."""
    values = np.full((num_layers, SLOTS_PER_LAYER), np.nan)
    for layer in range(num_layers):
        for slot in range(SLOTS_PER_LAYER):
            quad = calibrator.project(slot_face_quad(layer, slot, lean_deg))
            v = _sample_patch_value(frame, quad)
            if v is not None:
                values[layer, slot] = v
    return values


def fit_occupancy_model(reference_frames: Sequence[np.ndarray],
                        calibrator: ArucoFrameCalibrator, num_layers: int,
                        sigma: float = 4.0) -> OccupancyModel:
    """Fit the occupied-brightness distribution from full-tower references."""
    samples = []
    for frame in reference_frames:
        lean, _ = estimate_lean_from_silhouette(frame, calibrator, num_layers)
        vals = sample_slot_values(frame, calibrator, num_layers, lean)
        samples.append(vals[np.isfinite(vals)])
    if not samples:
        raise ValueError("No reference frames supplied — cannot fit an occupancy model.")

    flat = np.concatenate(samples)
    if flat.size == 0:
        raise ValueError("Reference frames produced no sampleable slot regions.")

    mean, std = float(flat.mean()), float(flat.std())
    return OccupancyModel(
        threshold=mean - sigma * max(std, 1.0),
        occupied_mean=mean, occupied_std=std, n_samples=int(flat.size), sigma=sigma,
    )


def detect_block_occupancy(frame: np.ndarray, calibrator: ArucoFrameCalibrator,
                           num_layers: int, model: OccupancyModel,
                           lean_deg: float = 0.0) -> tuple:
    """Occupancy grid plus per-slot decision margins."""
    values = sample_slot_values(frame, calibrator, num_layers, lean_deg)
    occupancy = np.zeros_like(values, dtype=bool)
    margins = np.zeros_like(values)
    for layer in range(values.shape[0]):
        for slot in range(values.shape[1]):
            v = values[layer, slot]
            if not np.isfinite(v):
                # Unsampleable regions are reported empty with zero margin, so
                # they drag confidence down instead of passing silently.
                continue
            occupancy[layer, slot] = model.classify(v)
            margins[layer, slot] = model.margin(v)
    return occupancy, margins


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #
def get_tower_state(overhead_frame: np.ndarray, side_frame: np.ndarray,
                    side_calibrator: ArucoFrameCalibrator,
                    overhead_calibrator: ArucoFrameCalibrator,
                    num_layers: int, model: OccupancyModel) -> TowerState:
    """Compose the pieces above into one tower state.

    Occupancy comes from the side camera, which is the only view that resolves
    every layer. The overhead camera independently re-reads the top layer and
    that agreement feeds confidence: two views disagreeing about the one thing
    they can both see is the cheapest available signal that the rig has moved.
    """
    lean_deg, iou = estimate_lean_from_silhouette(side_frame, side_calibrator, num_layers)
    occupancy, margins = detect_block_occupancy(side_frame, side_calibrator,
                                                num_layers, model, lean_deg)

    agreement = 1.0
    if overhead_calibrator.is_calibrated():
        top = num_layers - 1
        over_occ, _ = detect_block_occupancy(overhead_frame, overhead_calibrator,
                                             num_layers, model, lean_deg)
        agreement = float((over_occ[top] == occupancy[top]).mean())

    finite = margins[margins > 0]
    margin_score = float(np.clip(np.median(finite) / 6.0, 0.0, 1.0)) if finite.size else 0.0
    confidence = float(np.clip(0.5 * margin_score + 0.3 * min(iou / 0.85, 1.0)
                               + 0.2 * agreement, 0.0, 1.0))

    return TowerState(
        layer_occupancy=occupancy.tolist(),
        lean_angle_deg=float(lean_deg),
        num_layers_detected=int(num_layers),
        confidence=confidence,
    )
