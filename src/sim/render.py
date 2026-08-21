"""Renders the two camera views of a simulated tower.

Output is deliberately plain: flat-shaded boxes, a dark interior core so a
removed block reads as a recess, four ArUco markers on the table, and
per-frame lighting jitter plus sensor noise. It is not trying to look
photographic. It is trying to be geometrically honest, so that the Step 2
perception code has to do the same pose recovery and region sampling it will
do on real frames.

Camera placement follows the rig described in PLAN.md Step 1: one overhead,
one side. The side camera sits at a corner rather than square-on, because a
layer's three slots are only countable on the face perpendicular to their
offset, and those faces alternate with layer parity. From 45 degrees both
faces are in view, so every layer is resolvable. See rig.visible_face_for_slot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.rig import (
    BLOCK_H_M,
    BLOCK_L_M,
    BOX_FACES,
    layer_axis,
    MarkerLayout,
    SLOTS_PER_LAYER,
    apply_lean,
    block_corners_world,
    tower_height_m,
)
from src.sim.camera import CameraIntrinsics, PinholeCamera
from src.sim.tower import TowerState

# BGR, matching OpenCV's channel order.
COLOR_BACKDROP = (46, 44, 42)
COLOR_TABLE = (188, 190, 194)
COLOR_BLOCK = (112, 163, 208)
COLOR_CORE = (28, 32, 40)
LIGHT_DIR = np.array([-0.35, -0.55, 1.0])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)


@dataclass
class RenderConfig:
    width: int = 640
    height: int = 480
    horizontal_fov_deg: float = 55.0
    table_half_extent_m: float = 0.40
    noise_sigma: float = 2.5
    # Lighting drifts slightly frame to frame. Fixed lighting is a Step 1
    # exit condition, but "fixed" on real hardware still means a few percent.
    brightness_jitter: float = 0.03
    draw_markers: bool = True


@dataclass
class Rig:
    overhead: PinholeCamera
    side: PinholeCamera
    markers: MarkerLayout = field(default_factory=MarkerLayout)
    config: RenderConfig = field(default_factory=RenderConfig)


def default_rig(config: Optional[RenderConfig] = None,
                markers: Optional[MarkerLayout] = None) -> Rig:
    cfg = config or RenderConfig()
    intr = CameraIntrinsics.from_fov(cfg.width, cfg.height, cfg.horizontal_fov_deg)
    tower_mid = tower_height_m(12) / 2.0

    # Overhead: high and slightly off-axis. A perfectly nadir view makes the
    # marker-plane homography poorly conditioned in one direction.
    overhead = PinholeCamera.look_at(
        intr, eye=(0.045, -0.06, 0.62), target=(0.0, 0.0, 0.02), up=(0.0, 1.0, 0.0)
    )
    # Side: corner view, low elevation, sees the +x and +y faces.
    side = PinholeCamera.look_at(
        intr, eye=(0.34, 0.30, 0.13), target=(0.0, 0.0, tower_mid), up=(0.0, 0.0, 1.0)
    )
    return Rig(overhead=overhead, side=side, markers=markers or MarkerLayout(), config=cfg)


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
NEAR_PLANE_M = 1e-3


def _clip_to_near_plane(camera: PinholeCamera, quad_world: np.ndarray) -> Optional[np.ndarray]:
    """Sutherland-Hodgman clip of a world-space polygon against the camera's
    near plane, returning pixel coordinates.

    Without this, any polygon with a vertex behind the camera projects to NaN
    and silently disappears. That is how the table dropped out of the side
    view: it extends past the camera, so two of its corners were behind the
    lens.
    """
    cam = camera.to_camera_frame(quad_world)
    out = []
    n = len(cam)
    for i in range(n):
        cur, nxt = cam[i], cam[(i + 1) % n]
        cur_in, nxt_in = cur[2] > NEAR_PLANE_M, nxt[2] > NEAR_PLANE_M
        if cur_in:
            out.append(cur)
        if cur_in != nxt_in:
            span = nxt[2] - cur[2]
            if abs(span) > 1e-12:
                out.append(cur + (nxt - cur) * ((NEAR_PLANE_M - cur[2]) / span))
    if len(out) < 3:
        return None

    cam_clipped = np.array(out)
    k = camera.intrinsics
    px = np.empty((cam_clipped.shape[0], 2))
    px[:, 0] = k.fx * cam_clipped[:, 0] / cam_clipped[:, 2] + k.cx
    px[:, 1] = k.fy * cam_clipped[:, 1] / cam_clipped[:, 2] + k.cy
    return px


def _fill_world_poly(img: np.ndarray, camera: PinholeCamera, quad_world: np.ndarray,
                     color, outline: bool = False) -> None:
    px = _clip_to_near_plane(camera, quad_world)
    if px is None or not np.isfinite(px).all():
        return
    pts = px.round().astype(np.int32)
    cv2.fillConvexPoly(img, pts, color, lineType=cv2.LINE_AA)
    if outline:
        # Block seams have to survive into the image: an edge detector with
        # nothing to find would make the Step 2 occupancy check meaningless.
        edge = tuple(float(max(0.0, c * 0.62)) for c in color)
        cv2.polylines(img, [pts], isClosed=True, color=edge, thickness=1,
                      lineType=cv2.LINE_AA)


def _fill_poly(img: np.ndarray, pts_px: np.ndarray, color) -> None:
    if not np.isfinite(pts_px).all():
        return
    cv2.fillConvexPoly(img, pts_px.round().astype(np.int32), color, lineType=cv2.LINE_AA)


def _shade(color, normal_world: np.ndarray) -> tuple:
    lambert = abs(float(np.dot(normal_world, LIGHT_DIR)))
    factor = 0.55 + 0.45 * lambert
    return tuple(float(np.clip(c * factor, 0, 255)) for c in color)


def _box_faces(corners: np.ndarray, colour) -> list:
    """Front-facing quads of one box, as (centroid, normal, quad, colour).

    Normals are flipped outward relative to the box centre, so the caller can
    cull back faces and avoid painter's-algorithm artefacts on concave
    arrangements."""
    centre = corners.mean(axis=0)
    out = []
    for idx in BOX_FACES:
        quad = corners[list(idx)]
        c = quad.mean(axis=0)
        n = np.cross(quad[1] - quad[0], quad[2] - quad[0])
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm
        if np.dot(n, c - centre) < 0:
            n = -n
        out.append((c, n, quad, colour))
    return out


def _draw_table(img: np.ndarray, camera: PinholeCamera, cfg: RenderConfig) -> None:
    e = cfg.table_half_extent_m
    quad = np.array([[-e, -e, 0.0], [e, -e, 0.0], [e, e, 0.0], [-e, e, 0.0]])
    _fill_world_poly(img, camera, quad, COLOR_TABLE)


def _draw_markers(img: np.ndarray, camera: PinholeCamera, markers: MarkerLayout) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, markers.dictionary_name)
    )
    size_px = 220
    for i, marker_id in enumerate(markers.marker_ids):
        tile = cv2.aruco.generateImageMarker(dictionary, int(marker_id), size_px)
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)

        dst = camera.project(markers.corners_world(i))
        if not np.isfinite(dst).all():
            continue
        src = np.array([[0, 0], [size_px - 1, 0],
                        [size_px - 1, size_px - 1], [0, size_px - 1]], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))

        h, w = img.shape[:2]
        warped = cv2.warpPerspective(tile, H, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(np.full((size_px, size_px), 255, np.uint8), H, (w, h),
                                   flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
        np.copyto(img, warped, where=mask[:, :, None].astype(bool))


def _cavity_corners(layer: int, slot: int, lean_deg: float,
                    inset: float = 0.003) -> np.ndarray:
    """The recess left behind by a removed block.

    Modelled as the block's own volume pulled back along the axis it is
    visible from, so the gap reads as shadowed depth rather than as a hole
    straight through the tower.

    The inset is small on purpose. At 7 mm the recess was occluded by the
    courses above and below when viewed from the side camera's grazing angle,
    so the occupancy sampler read the neighbouring block's face and scored an
    empty slot as occupied. That is faithful to how a deep recess behaves, but
    a removed Jenga block does not leave a shallow dent: it leaves a hole you
    look into, whose dark interior sits near the plane of the block faces.
    Values between 1.5 and 5 mm all reconstruct 1080/1080 slots correctly.
    """
    corners = block_corners_world(layer, slot, 0.0)
    axis = 0 if layer_axis(layer) == 0 else 1
    lo, hi = corners[:, axis].min(), corners[:, axis].max()
    corners[corners[:, axis] <= lo + 1e-9, axis] = lo + inset
    corners[corners[:, axis] >= hi - 1e-9, axis] = hi - inset
    return apply_lean(corners, lean_deg)


def _collect_faces(state: TowerState) -> list:
    """Every standing block, plus a recessed cavity for each removed one.

    An earlier version drew a single dark box spanning the whole tower
    interior. That sorted incorrectly: painter's ordering uses centroid
    depth, and one large face's centroid is not comparable with those of
    many small ones, so the core painted over blocks in the lower layers.
    Per-slot cavities are the same size as the blocks they sit among, so
    depth ordering is well behaved.
    """
    num_layers = state.occupancy.shape[0]
    faces = []
    for layer in range(num_layers):
        for slot in range(SLOTS_PER_LAYER):
            if state.occupancy[layer, slot]:
                corners = block_corners_world(layer, slot, state.lean_deg)
                faces.extend(_box_faces(corners, COLOR_BLOCK))
            else:
                faces.extend(_box_faces(_cavity_corners(layer, slot, state.lean_deg),
                                        COLOR_CORE))
    return faces


def _render_view(camera: PinholeCamera, state: TowerState, rig: Rig,
                 rng: np.random.Generator) -> np.ndarray:
    cfg = rig.config
    img = np.full((cfg.height, cfg.width, 3), COLOR_BACKDROP, dtype=np.uint8)

    _draw_table(img, camera, cfg)
    if cfg.draw_markers:
        _draw_markers(img, camera, rig.markers)

    if state.toppled:
        # A collapsed tower is not a leaning tower. Draw scattered blocks flat
        # on the table so downstream code never sees a tidy structure in a
        # frame labelled as a collapse.
        _draw_collapse(img, camera, state, rng)
    else:
        eye = camera.position
        faces = [f for f in _collect_faces(state) if np.dot(f[1], eye - f[0]) > 0]
        faces.sort(key=lambda f: camera.depth_of(f[0]), reverse=True)
        for centroid, normal, quad, colour in faces:
            _fill_world_poly(img, camera, quad, _shade(colour, normal), outline=True)

    return _apply_sensor_effects(img, cfg, rng)


def _draw_collapse(img: np.ndarray, camera: PinholeCamera, state: TowerState,
                   rng: np.random.Generator) -> None:
    n_blocks = int(state.occupancy.sum())
    faces = []
    for _ in range(n_blocks):
        cx, cy = rng.uniform(-0.11, 0.11, 2)
        angle = rng.uniform(0, np.pi)
        hl, hw = BLOCK_L_M / 2, BLOCK_W_M / 2
        base = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        xy = base @ rot.T + np.array([cx, cy])
        z0 = 0.0
        z1 = BLOCK_H_M
        corners = np.array([[x, y, z] for z in (z0, z1) for x, y in xy])
        corners = corners[[0, 1, 2, 3, 4, 5, 6, 7]]
        faces.extend(_box_faces(corners, COLOR_BLOCK))

    eye = camera.position
    faces = [f for f in faces if np.dot(f[1], eye - f[0]) > 0]
    faces.sort(key=lambda f: camera.depth_of(f[0]), reverse=True)
    for centroid, normal, quad, colour in faces:
        _fill_world_poly(img, camera, quad, _shade(colour, normal), outline=True)


def _apply_sensor_effects(img: np.ndarray, cfg: RenderConfig,
                          rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float32)
    if cfg.brightness_jitter:
        out *= 1.0 + rng.normal(0.0, cfg.brightness_jitter)
    if cfg.noise_sigma:
        out += rng.normal(0.0, cfg.noise_sigma, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def render_frame_pair(state: TowerState, rig: Optional[Rig] = None,
                      rng: Optional[np.random.Generator] = None) -> tuple:
    """Returns (overhead_bgr, side_bgr) for one tower state."""
    rig = rig or default_rig()
    rng = rng or np.random.default_rng(0)
    return (_render_view(rig.overhead, state, rig, rng),
            _render_view(rig.side, state, rig, rng))


def render_empty_table(rig: Optional[Rig] = None,
                       rng: Optional[np.random.Generator] = None) -> tuple:
    """The empty-table reference pair from PLAN.md Step 1.

    Calibration is done against these rather than against a frame containing
    a tower, because the tower occludes markers. On hardware this is the shot
    you take once, before building anything.
    """
    rig = rig or default_rig()
    rng = rng or np.random.default_rng(0)
    blank = TowerState(occupancy=np.zeros((0, SLOTS_PER_LAYER), dtype=bool),
                       lean_deg=0.0, tightness=np.zeros((0, SLOTS_PER_LAYER)))
    return (_render_view(rig.overhead, blank, rig, rng),
            _render_view(rig.side, blank, rig, rng))


def save_frame_pair(overhead: np.ndarray, side: np.ndarray, out_dir: Path,
                    stem: str) -> tuple:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_over = out_dir / f"{stem}_overhead.png"
    p_side = out_dir / f"{stem}_side.png"
    cv2.imwrite(str(p_over), overhead)
    cv2.imwrite(str(p_side), side)
    return p_over, p_side
