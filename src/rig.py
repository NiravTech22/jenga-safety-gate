"""Physical rig description: marker layout and tower geometry.

Imported by both the simulation renderer and the perception pipeline, which
is deliberate. On hardware these numbers come from a tape measure and the
printed marker sheet; in simulation they come from the same place the
renderer draws from. If the two ever disagree, perception fails loudly in
the Step 2 validation rather than silently mis-locating blocks.

Coordinate frame (shared by everything downstream):
  origin  centre of the tower's base footprint, on the table surface
  +x, +y  horizontal, aligned with the table edges
  +z      up
Units are metres throughout.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.sim.tower import BLOCK_H_M, BLOCK_L_M, BLOCK_W_M, SLOTS_PER_LAYER


@dataclass(frozen=True)
class MarkerLayout:
    """Four ArUco markers on the table plane, one per corner of the working
    area, ids 0-3. Four is the minimum for a well-conditioned pose estimate
    and gives one spare if a gripper occludes a marker mid-episode.
    """
    marker_ids: tuple = (0, 1, 2, 3)
    marker_length_m: float = 0.03
    # Centres, in marker_ids order: -x-y, +x-y, +x+y, -x+y.
    offset_m: float = 0.115
    dictionary_name: str = "DICT_4X4_50"

    def centres(self) -> np.ndarray:
        d = self.offset_m
        return np.array([[-d, -d, 0.0], [d, -d, 0.0], [d, d, 0.0], [-d, d, 0.0]])

    def corners_world(self, index: int) -> np.ndarray:
        """The four corners of one marker, in the order cv2.aruco reports
        them: top-left, top-right, bottom-right, bottom-left, where "top"
        is +y in the world frame."""
        cx, cy, _ = self.centres()[index]
        h = self.marker_length_m / 2.0
        return np.array([
            [cx - h, cy + h, 0.0],
            [cx + h, cy + h, 0.0],
            [cx + h, cy - h, 0.0],
            [cx - h, cy - h, 0.0],
        ])

    def all_corners_world(self) -> np.ndarray:
        return np.vstack([self.corners_world(i) for i in range(len(self.marker_ids))])


def layer_axis(layer: int) -> int:
    """0 = blocks run along x with slots offset in y; 1 = the alternating
    layer. Must match src.sim.tower._layer_axis."""
    return layer % 2


def slot_offsets_m() -> np.ndarray:
    return np.array([-BLOCK_W_M, 0.0, BLOCK_W_M])


def block_corners_world(layer: int, slot: int, lean_deg: float = 0.0) -> np.ndarray:
    """Eight corners of one block, with the tower's lean applied.

    Lean is modelled as a shear about the base along +x, matching the
    centre-of-mass model in src.sim.tower: a point at height z is displaced
    by z * tan(lean).
    """
    axis = layer_axis(layer)
    offset = slot_offsets_m()[slot]
    z0, z1 = layer * BLOCK_H_M, (layer + 1) * BLOCK_H_M

    if axis == 0:
        x0, x1 = -BLOCK_L_M / 2, BLOCK_L_M / 2
        y0, y1 = offset - BLOCK_W_M / 2, offset + BLOCK_W_M / 2
    else:
        x0, x1 = offset - BLOCK_W_M / 2, offset + BLOCK_W_M / 2
        y0, y1 = -BLOCK_L_M / 2, BLOCK_L_M / 2

    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float64)
    return apply_lean(corners, lean_deg)


def apply_lean(points: np.ndarray, lean_deg: float) -> np.ndarray:
    pts = np.array(points, dtype=np.float64, copy=True).reshape(-1, 3)
    if lean_deg:
        pts[:, 0] += pts[:, 2] * np.tan(np.radians(lean_deg))
    return pts


BOX_FACES = (
    (0, 1, 2, 3),   # bottom
    (4, 5, 6, 7),   # top
    (0, 1, 5, 4),   # -y
    (2, 3, 7, 6),   # +y
    (1, 2, 6, 5),   # +x
    (3, 0, 4, 7),   # -x
)


def visible_face_for_slot(layer: int) -> str:
    """Which vertical face of the tower resolves this layer's three slots.

    A layer's blocks are offset across their own axis, so the face you can
    count them on is the one perpendicular to that offset. Layers alternate,
    so the two faces alternate with them. This is why the side camera sits
    at a corner rather than square-on to one face: from 45 degrees both the
    +x and +y faces are visible, and between them every layer is resolvable.
    """
    return "+x" if layer_axis(layer) == 0 else "+y"


def slot_face_quad(layer: int, slot: int, lean_deg: float = 0.0) -> np.ndarray:
    """The four corners of the outward-facing rectangle of one block, on
    whichever face resolves that block's layer. This is the patch the
    occupancy detector samples."""
    corners = block_corners_world(layer, slot, lean_deg)
    face = visible_face_for_slot(layer)
    idx = (1, 2, 6, 5) if face == "+x" else (2, 3, 7, 6)
    return corners[list(idx)]


def tower_height_m(num_layers: int) -> float:
    return num_layers * BLOCK_H_M


__all__ = [
    "MarkerLayout", "layer_axis", "slot_offsets_m", "block_corners_world",
    "apply_lean", "BOX_FACES", "visible_face_for_slot", "slot_face_quad",
    "tower_height_m", "BLOCK_L_M", "BLOCK_W_M", "BLOCK_H_M", "SLOTS_PER_LAYER",
]
