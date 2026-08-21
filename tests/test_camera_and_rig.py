"""Geometry tests for the sim camera model and the shared rig description.

These are the pieces perception is scored against, so an error here would
make Step 2 validate against the wrong thing and look like a success.
"""
import cv2
import numpy as np
import pytest

from src.rig import (
    BLOCK_H_M,
    BLOCK_L_M,
    BLOCK_W_M,
    MarkerLayout,
    apply_lean,
    block_corners_world,
    layer_axis,
    slot_face_quad,
    slot_offsets_m,
    visible_face_for_slot,
)
from src.sim.camera import CameraIntrinsics, PinholeCamera
from src.sim.tower import _layer_axis, _slot_offsets_m


def make_camera():
    intr = CameraIntrinsics.from_fov(640, 480, 55.0)
    return PinholeCamera.look_at(intr, eye=(0.34, 0.30, 0.13), target=(0.0, 0.0, 0.09))


def test_projection_matches_opencv():
    """The renderer must project the way cv2.projectPoints does, or the pose
    perception recovers with solvePnP will not describe the rendered scene."""
    cam = make_camera()
    pts = np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0.02], [0, 0, 0.18]], float)
    rvec, tvec = cam.rvec_tvec()
    expected, _ = cv2.projectPoints(pts, rvec, tvec,
                                    cam.intrinsics.matrix(), cam.intrinsics.distortion())
    assert np.allclose(cam.project(pts), expected.reshape(-1, 2), atol=1e-9)


def test_look_at_places_camera_at_eye():
    cam = make_camera()
    assert np.allclose(cam.position, [0.34, 0.30, 0.13], atol=1e-9)


def test_points_behind_camera_are_nan_not_wrapped():
    """A point behind the lens must not silently project to a valid-looking
    pixel somewhere in frame."""
    cam = make_camera()
    behind = np.array([[2.0, 2.0, 0.0]])
    assert np.isnan(cam.project(behind)).all()


def test_degenerate_look_at_raises():
    intr = CameraIntrinsics.from_fov(64, 48, 55.0)
    with pytest.raises(ValueError):
        PinholeCamera.look_at(intr, eye=(0, 0, 0.1), target=(0, 0, 0.1))
    with pytest.raises(ValueError):
        PinholeCamera.look_at(intr, eye=(0, 0, 0.5), target=(0, 0, 0.0), up=(0, 0, 1))


def test_rig_geometry_matches_sim_geometry():
    """rig.py and tower.py both encode layer orientation and slot offsets.
    If they drift apart, perception samples the wrong regions while the sim
    reports the old ones, and validation passes on a lie."""
    assert np.allclose(slot_offsets_m(), _slot_offsets_m())
    for layer in range(6):
        assert layer_axis(layer) == _layer_axis(layer)


def test_block_corners_have_correct_extents():
    corners = block_corners_world(0, 1)
    assert np.isclose(corners[:, 2].min(), 0.0)
    assert np.isclose(corners[:, 2].max(), BLOCK_H_M)
    # Layer 0 runs along x, so it is long in x and narrow in y.
    assert np.isclose(corners[:, 0].max() - corners[:, 0].min(), BLOCK_L_M)
    assert np.isclose(corners[:, 1].max() - corners[:, 1].min(), BLOCK_W_M)


def test_alternating_layers_swap_axis_and_visible_face():
    even = block_corners_world(0, 1)
    odd = block_corners_world(1, 1)
    assert np.isclose(even[:, 0].max() - even[:, 0].min(), BLOCK_L_M)
    assert np.isclose(odd[:, 1].max() - odd[:, 1].min(), BLOCK_L_M)
    assert visible_face_for_slot(0) == "+x"
    assert visible_face_for_slot(1) == "+y"


def test_lean_shears_proportionally_to_height():
    lean = 5.0
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.10]])
    sheared = apply_lean(pts, lean)
    assert np.isclose(sheared[0, 0], 0.0)
    assert np.isclose(sheared[1, 0], 0.10 * np.tan(np.radians(lean)))


def test_slot_face_quads_are_distinct_per_slot():
    """The three slots of a layer must map to three non-overlapping patches,
    otherwise occupancy for one block would be read off another."""
    quads = [slot_face_quad(0, s) for s in range(3)]
    centres = np.array([q.mean(axis=0) for q in quads])
    spread = centres[:, 1]          # layer 0 offsets in y
    assert len(set(np.round(spread, 4))) == 3
    assert np.allclose(np.diff(np.sort(spread)), BLOCK_W_M)


def test_marker_corners_are_square_and_centred():
    layout = MarkerLayout()
    for i in range(4):
        c = layout.corners_world(i)
        assert np.allclose(c[:, 2], 0.0)                     # markers lie on the table
        side_a = np.linalg.norm(c[1] - c[0])
        side_b = np.linalg.norm(c[2] - c[1])
        assert np.isclose(side_a, layout.marker_length_m)
        assert np.isclose(side_b, layout.marker_length_m)
        assert np.allclose(c.mean(axis=0), layout.centres()[i])
