import numpy as np
import pytest

from src.hardware.passive import CAMERA_IDS, black_cameras, resolve_camera


def test_index_and_path_resolution(tmp_path):
    assert resolve_camera("2", "overhead") == 2 and resolve_camera(0, "wrist") == 0
    real = tmp_path / "video2"; real.touch()
    link = tmp_path / "usb-Jieli-video-index0"; link.symlink_to(real)
    assert resolve_camera(str(link), "overhead") == str(real)       # by-id link -> real device


def test_missing_camera_is_a_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="not found"):
        resolve_camera(str(tmp_path / "nope"), "wrist")


def test_defaults_are_distinct_by_id_links():
    assert CAMERA_IDS["overhead"] != CAMERA_IDS["wrist"]
    assert all(v.startswith("/dev/v4l/by-id/") for v in CAMERA_IDS.values())


def test_black_camera_detected():
    black = np.ones((72, 128, 3), np.uint8)            # mean 1, like the swapped demo
    room = np.full((72, 128, 3), 30, np.uint8)
    snap = {"overhead": (black, 0.0, 1, 0), "wrist": (room, 0.0, 1, 0)}
    assert black_cameras(snap) == ["overhead"]
