"""SO-101 digital twin (twin/): the custom-claw scene loads and a short
scripted pick-and-place succeeds. Skips when MuJoCo or the upstream meshes
are missing (python scripts/tools/fetch_so101_assets.py)."""
import importlib.util
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

TWIN = Path(__file__).resolve().parents[1] / "twin"
MESHES = TWIN / "models/meshes"
needs_meshes = pytest.mark.skipif(not (MESHES / "base_so101_v2.stl").exists(),
                                  reason="upstream SO-101 meshes not downloaded")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TWIN / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_meshes
def test_pickplace_succeeds(tmp_path, monkeypatch, capsys):
    pp = _load("so101_pickplace")
    monkeypatch.setattr(pp, "RUN_LOG", tmp_path / "run.json")
    model, data, top = pp.workspace_setup()
    plan = pp.plan_motion(model, data, top)
    report = pp.perform_auto(model, data, top, plan)
    assert report["success"], report
    assert report["placement_xy_error_m"] < 0.005


def test_camera_conventions_selftest():
    """Synthetic calibration round-trip (intrinsics, extrinsics, tower pose, MuJoCo camera)."""
    pytest.importorskip("cv2")
    jv = _load("jenga_vision")
    import argparse, tempfile
    jv.cmd_selftest(argparse.Namespace(workdir=tempfile.mkdtemp()))


def test_tower_yaw_average_handles_wraparound():
    jv = _load("jenga_vision")
    assert abs(abs(jv.circular_mean_deg([179.9] * 15 + [-179.9] * 15)) - 180) < 1e-6
    assert abs(jv.circular_mean_deg([10, 11, 12]) - 11) < 1e-9
