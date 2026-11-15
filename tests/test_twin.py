"""SO-101 digital twin (twin/): the custom-claw scene loads and a short
scripted pick-and-place succeeds. Skips when MuJoCo or the upstream meshes
are missing (python scripts/tools/fetch_so101_assets.py)."""
import importlib.util
import sys
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
    sys.modules[name] = mod          # dataclasses look their module up here
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


@needs_meshes
def test_jenga_push_geometry_and_reach():
    """Fingertip contact boxes match a ~10 mm tip, and every candidate block is reachable."""
    jp = _load("jenga_push")
    tips = jp._tip_boxes()
    for t in tips.values():
        full = 2 * t["half"] * 1000
        assert full[2] < 15, "fingertip taller than a Jenga block cannot push one end-on"
    truth = jp.Truth(jp.TowerSpec.sample(1))
    pos = truth.block_pos()
    for i in jp.CANDIDATES:
        plan = jp.plan_push(truth.arm, pos[i])
        assert plan is not None and plan.ik_err_mm < 1.5, i


@needs_meshes
def test_world_model_rollout_and_gate_rules():
    jp = _load("jenga_push")
    import numpy as np
    wm = jp.WorldModel()
    truth = jp.Truth(jp.TowerSpec.sample(1))
    i = jp.block_index(6, 1)
    obs = dict(pushed_mm=12.0, response_mm=2.0)
    g = wm.p_fall(truth.block_qpos(), [], i, obs, np.random.default_rng(0), samples=4)
    assert 0.0 <= g["p_fall"] <= 1.0 or np.isnan(g["p_fall"])
    assert jp.gate_decision(dict(p_fall=.9, ess=8))[0] is False
    assert jp.gate_decision(dict(p_fall=.0, ess=1))[1] == "uncertain"
    assert jp.gate_decision(dict(p_fall=.1, ess=8))[0] is True
