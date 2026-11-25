"""Digital twin joint mapping, telemetry mirror and gamepad-driven sim (no viewer, no hardware)."""
import importlib.util
import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/tools"))
_spec = importlib.util.spec_from_file_location("arm_twin", ROOT / "twin/arm_twin.py")
at = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(at)
import gamepad_teleop as gt  # noqa: E402


def test_ticks_and_angles_round_trip():
    jm = at.JointMap({"elbow_flex": {"sign": -1, "zero_tick": 2100}})
    for n in at.JOINT_NAMES:
        for q in (-.5, 0.0, .01, .7) if n != "gripper" else (0.0, .01, .03):
            assert abs(jm.q(n, jm.ticks(n, q)) - q) < 1e-9
    assert abs(jm.q("shoulder_pan", 2048 + 1024) - math.pi / 2) < 1e-9     # quarter turn
    assert jm.q("elbow_flex", 2100 + 1024) == pytest.approx(-math.pi / 2)  # flipped joint


def test_zero_ticks_put_the_model_in_its_zero_pose():
    arm, jm = at.Arm(), at.JointMap()
    at.apply_telemetry(arm, jm, {"joint_state": [2048.0] * 5 + [jm.claw["closed_tick"]]})
    assert all(abs(v) < 1e-9 for v in arm.get_q().values())
    x, y, z = arm.tip_xyz()
    assert 0.1 < math.hypot(x, y) < 0.6 and z > -0.1                       # tip somewhere in front/above


def test_joint_motion_moves_the_claw_tip():
    arm, jm = at.Arm(), at.JointMap()
    at.apply_telemetry(arm, jm, {"joint_state": [2048.0] * 6})
    axis = arm.d.xanchor[arm.m.joint("shoulder_pan").id].copy()   # angles about the pan axis
    p0 = arm.d.site_xpos[arm.tip] - axis
    at.apply_telemetry(arm, jm, {"joint_state": [2048.0 + 512] + [2048.0] * 5})   # pan +45 deg
    p1 = arm.d.site_xpos[arm.tip] - axis
    ang = math.degrees(math.atan2(p1[1], p1[0]) - math.atan2(p0[1], p0[0]))
    assert abs(abs(ang) - 45) < 3 and abs(p1[2] - p0[2]) < 1e-6   # model pan axis points down: +q is clockwise


def test_gamepad_drives_the_sim_arm(tmp_path):
    arm, jm = at.Arm(), at.JointMap()
    arm.set_q({n: 0.0 for n in at.JOINT_NAMES})
    bus = at.SimBus(arm, jm)
    t = gt.Teleop(bus, arm.joint_ticks_limits(jm), 250, False, str(tmp_path / "t.json"), "sim")
    start = [0.0] * 6
    t.step(dict(axes=start, buttons=[0] * 7 + [1] + [0] * 3, hats=[(0, 0)]), .05)   # START
    assert t.armed and bus.torque
    for _ in range(20):                                       # left stick X for 1 s: pan ~ +22 deg
        t.step(dict(axes=[1, 0, 0, 0, 0, 0], buttons=[0] * 11, hats=[(0, 0)]), .05)
        bus.step(.05)
    assert math.degrees(arm.get_q()["shoulder_pan"]) == pytest.approx(22, abs=4)
    t.publish()
    assert (tmp_path / "t.json").exists()
