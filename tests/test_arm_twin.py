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
        for q in (-.5, 0.0, .01, .7) if n != "gripper" else (0.0, .01, .03):  # noqa
            assert abs(jm.q(n, jm.ticks(n, q)) - q) < 1e-9
    assert abs(jm.q("shoulder_pan", 2048 + 1024) - math.pi / 2) < 1e-9     # quarter turn
    assert jm.q("elbow_flex", 2100 + 1024) == pytest.approx(-math.pi / 2)  # flipped joint


def test_zero_ticks_put_the_model_in_its_zero_pose():
    arm, jm = at.Arm(), at.JointMap()
    at.apply_telemetry(arm, jm, {"joint_state": [2048.0] * 5 + [jm.claw["open_tick"]]})   # open claw = slide 0
    assert all(abs(v) < 1e-9 for v in arm.get_q().values())
    x, y, z = arm.tip_xyz()
    assert 0.1 < math.hypot(x, y) < 0.6 and z > -0.1                       # tip somewhere in front/above


def test_joint_motion_moves_the_claw_tip():
    arm, jm = at.Arm(), at.JointMap()
    at.apply_telemetry(arm, jm, {"joint_state": [2048.0] * 6})
    axis = arm.d.xanchor[arm.m.joint("shoulder_pan").id].copy()   # angles about the pan axis
    p0 = arm.tip_world() - axis
    at.apply_telemetry(arm, jm, {"joint_state": [2048.0 + 512] + [2048.0] * 5})   # pan +45 deg
    p1 = arm.tip_world() - axis
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


def test_claw_direction_matches_the_model():
    import numpy as np
    arm, jm = at.Arm(), at.JointMap({"gripper": {"closed_tick": 2000, "open_tick": 3500}})
    gap = {}
    for name, tick in (("closed", 2000), ("open", 3500)):
        at.apply_telemetry(arm, jm, {"joint_state": [2048.0] * 5 + [tick]})
        L, R = (arm.d.site_xpos[i] for i in arm.pads)
        gap[name] = np.linalg.norm(L - R)
    assert gap["open"] > gap["closed"] + 0.04                         # 88 mm vs 28 mm in the model


def test_tip_is_the_pad_midpoint():
    import numpy as np
    arm = at.Arm()
    L, R = (arm.d.site_xpos[i] for i in arm.pads)
    assert np.allclose(arm.tip_world(), (L + R) / 2)


def test_limp_sim_has_no_actuator_force():
    arm, jm = at.Arm(), at.JointMap()
    arm.set_q({n: 0.0 for n in at.JOINT_NAMES})
    bus = at.SimBus(arm, jm)
    bus.enable_torque(); bus.step(.2)
    held = arm.get_q()["shoulder_lift"]
    bus.disable_torque(); bus.step(.5)
    assert abs(arm.d.actuator_force[:5]).max() < 1e-9
    assert abs(arm.get_q()["shoulder_lift"] - held) > 0.02           # gravity takes it, as on the real arm


def test_zero_is_the_middle_of_the_lerobot_range(tmp_path):
    import json
    cal = {n: dict(id=i + 1, drive_mode=0, homing_offset=0, range_min=lo, range_max=hi)
           for i, (n, lo, hi) in enumerate([("shoulder_pan", 1328, 2776), ("shoulder_lift", 1552, 3877),
                                           ("elbow_flex", 269, 2521), ("wrist_flex", 858, 3287),
                                           ("wrist_roll", 0, 4095), ("gripper", 2042, 3549)])}
    cal["wrist_flex"]["drive_mode"] = 1
    p = tmp_path / "jenga_follower.json"; p.write_text(json.dumps(cal))
    jm = at.JointMap.from_lerobot(p)
    assert jm.q("shoulder_lift", (1552 + 3877) / 2) == pytest.approx(0)
    assert jm.q("elbow_flex", 1395) == pytest.approx(0)
    assert jm.j["wrist_flex"]["sign"] == -1
    assert jm.claw == {"closed_tick": 2042, "open_tick": 3549}
    # the old 2048 default was ~58 deg off for this arm's shoulder_lift
    assert abs(math.degrees(at.JointMap().q("shoulder_lift", 2714.5))) == pytest.approx(58.6, abs=.5)


def test_saved_map_wins_over_calibration(tmp_path):
    import json
    m = tmp_path / "map.json"; m.write_text(json.dumps({"elbow_flex": {"sign": -1, "zero_tick": 1000}}))
    cal = tmp_path / "cal.json"; cal.write_text("{}")
    assert at.JointMap.load(m, cal).j["elbow_flex"]["zero_tick"] == 1000


def test_leftover_cube_is_hidden():
    arm = at.Arm()
    g = arm.m.geom("grasp_cube_geom").id
    assert arm.m.geom_rgba[g, 3] == 0 and arm.m.geom_contype[g] == 0
