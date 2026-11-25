"""Gamepad teleop safety logic with a fake bus and scripted pad states (no hardware)."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts/tools"))
_spec = importlib.util.spec_from_file_location("gamepad_teleop", ROOT / "scripts/tools/gamepad_teleop.py")
gt = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(gt)
from src.hardware.kinematics import JOINT_NAMES  # noqa: E402

LIMITS = {n: (1000, 3000) for n in JOINT_NAMES}


class FakeBus:
    def __init__(self):
        self.pos = {n: 2000 for n in JOINT_NAMES}
        self.writes, self.torque = [], None

    def sync_read(self, reg, normalize=False):
        return dict(self.pos) if reg != "Present_Current" else {n: 3 for n in JOINT_NAMES}

    def sync_write(self, reg, values, normalize=True):
        assert reg == "Goal_Position" and normalize is False
        self.writes.append(dict(values))

    def enable_torque(self):
        self.torque = True

    def disable_torque(self):
        self.torque = False


def pad(axes=(0,) * 6, buttons=(), hat=(0, 0)):
    b = [0] * 11
    for i in buttons:
        b[i] = 1
    return dict(axes=list(axes), buttons=b, hats=[hat])


def make(tmp_path, dry=False):
    bus = FakeBus()
    return bus, gt.Teleop(bus, LIMITS, speed=250, dry_run=dry, out=str(tmp_path / "t.json"), cal_version="c")


def test_starts_disarmed_and_sticks_do_nothing(tmp_path):
    bus, t = make(tmp_path)
    t.step(pad(axes=(1, 0, 0, 0, 0, 0)), .1)
    assert not t.armed and bus.writes == [] and bus.torque is None


def test_arming_holds_present_pose_then_moves_at_capped_speed(tmp_path):
    bus, t = make(tmp_path)
    t.step(pad(buttons=[gt.BUTTONS["start"]]), .1)
    assert t.armed and bus.torque is True and bus.writes[0]["shoulder_pan"] == 2000   # no jump
    t.step(pad(axes=(1, 0, 0, 0, 0, 0)), .1)                                          # full left stick X
    assert bus.writes[-1]["shoulder_pan"] == 2025                                     # 250 ticks/s * 0.1 s


def test_goals_stay_inside_calibrated_limits(tmp_path):
    bus, t = make(tmp_path)
    t.step(pad(buttons=[gt.BUTTONS["start"]]), .1)
    for _ in range(200):
        t.step(pad(axes=(1, 0, 0, 0, 0, 0)), .1)
    assert bus.writes[-1]["shoulder_pan"] == 3000


def test_freeze_stops_and_limp_drops_torque(tmp_path):
    bus, t = make(tmp_path)
    t.step(pad(buttons=[gt.BUTTONS["start"]]), .1)
    t.step(pad(buttons=[gt.BUTTONS["freeze"]]), .1)
    n = len(bus.writes)
    t.step(pad(axes=(1, 0, 0, 0, 0, 0)), .1)
    assert not t.armed and len(bus.writes) == n and bus.torque is True
    t.step(pad(buttons=[gt.BUTTONS["limp"]]), .1)
    assert bus.torque is False


def test_dry_run_never_writes_or_enables_torque(tmp_path):
    bus, t = make(tmp_path, dry=True)
    t.step(pad(buttons=[gt.BUTTONS["start"]]), .1)
    t.step(pad(axes=(1, 1, 0, 1, 1, 0), buttons=[5]), .1)
    t.step(pad(buttons=[gt.BUTTONS["limp"]]), .1)
    assert bus.writes == [] and bus.torque is None and t.goal["shoulder_pan"] > 2000


def test_publishes_telemetry_the_recorder_accepts(tmp_path):
    from src.hardware.passive import TelemetryFile
    bus, t = make(tmp_path)
    t.publish(); t.publish()
    v = TelemetryFile(tmp_path / "t.json", JOINT_NAMES, gt.UNITS, "c").read()
    assert v["sequence_id"] == 1 and len(v["current_ma"]) == 6 and v["source"] == "gamepad_teleop"


def test_held_button_acts_once(tmp_path):
    bus, t = make(tmp_path)
    t.step(pad(buttons=[gt.BUTTONS["start"]]), .1)
    t.step(pad(buttons=[gt.BUTTONS["freeze"]]), .1)
    n = len(bus.writes)
    for _ in range(5):
        t.step(pad(buttons=[gt.BUTTONS["freeze"]]), .1)          # still held: no repeat
    assert len(bus.writes) == n
    t.step(pad(buttons=[gt.BUTTONS["start"]]), .1)               # new press re-arms
    assert t.armed


def test_freeze_disarms_even_if_the_bus_fails(tmp_path):
    bus, t = make(tmp_path)
    t.step(pad(buttons=[gt.BUTTONS["start"]]), .1)
    real = bus.sync_read
    bus.sync_read = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no status packet"))
    try:
        t.step(pad(buttons=[gt.BUTTONS["freeze"]]), .1)
    except RuntimeError:
        pass                                        # main() logs and continues
    bus.sync_read = real
    n = len(bus.writes)
    t.step(pad(axes=(1, 0, 0, 0, 0, 0), buttons=[gt.BUTTONS["freeze"]]), .1)   # B still held + stick
    t.step(pad(axes=(1, 0, 0, 0, 0, 0)), .1)                                    # B released + stick
    assert not t.armed and len(bus.writes) == n


def test_failed_limp_is_retried_while_held(tmp_path):
    bus, t = make(tmp_path)
    t.step(pad(buttons=[gt.BUTTONS["start"]]), .1)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("no status packet")
        bus.torque = False
    bus.disable_torque = flaky
    try:
        t.step(pad(buttons=[gt.BUTTONS["limp"]]), .1)
    except RuntimeError:
        pass
    t.step(pad(buttons=[gt.BUTTONS["limp"]]), .1)                               # still held: retry
    assert bus.torque is False and calls["n"] == 2 and not t.armed
    t.step(pad(buttons=[gt.BUTTONS["limp"]]), .1)                               # done: no more calls
    assert calls["n"] == 2


def test_limp_still_acts_after_b_and_back_pressed_together(tmp_path):
    bus, t = make(tmp_path)
    L, B, S = gt.BUTTONS["limp"], gt.BUTTONS["freeze"], gt.BUTTONS["start"]
    t.step(pad(buttons=[S]), .1); t.step(pad(), .1)
    t.step(pad(buttons=[L]), .1); t.step(pad(), .1)          # limp once
    t.step(pad(buttons=[S]), .1); t.step(pad(), .1)          # re-arm: torque on
    assert bus.torque is True
    t.step(pad(buttons=[B, L]), .1)                          # both together: freeze wins
    t.step(pad(buttons=[L]), .1)                             # release B, keep BACK
    assert bus.torque is False
