#!/usr/bin/env python3
"""Read-only SO-101 telemetry for the episode recorder (replaces the VR teleop hook).

Reads every servo's Present_Position, Goal_Position and Present_Current over the
bus and publishes them with `publish_telemetry` for
`scripts/pipeline/03_record_episode.py --hardware --telemetry ...`.

It never writes a register: no torque change, no configuration, no motion. The
arm keeps whatever torque state it is in (move it by hand with torque off, or let
the scripted controller drive it in another process later, once that exists).

  joint_state = Present_Position   (raw STS3215 ticks, 0-4095; 4096 ticks per turn)
  action      = Goal_Position      (the command the servo is holding: what was actually sent)
  current_ma  = Present_Current    (6.5 mA per unit on the STS3215; the wiggle-force signal)

Usage (same machine as the recorder; both use time.monotonic()):
  python scripts/tools/so101_telemetry.py --port /dev/ttyACM0
  python scripts/tools/so101_telemetry.py --mock      # no arm: synthetic values to test the pipeline
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.hardware.kinematics import JOINT_NAMES  # noqa: E402
from src.hardware.passive import publish_telemetry  # noqa: E402

UNITS = "sts3215_ticks"
CURRENT_MA_PER_UNIT = 6.5
IDS = dict(zip(JOINT_NAMES, range(1, 7)))      # LeRobot SO-101 follower ids 1-6


def open_bus(port):
    """LeRobot's bus class moved between releases; both expose the same read API."""
    try:
        from lerobot.motors import Motor, MotorNormMode
    except ImportError as exc:
        raise SystemExit(f"LeRobot is not installed in this environment ({exc}); pip install lerobot") from exc
    try:
        from lerobot.motors import SerialMotorsBus as Bus          # 0.5+
    except ImportError:
        from lerobot.motors.feetech import FeetechMotorsBus as Bus  # 0.3-0.4
    motors = {n: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for n, i in IDS.items()}
    bus = Bus(port=port, motors=motors)
    bus.connect()                      # handshake = ping only
    return bus


class MockBus:
    def __init__(self):
        self.t0 = time.monotonic()

    def sync_read(self, name, normalize=False):
        t = time.monotonic() - self.t0
        base = {"Present_Position": 2048, "Goal_Position": 2048, "Present_Current": 20}[name]
        amp = 0 if name == "Present_Current" else 200
        return {n: int(base + amp * math.sin(t + k)) for k, n in enumerate(JOINT_NAMES)}

    def disconnect(self, disable_torque=False):
        pass


def _sync_read(bus, reg):
    try:
        return bus.sync_read(reg, normalize=False, num_retry=2)
    except TypeError:                      # older LeRobot without num_retry
        return bus.sync_read(reg, normalize=False)


def read_once(bus):
    out = {}
    for reg in ("Present_Position", "Goal_Position", "Present_Current"):
        vals = _sync_read(bus, reg)
        out[reg] = ([float(vals[n]) for n in JOINT_NAMES], time.monotonic())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="servo bus serial port, e.g. /dev/ttyACM0")
    src.add_argument("--mock", action="store_true", help="synthetic values, no arm")
    ap.add_argument("--out", default="logs/arm_latest.json")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--calibration-version", default="so101-raw-ticks",
                    help="must match the recorder's --calibration-version")
    ap.add_argument("--seconds", type=float, default=float("inf"), help="stop after this long (default: Ctrl-C)")
    a = ap.parse_args(argv)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    bus = MockBus() if a.mock else open_bus(a.port)
    print(f"publishing {'MOCK ' if a.mock else ''}telemetry to {a.out} at {a.hz:g} Hz; Ctrl-C to stop")
    seq, t_end = 0, time.monotonic() + a.seconds
    failures, last_ok = 0, time.monotonic()
    try:
        while time.monotonic() < t_end:
            tick = time.monotonic()
            try:
                r = read_once(bus)
            except Exception as exc:       # a missed servo reply must not end the session
                failures += 1
                if time.monotonic() - last_ok > 2.0:
                    print(f"\nWARNING: no good reading for {time.monotonic() - last_ok:.1f} s "
                          f"({failures} failed reads so far; last error: {exc}). Check power and cables.")
                time.sleep(0.005)
                continue
            last_ok = time.monotonic()
            pos, t_pos = r["Present_Position"]
            goal, t_goal = r["Goal_Position"]
            cur, _ = r["Present_Current"]
            publish_telemetry(a.out, joint_state=pos, action=goal, state_timestamp=t_pos, action_timestamp=t_goal,
                              sequence_id=seq, joint_names=list(JOINT_NAMES), joint_units=UNITS,
                              calibration_version=a.calibration_version,
                              extra=dict(current_ma=[c * CURRENT_MA_PER_UNIT for c in cur], mock=bool(a.mock)))
            seq += 1
            if seq % max(1, int(a.hz * 10)) == 0:
                print(f"\r{seq} samples, {failures} failed reads", end="", flush=True)
            time.sleep(max(0.0, 1 / a.hz - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        bus.disconnect(disable_torque=False)     # leave torque exactly as it was
    print(f"\nstopped after {seq} samples, {failures} failed reads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
