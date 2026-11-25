#!/usr/bin/env python3
"""Recalibrate ONLY the claw (gripper, motor 6) of the SO-101, keeping the other joints.

`lerobot-calibrate` always redoes every joint. After reassembling just the claw, this
redoes its homing offset and open/closed range and writes them into the same LeRobot
calibration file, so the arm joints' numbers (and episodes recorded with them) stay valid.

  python scripts/tools/calibrate_gripper.py --port /dev/ttyACM0 --robot-id jenga_follower

Steps: claw HALF open -> Enter; then fully closed and fully open -> Enter.
A backup of the old file is written next to it. Re-run `twin/arm_twin.py mirror`
`claw open` / `claw closed` afterwards.
"""
import argparse
import inspect
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

CAL_DIR = Path.home() / ".cache/huggingface/lerobot/calibration/robots"
IDS = dict(shoulder_pan=1, shoulder_lift=2, elbow_flex=3, wrist_flex=4, wrist_roll=5, gripper=6)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", required=True)
    ap.add_argument("--robot-id", default="jenga_follower")
    a = ap.parse_args(argv)
    hits = list(CAL_DIR.glob(f"*/{a.robot_id}.json"))
    if not hits:
        sys.exit(f"no calibration '{a.robot_id}' under {CAL_DIR}: run lerobot-calibrate once first")
    path = hits[0]
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    try:
        from lerobot.motors import SerialMotorsBus as Bus
    except ImportError:
        from lerobot.motors.feetech import FeetechMotorsBus as Bus
    if "motors" not in inspect.signature(Bus.record_calibration).parameters:
        sys.exit("this LeRobot version cannot calibrate a single motor; upgrade LeRobot (pip install -U 'lerobot[feetech]')")
    cal = {n: MotorCalibration(**v) for n, v in json.loads(path.read_text()).items()}
    motors = {n: Motor(i, "sts3215", MotorNormMode.RANGE_0_100 if n == "gripper" else MotorNormMode.DEGREES)
              for n, i in IDS.items()}
    bus = Bus(port=a.port, motors=motors, calibration=cal)
    bus.connect()
    try:
        old = cal["gripper"]
        new = bus.record_calibration("position", full_turn_motors=[], motors=["gripper"])["gripper"]
        if new.range_max - new.range_min < 200:
            sys.exit(f"claw range only {new.range_max - new.range_min} ticks: did it move fully open and closed? nothing saved")
        cal["gripper"] = new
        bus.write_calibration(cal)
        backup = path.with_suffix(f".backup-{time.strftime('%Y%m%d-%H%M%S')}.json")
        shutil.copy(path, backup)
        path.write_text(json.dumps({n: asdict(c) for n, c in cal.items()}, indent=4) + "\n")
        print(f"claw: range {old.range_min}-{old.range_max} -> {new.range_min}-{new.range_max} "
              f"(homing offset {old.homing_offset} -> {new.homing_offset})")
        print(f"saved {path} (old file: {backup.name})")
    finally:
        bus.disconnect(disable_torque=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
