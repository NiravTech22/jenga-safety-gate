#!/usr/bin/env python3
"""Characterise the teleoperated arm before trusting it to record data.

Not a plan step. It exists because docs/PROGRESS.md carries three unresolved
teleop issues, and PLAN.md Step 4 warns that recording before fixing them
bakes the same bias into all twenty-two episodes.

  run       drive the arm through probe motions and report where the error is
  z-floor   measure the table height so the disabled safety floor can be
            restored to a real value

Examples
  python scripts/diagnose_teleop.py run --sim
  python scripts/diagnose_teleop.py run --sim --demo-faults
  python scripts/diagnose_teleop.py run --hardware
  python scripts/diagnose_teleop.py z-floor --hardware
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

from src.hardware.arm import ArmDefects, N_JOINTS, SimulatedArm, VRTeleopArm
from src.hardware.diagnostics import run_diagnostics
from src.hardware.kinematics import JOINT_NAMES, ArmChain

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_arm(args):
    if args.hardware:
        cfg = load_config()
        return VRTeleopArm(cfg["vr_control"]["server_url"])
    if args.demo_faults:
        # The faults recorded in docs/PROGRESS.md, so the output shows what
        # those issues would look like if they were measured.
        return SimulatedArm(ArmDefects(
            offset_rad=np.array([0.0, np.radians(1.8), 0.0, np.radians(-3.5), 0.0, 0.0]),
            scale=np.array([1.0, 1.0, 1.0, 1.14, 1.0, 0.82]),
            backlash_rad=np.array([0.0, 0.0, np.radians(2.2), 0.0, 0.0, 0.0]),
            latency_s=0.09,
            noise_rad=np.radians(0.02),
        ))
    return SimulatedArm()


def cmd_run(args):
    arm = build_arm(args)
    label = "hardware" if args.hardware else ("simulated arm with injected faults"
                                              if args.demo_faults else "clean simulated arm")
    print(f"Characterising: {label}\n")
    report = run_diagnostics(arm, settle_s=args.settle)
    print(report.render())

    out = Path(args.out)
    report.save(out)
    print(f"\nWritten to {out}")

    if not args.hardware:
        print("\nThis is a simulation. It proves the probes work; it says nothing\n"
              "about your arm. Run with --hardware once VRTeleopArm is wired up.")
    print("""
Reading the table
  offset       a constant bias. Correctable in the VR layer, or by fixing the
               joint's home value in config.yaml arm.joints.
  scale        travel error growing with distance from zero. Not correctable
               by an offset. The joint's min/max in the calibration table does
               not match its real mechanical range.
  lost motion  the joint settles short and stays short, from backlash or servo
               deadband. Not fixable in the VR layer at all.
  repeat       spread when returning to the same target from random starts.
               Includes hysteresis on purpose, because that is what the
               operator feels.
""")
    return 0


def cmd_z_floor(args):
    """Restore the safety floor to a measured value.

    config.yaml records that z_floor was once set to -0.05 to work around a
    miscalibrated ground plane, which disabled the constraint rather than
    correcting it. The fix is to measure the table.
    """
    arm = build_arm(args)
    chain = ArmChain()

    print("Jog the gripper down until it just touches the table, then press Enter.")
    print("Do this by hand through the teleop, slowly. Ctrl-C to abort.\n")
    try:
        input("  press Enter at contact > ")
    except (EOFError, KeyboardInterrupt):
        print("\naborted")
        return 1

    joints = np.asarray(arm.read_joints(), float)
    pos = chain.tool_position(joints)
    clearance = args.clearance_mm / 1000.0
    z_floor = float(pos[2] - clearance)

    print(f"\n  joints (deg): {np.round(np.degrees(joints), 2)}")
    print(f"  tool position (m): {np.round(pos, 4)}")
    print(f"  contact height Z = {pos[2]:.4f} m")
    print(f"  suggested z_floor = {z_floor:.4f} m "
          f"({args.clearance_mm:.0f} mm below contact)\n")
    print("Set this in config/config.yaml:")
    print(f"  vr_control:\n    z_floor_m: {z_floor:.4f}\n    z_floor_measured: true")
    print("\nRepeat two or three times at different XY positions. If the contact\n"
          "height disagrees by more than a few mm across them, the table is not\n"
          "level in the base frame and a single scalar floor will be wrong\n"
          "somewhere in the workspace.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--hardware", action="store_true", help="Use the real arm.")
        p.add_argument("--sim", action="store_true", help="Use the simulated arm (default).")
        p.add_argument("--demo-faults", action="store_true",
                       help="Simulated arm carrying the issues logged in PROGRESS.md.")

    r = sub.add_parser("run", help="Probe the arm and report where the error is.")
    common(r)
    r.add_argument("--settle", type=float, default=1.0, help="Settle time per probe, seconds.")
    r.add_argument("--out", default="./data/diagnostics/teleop_report.json")
    r.set_defaults(func=cmd_run)

    z = sub.add_parser("z-floor", help="Measure table height for the safety floor.")
    common(z)
    z.add_argument("--clearance-mm", type=float, default=3.0)
    z.set_defaults(func=cmd_z_floor)

    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
