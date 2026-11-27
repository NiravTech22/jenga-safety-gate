"""Record passive hardware observations or explicitly synthetic smoke data."""
import argparse
import time

import cv2
import numpy as np

from src.data.recording_v2 import Recording, inspect_episode
from src.hardware.passive import (CameraStreams, TelemetryFile, TelemetryWatch, black_cameras,
                                  resolve_camera, wait_for_live)
from src.hardware.kinematics import JOINT_NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--hardware", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    p.add_argument("--root", default="data/episodes")
    p.add_argument("--session-id", required=True)
    p.add_argument("--tower-id", required=True)
    p.add_argument("--calibration-version", required=True)
    p.add_argument("--joint-units", required=True)
    p.add_argument("--operator", default="unknown")
    p.add_argument("--scenario", default="stable_easy")
    p.add_argument("--telemetry")
    p.add_argument("--overhead", help="index, /dev/videoN or /dev/v4l/by-id path (default: the overhead's by-id link)")
    p.add_argument("--wrist", help="index, /dev/videoN or /dev/v4l/by-id path (default: the wrist's by-id link)")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--fps", type=float, default=15)
    p.add_argument("--max-age", type=float, default=.2)
    p.add_argument("--max-skew", type=float, default=.05)
    p.add_argument("--preview", action="store_true", help="r=removal c=collapse q=finish")
    args = p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0 or not np.isfinite(args.fps) or args.fps <= 0:
        p.error("seconds and fps must be positive finite values")
    if args.hardware and not args.telemetry:
        p.error("hardware requires --telemetry; dummy state is forbidden")
    source = None
    if args.hardware:
        source = TelemetryFile(args.telemetry, JOINT_NAMES, args.joint_units, args.calibration_version)
        source.read()  # fail before opening cameras if bridge is unavailable
        wait_for_live(source, max_age=args.max_age)  # ...or present but frozen (teleop not running)
        print("telemetry live")
    rec = Recording(args.root, backend="hardware" if args.hardware else "synthetic",
        session_id=args.session_id, tower_id=args.tower_id, calibration_version=args.calibration_version,
        joint_names=list(JOINT_NAMES), joint_units=args.joint_units, operator=args.operator,
        max_age_s=args.max_age, max_skew_s=args.max_skew)
    rec.data["metadata"]["scenario_category"] = args.scenario
    rec.data["metadata"]["devices"] = {"overhead": args.overhead, "wrist": args.wrist}
    cameras, interrupted, reason = None, False, ""
    watch = TelemetryWatch()
    try:
        if args.hardware:
            devices = {role: resolve_camera(getattr(args, role), role) for role in ("overhead", "wrist")}
            rec.data["metadata"]["devices"] = devices
            print("cameras:", devices)
            cameras = CameraStreams(devices)
            rec.data["metadata"]["camera_formats"] = cameras.formats
            print("cameras:", cameras.formats)
            deadline = time.monotonic() + 5
            while len(cameras.snapshot()) < 2:
                if time.monotonic() >= deadline:
                    raise RuntimeError("both cameras must deliver initial frames within 5 seconds")
                time.sleep(.01)
            time.sleep(.5)                     # let auto-exposure settle before judging the picture
            dark = black_cameras(cameras.snapshot())
            if dark:
                raise RuntimeError(f"{', '.join(dark)} camera shows a black picture: wrong device, lens covered, "
                                   "or exposure not locked (run scripts/tools/lock_cameras.sh)")
        for i in range(max(1, int(args.seconds * args.fps))):
            tick = time.monotonic()
            if args.hardware:
                frames, telemetry = cameras.snapshot(), source.read()
                warning = watch.check(telemetry)
                if warning:
                    print("\n" + warning)
            else:
                frames = {}
                for j, name in enumerate(("overhead", "wrist")):
                    frame = np.full((240, 320, 3), (30, 60+j*60, 50), np.uint8)
                    cv2.putText(frame, f"SYNTHETIC {name} {i}", (10, 120), 0, .55, (255,255,255), 1)
                    frames[name] = (frame, tick, i, 0)
                telemetry = dict(joint_state=[float(np.sin(i/10))]*6,
                    action=[float(np.sin((i+1)/10))]*6, state_timestamp=tick,
                    action_timestamp=tick, sequence_id=i)
                if i == 3:
                    rec.event("removal", "synthetic marker")
                if i == 7:
                    rec.event("collapse_onset", "synthetic marker, not physical collapse")
            rec.append(frames, telemetry)
            if args.preview:
                from src.data.replay import render_row
                cv2.imshow("Passive recording: r removal / c collapse / q finish", render_row(rec.data, rec.directory, i))
                key = cv2.waitKey(1) & 255
                if key in (ord('r'), ord('c')):
                    rec.event("removal" if key == ord('r') else "collapse_onset", "operator annotation")
                if key == ord('q'):
                    break
            time.sleep(max(0, 1/args.fps-(time.monotonic()-tick)))
    except KeyboardInterrupt:
        interrupted, reason = True, "operator Ctrl-C"
    except Exception as exc:
        interrupted, reason = True, str(exc)
        raise
    finally:
        if cameras:
            cameras.close()
        cv2.destroyAllWindows()
        path = rec.close(interrupted, reason)
        print(f"Episode: {path}")
        report = inspect_episode(path)
        print(report)
    return 2 if interrupted or not report["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
