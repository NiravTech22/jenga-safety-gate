"""Offline replay; no arm or teleop imports."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.data.recording_v2 import load_episode, inspect_episode


def render_row(data, directory, index):
    row = data["timesteps"][index]
    modern = data["schema_version"] == 2
    names = ("overhead", "wrist") if modern else ("overhead", "side")
    panels = []
    for name in names:
        sample = row["cameras"].get(name) if modern else {"path": row.get(name+"_frame_path")}
        frame = None
        if sample and sample.get("path"):
            target = (directory / sample["path"]).resolve()
            if target.is_relative_to(directory.resolve()):
                frame = cv2.imread(str(target))
        if frame is None:
            frame = np.zeros((360, 480, 3), np.uint8)
            cv2.putText(frame, "MISSING FRAME", (40,180), 0, .8, (0,0,255), 2)
        frame = cv2.resize(frame, (480,360))
        cv2.putText(frame, name + (" (moving)" if name == "wrist" else ""), (10,25), 0, .7, (255,255,255), 2)
        if modern and sample:
            text = f"seq={sample['sequence_id']} age={(row['receipt_timestamp']-sample['timestamp'])*1000:.1f}ms"
            cv2.putText(frame, text, (10,50), 0, .5, (255,255,255), 1)
        panels.append(frame)
    canvas = np.zeros((530,960,3), np.uint8)
    canvas[:360] = np.hstack(panels)
    telemetry = row["telemetry"] if modern else row
    events = [e for e in data.get("events", []) if e["t"] <= row["t"]]
    lines = [f"t={row['t']:.3f}s frame={index} backend={data['metadata'].get('backend', 'legacy unknown')}",
        "state: " + str(np.round(telemetry['joint_state'],4).tolist()),
        "action: " + str(np.round(telemetry['action'],4).tolist()),
        "events: " + "; ".join(f"{e['t']:.2f}s {e['kind']}" for e in events[-3:]),
        "Units: " + data['metadata'].get('joint_units', 'legacy unspecified') + " | space pause, q quit"]
    for j, text in enumerate(lines):
        cv2.putText(canvas, text, (10,390+j*28), 0, .55, (255,255,255), 1)
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episode", type=Path)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--video", type=Path)
    p.add_argument("--fps", type=float, default=15, help="export video frame rate")
    args = p.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        p.error("fps must be positive and finite")
    data = load_episode(args.episode)
    report = inspect_episode(args.episode)
    print(json.dumps(report, indent=2))
    writer = None
    if args.video:
        if args.video.exists():
            p.error("refusing to overwrite existing video")
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (960,530))
        if not writer.isOpened():
            raise RuntimeError("video writer failed")
    try:
        for i, row in enumerate(data["timesteps"]):
            frame = render_row(data, args.episode.parent, i)
            if writer:
                writer.write(frame)
            if not args.headless:
                cv2.imshow("Episode replay", frame)
                rows = data["timesteps"]
                delay = max(1, min(1000, int(1000*(rows[i+1]["t"]-row["t"])))) if i+1 < len(rows) else 30
                key = cv2.waitKey(delay) & 255
                if key == ord(' '):
                    key = cv2.waitKey(0) & 255
                if key == ord('q'):
                    break
    finally:
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
