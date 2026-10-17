"""Passive recording contract. All times are host monotonic seconds, not exposure."""
from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

CAMERAS = {"overhead": {"mount": "fixed"}, "wrist": {"mount": "arm", "extrinsics": "joint-dependent"}}


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


class Recording:
    def __init__(self, root, *, backend, session_id, tower_id, calibration_version,
                 joint_names, joint_units, operator="unknown", max_age_s=.2, max_skew_s=.05):
        if backend not in ("hardware", "synthetic"):
            raise ValueError("backend must be hardware or synthetic")
        if not all((session_id, tower_id, calibration_version, joint_names, joint_units)):
            raise ValueError("session, tower, calibration, joint names and units required")
        if not (math.isfinite(max_age_s) and math.isfinite(max_skew_s) and max_age_s > 0 and max_skew_s > 0):
            raise ValueError("positive finite timing limits required")
        self.directory = Path(root) / str(uuid.uuid4())
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "frames").mkdir()
        self.start = time.monotonic()
        self.data = {"schema_version": 2, "metadata": {
            "episode_id": self.directory.name, "backend": backend, "session_id": session_id,
            "tower_id": tower_id, "calibration_version": calibration_version, "operator": operator,
            "joint_names": joint_names, "joint_units": joint_units,
            "cameras": CAMERAS, "timestamp_semantics": "host_monotonic_receipt_not_exposure",
            "created_at_unix": time.time(), "origin_monotonic_s": self.start,
            "max_age_s": max_age_s, "max_skew_s": max_skew_s,
            "status": "interrupted", "motor_commands_sent": 0}, "timesteps": [], "events": []}
        self.closed = False
        self.save()

    def save(self):
        atomic_json(self.directory / "episode.json", self.data)

    def append(self, frames, telemetry, now=None):
        if self.closed:
            raise RuntimeError("recording already closed")
        now = time.monotonic() if now is None else now
        n = len(self.data["metadata"]["joint_names"])
        for key in ("joint_state", "action"):
            values = telemetry[key]
            if len(values) != n or not all(math.isfinite(v) for v in values):
                raise ValueError(f"invalid {key}; real measured state and sent command required")
        for key in ("state_timestamp", "action_timestamp"):
            if not math.isfinite(telemetry[key]):
                raise ValueError(f"invalid {key}")
        if not isinstance(telemetry["sequence_id"], int) or telemetry["sequence_id"] < 0:
            raise ValueError("invalid telemetry sequence")
        row = {"t": now - self.start, "receipt_timestamp": now, "telemetry": telemetry, "cameras": {}}
        idx = len(self.data["timesteps"])
        for name in CAMERAS:
            sample = frames.get(name)
            if sample is None:
                row["cameras"][name] = None
                continue
            frame, stamp, seq, failed_reads = sample
            relative = f"frames/{idx:06d}_{name}.jpg"
            if not cv2.imwrite(str(self.directory / relative), frame):
                raise IOError(f"frame write failed: {relative}")
            row["cameras"][name] = {"path": relative, "timestamp": stamp,
                "sequence_id": seq, "failed_reads": failed_reads}
        self.data["timesteps"].append(row)
        self.save()  # durable per-tick manifest; unfinished runs remain interrupted

    def event(self, kind, details="", t=None):
        if kind not in ("removal", "collapse_onset", "note", "interruption"):
            raise ValueError("unknown event")
        self.data["events"].append({"kind": kind, "t": time.monotonic()-self.start if t is None else t, "details": details})
        self.save()

    def close(self, interrupted=False, reason=""):
        if interrupted:
            self.event("interruption", reason)
        self.data["metadata"]["status"] = "interrupted" if interrupted else "complete"
        self.data["metadata"]["num_timesteps"] = len(self.data["timesteps"])
        self.save()
        self.closed = True
        report = inspect_episode(self.directory / "episode.json")
        atomic_json(self.directory / "integrity.json", report)
        return self.directory / "episode.json"


def load_episode(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version", 1) == 1:
        # Never relabel a historical fixed side camera as a moving wrist camera.
        data["schema_version"] = 1
        data["metadata"]["cameras"] = {"overhead": {"mount": "unknown"}, "side": {"mount": "legacy_fixed"}}
    elif data["schema_version"] != 2:
        raise ValueError("unsupported episode version")
    return data


def inspect_episode(path):
    try:
        return _inspect_episode(path)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return {"ok": False, "issues": [f"malformed episode: {exc}"], "warnings": [], "dropped_frames": {}}


def _inspect_episode(path):
    path = Path(path)
    data = load_episode(path)
    issues, warnings, dropped = [], [], {}
    rows, meta = data["timesteps"], data["metadata"]
    if not rows:
        issues.append("empty episode")
    if meta.get("num_timesteps", len(rows)) != len(rows):
        issues.append("timestep count mismatch")
    if meta.get("status") == "interrupted":
        warnings.append("interrupted run")
    previous, prev_t = {}, -float("inf")
    if data["schema_version"] == 1:
        warnings.append("legacy side camera; per-camera timing and dropped frames unknown")
    for i, row in enumerate(rows):
        def problem(message):
            issues.append(f"{i}: {message}")
        if not math.isfinite(row["t"]) or row["t"] <= prev_t:
            problem("non-increasing/nonfinite timestep")
        prev_t = row["t"]
        if data["schema_version"] == 1:
            cameras = {name: {"path": row.get(name+"_frame_path")} for name in ("overhead", "side")}
        else:
            cameras = {name: row["cameras"].get(name) for name in CAMERAS}
            now = row["receipt_timestamp"]
            telemetry = row["telemetry"]
            if not math.isfinite(now):
                problem("nonfinite receipt timestamp")
            for key in ("joint_state", "action"):
                if len(telemetry[key]) != len(meta["joint_names"]) or not all(math.isfinite(v) for v in telemetry[key]):
                    problem(f"invalid {key}")
            for key in ("state_timestamp", "action_timestamp"):
                age = now - telemetry[key]
                if not math.isfinite(age) or not 0 <= age <= meta["max_age_s"]:
                    problem(f"stale/future {key}")
            sequence = telemetry["sequence_id"]
            if not isinstance(sequence, int) or sequence < 0:
                problem("invalid telemetry sequence")
            if sequence <= previous.get("telemetry", -1):
                problem("repeated/reversed telemetry sequence")
            previous["telemetry"] = sequence
            stamps = [v["timestamp"] for v in cameras.values() if v]
            if len(stamps) == 2 and abs(stamps[0]-stamps[1]) > meta["max_skew_s"]:
                problem("camera skew")
        for name, sample in cameras.items():
            if not sample or not sample.get("path"):
                problem(f"missing {name} frame")
                continue
            image_path = (path.parent / sample["path"]).resolve()
            if not image_path.is_relative_to(path.parent.resolve()) or cv2.imread(str(image_path)) is None:
                problem(f"missing/unreadable/unsafe {name} frame")
            if data["schema_version"] == 1:
                continue
            age = now - sample["timestamp"]
            if not math.isfinite(age) or not 0 <= age <= meta["max_age_s"]:
                problem(f"stale/future {name} frame")
            seq = sample["sequence_id"]
            if not isinstance(seq, int) or seq < 0 or not isinstance(sample["failed_reads"], int) or sample["failed_reads"] < 0:
                problem(f"invalid {name} sequence/drop count")
            prior = previous.get(name)
            dropped.setdefault(name, {"sequence_gaps": 0, "failed_reads": 0})
            if prior is not None:
                if seq <= prior:
                    problem(f"repeated/reversed {name} sequence")
                dropped[name]["sequence_gaps"] += max(0, seq-prior-1)
            previous[name] = seq
            dropped[name]["failed_reads"] = max(dropped[name]["failed_reads"], sample["failed_reads"])
    for event in data.get("events", []):
        if not math.isfinite(event["t"]) or event["t"] < 0:
            issues.append("invalid event time")
    return {"ok": not issues, "issues": issues, "warnings": warnings, "dropped_frames": dropped,
            "drop_semantics": "host sequence gaps and failed reads; sensor-internal drops unknown", "frames": len(rows)}
