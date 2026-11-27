"""Read-only telemetry bridge and independently drained camera streams.

The external teleop loop must publish measured joints and the command it actually
sent, using publish_telemetry after its existing read/send operations. This module
never connects to a servo bus or exposes a motor command method.
"""
import json
import sys
import threading
import time
from pathlib import Path

import cv2

from src.data.recording_v2 import atomic_json


def publish_telemetry(path, *, joint_state, action, state_timestamp,
                      action_timestamp, sequence_id, joint_names, joint_units,
                      calibration_version, extra=None):
    """Call on the recorder host. Timestamps must come from time.monotonic().

    State is encoder feedback, action is the command sent by teleop, not IK intent.
    Units must be explicit; no automatic radians/normalized/ticks conversion.
    extra: further measured fields (e.g. servo current), stored with each tick.
    """
    atomic_json(path, dict(joint_state=list(joint_state), action=list(action),
        state_timestamp=state_timestamp, action_timestamp=action_timestamp,
        sequence_id=sequence_id, joint_names=list(joint_names), joint_units=joint_units,
        calibration_version=calibration_version, **(extra or {})))


class TelemetryFile:
    def __init__(self, path, joint_names, joint_units, calibration_version):
        self.path = Path(path)
        self.expected = dict(joint_names=list(joint_names), joint_units=joint_units,
                             calibration_version=calibration_version)

    def read(self):
        value = json.loads(self.path.read_text(encoding="utf-8"))
        for key, expected in self.expected.items():
            if value.get(key) != expected:
                raise ValueError(f"telemetry {key} mismatch")
        return value


class TelemetryStale(RuntimeError):
    pass


def wait_for_live(source, timeout=3.0, max_age=0.2, clock=time.monotonic, sleep=time.sleep):
    """Block until the telemetry file is being updated (sequence advances, fresh timestamp).

    A file that exists but is frozen (teleop/bridge exited) would otherwise be recorded as a
    whole episode of one repeated pose."""
    first = source.read()["sequence_id"]
    deadline = clock() + timeout
    while clock() < deadline:
        v = source.read()
        if v["sequence_id"] != first and 0 <= clock() - v["state_timestamp"] <= max_age:
            return v
        sleep(0.02)
    raise TelemetryStale(f"telemetry {source.path} is not updating (sequence stuck at {first}). "
                         "Start gamepad_teleop.py (or so101_telemetry.py) first, then the recorder.")


class TelemetryWatch:
    """Track sequence progress during a recording: warn after `warn_s`, stop after `stop_s`."""

    def __init__(self, warn_s=1.0, stop_s=5.0, clock=time.monotonic):
        self.warn_s, self.stop_s, self.clock = warn_s, stop_s, clock
        self.last_seq, self.last_change, self.warned = None, clock(), False

    def check(self, telemetry):
        now = self.clock()
        if telemetry["sequence_id"] != self.last_seq:
            self.last_seq, self.last_change, self.warned = telemetry["sequence_id"], now, False
            return None
        frozen = now - self.last_change
        if frozen > self.stop_s:
            raise TelemetryStale(f"telemetry stopped updating for {frozen:.1f} s; recording ended")
        if frozen > self.warn_s and not self.warned:
            self.warned = True
            return f"WARNING: telemetry frozen for {frozen:.1f} s (teleop stopped or bus errors?)"
        return None


# Stable names for Nirav's two cameras. /dev/videoN numbers swap on replug; these by-id
# links follow the physical camera (2026-11-27: a swap recorded a whole demo with a black overhead).
CAMERA_IDS = {
    "overhead": "/dev/v4l/by-id/usb-Jieli_Technology_USB_PHY_2.0-video-index0",
    "wrist": "/dev/v4l/by-id/usb-Generic_USB_Camera_200901010001-video-index0",
}
BLACK_MEAN = 5.0     # mean pixel value below this = no picture (lens covered, exposure far too low)


def resolve_camera(value, role):
    """'2' -> 2 (index); a path (or None -> the role's by-id link) -> its real /dev/videoN."""
    if value is None:
        value = CAMERA_IDS[role]
    if isinstance(value, int) or str(value).isdigit():
        return int(value)
    path = Path(value)
    if not path.exists():
        raise RuntimeError(f"{role} camera {value} not found (unplugged?); see `ls -l /dev/v4l/by-id/`")
    return str(path.resolve())


def black_cameras(snapshot):
    """Names of cameras whose current frame has no picture."""
    return [name for name, sample in snapshot.items() if float(sample[0].mean()) < BLACK_MEAN]


class CameraStreams:
    def __init__(self, devices, resolution=(1280, 720), fps=30):
        if set(devices) != {"overhead", "wrist"} or devices["overhead"] == devices["wrist"]:
            raise ValueError("distinct overhead and wrist devices required")
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.samples, self.caps, self.threads = {}, [], []
        self.formats = {}
        try:
            for name, device in devices.items():
                cap = (cv2.VideoCapture(device, cv2.CAP_V4L2) if sys.platform.startswith("linux")
                       else cv2.VideoCapture(device))
                self.caps.append(cap)
                if not cap.isOpened():
                    raise RuntimeError(f"cannot open {name} device {device}")
                # MJPG first: many webcams give 1280x720 at 30 fps only as MJPG (raw YUYV: 5 fps).
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
                cap.set(cv2.CAP_PROP_FPS, fps)
                got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                self.formats[name] = dict(size=list(got), fps=float(cap.get(cv2.CAP_PROP_FPS)),
                                          fourcc=int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("ascii", "replace"))
                if got != tuple(resolution):
                    raise RuntimeError(f"{name} camera gave {got[0]}x{got[1]}, not {resolution[0]}x{resolution[1]}; "
                                       "calibration would not match the frames")
                thread = threading.Thread(target=self._run, args=(name, cap), daemon=True)
                self.threads.append(thread)
                thread.start()
        except BaseException:
            self.close()
            raise

    def _run(self, name, cap):
        seq, failures = 0, 0
        while not self.stop.is_set():
            ok, frame = cap.read()
            receipt = time.monotonic()  # host receipt, never sensor exposure
            seq += 1
            if not ok:
                failures += 1
                self.stop.wait(.01)
            with self.lock:
                if ok:
                    self.samples[name] = (frame, receipt, seq, failures)
                elif name in self.samples:
                    old = self.samples[name]
                    self.samples[name] = (*old[:3], failures)

    def snapshot(self):
        with self.lock:
            return dict(self.samples)

    def close(self):
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=.5)
        for cap in self.caps:
            cap.release()
