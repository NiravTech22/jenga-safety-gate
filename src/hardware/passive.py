"""Read-only telemetry bridge and independently drained camera streams.

The external teleop loop must publish measured joints and the command it actually
sent, using publish_telemetry after its existing read/send operations. This module
never connects to a servo bus or exposes a motor command method.
"""
import json
import threading
import time
from pathlib import Path

import cv2

from src.data.recording_v2 import atomic_json


def publish_telemetry(path, *, joint_state, action, state_timestamp,
                      action_timestamp, sequence_id, joint_names, joint_units,
                      calibration_version):
    """Call on the recorder host. Timestamps must come from time.monotonic().

    State is encoder feedback, action is the command sent by teleop, not IK intent.
    Units must be explicit; no automatic radians/normalized/ticks conversion.
    """
    atomic_json(path, dict(joint_state=list(joint_state), action=list(action),
        state_timestamp=state_timestamp, action_timestamp=action_timestamp,
        sequence_id=sequence_id, joint_names=list(joint_names), joint_units=joint_units,
        calibration_version=calibration_version))


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


class CameraStreams:
    def __init__(self, devices, resolution=(1280, 720), fps=30):
        if set(devices) != {"overhead", "wrist"} or devices["overhead"] == devices["wrist"]:
            raise ValueError("distinct overhead and wrist devices required")
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.samples, self.caps, self.threads = {}, [], []
        try:
            for name, device in devices.items():
                cap = cv2.VideoCapture(device)
                self.caps.append(cap)
                if not cap.isOpened():
                    raise RuntimeError(f"cannot open {name} device {device}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
                cap.set(cv2.CAP_PROP_FPS, fps)
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
