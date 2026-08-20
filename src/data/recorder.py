"""
Step 3: VR teleop data recording pipeline.

Wraps an existing teleop loop and produces Episode objects (schema.py),
logging each stage through Invariant. This module does NOT talk to the
arm or the VR server directly — it expects the caller (your existing
SO-101-VR-Control loop) to push timesteps in as they happen. This keeps
the recorder testable without hardware: see tests/test_recorder.py.

Implemented: buffering, frame saving, schema validation on close, Invariant
logging of the recording stage.
Stubbed (needs real camera + VR loop wired in): `capture_frame_pair()` is a
placeholder — replace with real cv2.VideoCapture reads from the fixed
overhead/side rig once Step 1 is physically done.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Optional

from src.data.schema import Episode, EpisodeOutcome, ScenarioCategory, TimestepRecord
from src.safety.invariant_schema import (
    InvariantEvent,
    InvariantLogger,
    NodeStatus,
    StageType,
    new_node_id,
)


class FrameCapture:
    """Placeholder for the real fixed-camera rig (Step 1).

    Once cameras are physically mounted and config.yaml's cameras.*.fixed
    flags are true, replace capture_frame_pair with real cv2.VideoCapture
    reads. Kept as a separate class so the recorder never needs to change
    when that swap happens.
    """

    def __init__(self, overhead_device: Optional[int] = None, side_device: Optional[int] = None):
        self.overhead_device = overhead_device
        self.side_device = side_device
        self._overhead_cap = None
        self._side_cap = None
        if overhead_device is not None and side_device is not None:
            import cv2  # local import: keep this module importable without opencv installed

            self._overhead_cap = cv2.VideoCapture(overhead_device)
            self._side_cap = cv2.VideoCapture(side_device)

    def capture_frame_pair(self):
        """Returns (overhead_frame, side_frame) as numpy arrays, or raises
        RuntimeError if the rig isn't wired up yet — fail loud, not silent,
        since a silently-black frame is exactly the misalignment bug the
        plan warns about.
        """
        if self._overhead_cap is None or self._side_cap is None:
            raise RuntimeError(
                "FrameCapture has no real camera devices configured. "
                "Set cameras.overhead.device_index / cameras.side.device_index "
                "in config.yaml once Step 1 (physical camera mount) is done."
            )
        ok1, overhead = self._overhead_cap.read()
        ok2, side = self._side_cap.read()
        if not (ok1 and ok2):
            raise RuntimeError("Camera read failed — check device indices and connections.")
        return overhead, side


class EpisodeRecorder:
    def __init__(
        self,
        dataset_root: Path,
        scenario_category: ScenarioCategory,
        invariant_log_path: Path,
        operator: str = "unknown",
    ):
        self.dataset_root = Path(dataset_root)
        self.episode = Episode.new(scenario_category, operator=operator)
        self.episode_dir = self.dataset_root / self.episode.metadata.episode_id
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.episode_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        self._start_time = time.time()
        self.invariant = InvariantLogger(invariant_log_path)
        self._node_id = new_node_id(StageType.TASK_CONFIG_PARSE)
        self.invariant.log(
            InvariantEvent.start(self._node_id, StageType.TASK_CONFIG_PARSE,
                                  input_ref=self.episode.metadata.episode_id)
        )

    def record_timestep(
        self,
        overhead_frame,
        side_frame,
        joint_state: list[float],
        action: list[float],
        vr_controller_pose: Optional[list[float]] = None,
        vr_controller_linear_vel: Optional[list[float]] = None,
        vr_trigger_state: Optional[float] = None,
        servo_current_draw: Optional[list[float]] = None,
    ) -> None:
        """Call this once per control-loop tick from the live teleop loop."""
        import cv2  # local import to keep module importable without opencv for pure schema tests

        idx = len(self.episode.timesteps)
        overhead_path = self.frames_dir / f"{idx:06d}_overhead.jpg"
        side_path = self.frames_dir / f"{idx:06d}_side.jpg"
        cv2.imwrite(str(overhead_path), overhead_frame)
        cv2.imwrite(str(side_path), side_frame)

        ts = TimestepRecord(
            t=time.time() - self._start_time,
            overhead_frame_path=str(overhead_path.relative_to(self.episode_dir)),
            side_frame_path=str(side_path.relative_to(self.episode_dir)),
            joint_state=joint_state,
            action=action,
            vr_controller_pose=vr_controller_pose,
            vr_controller_linear_vel=vr_controller_linear_vel,
            vr_trigger_state=vr_trigger_state,
            servo_current_draw=servo_current_draw,
        )
        self.episode.add_timestep(ts)

    def close(self, outcome: EpisodeOutcome, notes: str = "") -> Path:
        """Finalize, validate, save, and log. Returns the saved episode path.

        If validation finds problems, the episode is still saved (so you
        don't lose the recording session) but the problems are written to
        notes and logged as an Invariant failure so they surface in review
        rather than silently corrupting Step 4's dataset.
        """
        self.episode.finalize(outcome, notes=notes)
        problems = self.episode.validate_time_alignment()

        latency_ms = (time.time() - self._start_time) * 1000
        if problems:
            combined_notes = (notes + " | VALIDATION PROBLEMS: " + "; ".join(problems)).strip(" |")
            self.episode.metadata.notes = combined_notes
            self.invariant.log(InvariantEvent(
                node_id=self._node_id,
                stage=StageType.TASK_CONFIG_PARSE,
                status=NodeStatus.FAILURE,
                timestamp=time.time(),
                latency_ms=latency_ms,
                input_ref=self.episode.metadata.episode_id,
                failure_reason="; ".join(problems),
            ))
        else:
            self.invariant.log(InvariantEvent(
                node_id=self._node_id,
                stage=StageType.TASK_CONFIG_PARSE,
                status=NodeStatus.SUCCESS,
                timestamp=time.time(),
                latency_ms=latency_ms,
                input_ref=self.episode.metadata.episode_id,
                output_ref=str(self.episode_dir),
            ))

        return self.episode.save(self.dataset_root)

    def discard(self) -> None:
        """Delete a bad recording session entirely (Step 4's 'discarded and
        re-recorded' path) rather than leaving a half-written episode dir
        around to confuse later dataset loading.
        """
        shutil.rmtree(self.episode_dir, ignore_errors=True)
