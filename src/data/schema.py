"""
Episode schema for jenga-safety-gate.

This is the contract between Step 3 (recording) and every step after it.
Get this stable before writing more code — the plan explicitly calls this
out as the thing everything else depends on.

Implemented, not stubbed: this module has no hardware or ML dependencies and
is fully testable as-is (see tests/test_schema.py).
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

# V1 remains unchanged for existing consumers and historical side-camera data.
# V2 is deliberately explicit; it cannot be loaded into a side-camera tuple.
def load_recording(path):
    from src.data.recording_v2 import load_episode
    return load_episode(path)


class ScenarioCategory(str, Enum):
    STABLE_EASY = "stable_easy"
    STABLE_TIGHT = "stable_tight"
    LEANING = "leaning"
    NEAR_COLLAPSE = "near_collapse"
    OFF_CENTER = "off_center"
    RECOVERY = "recovery"


class EpisodeOutcome(str, Enum):
    SUCCESS = "success"
    TOPPLE = "topple"
    ABANDONED = "abandoned"
    # Recorded even for ungated/unlabeled trial runs before Step 5 exists.
    UNLABELED = "unlabeled"


@dataclass
class TimestepRecord:
    """One synchronized (image, state, action) tuple.

    Frame paths are stored, not raw arrays, so an episode directory is
    self-contained and inspectable without loading a training framework.
    """
    t: float                          # seconds since episode start
    overhead_frame_path: str
    side_frame_path: str
    joint_state: list[float]          # radians or normalized units, document which in metadata
    action: list[float]               # same units as joint_state
    # Optional signals worth logging from day one even before anything
    # consumes them — cheap now, expensive to reconstruct later.
    vr_controller_pose: Optional[list[float]] = None      # [x,y,z,qx,qy,qz,qw]
    vr_controller_linear_vel: Optional[list[float]] = None
    vr_trigger_state: Optional[float] = None
    servo_current_draw: Optional[list[float]] = None       # per-joint, if available


@dataclass
class EpisodeMetadata:
    episode_id: str
    scenario_category: ScenarioCategory
    outcome: EpisodeOutcome
    created_at_unix: float
    operator: str = "unknown"
    notes: str = ""
    num_timesteps: int = 0
    duration_s: float = 0.0
    # Filled in only for gated runs (Step 8); left null for Step 4/5 data.
    gate_present: bool = False
    gate_block_count: Optional[int] = None


@dataclass
class Episode:
    metadata: EpisodeMetadata
    timesteps: list[TimestepRecord] = field(default_factory=list)

    @staticmethod
    def new(scenario_category: ScenarioCategory, operator: str = "unknown") -> "Episode":
        meta = EpisodeMetadata(
            episode_id=str(uuid.uuid4()),
            scenario_category=scenario_category,
            outcome=EpisodeOutcome.UNLABELED,
            created_at_unix=time.time(),
            operator=operator,
        )
        return Episode(metadata=meta)

    def add_timestep(self, ts: TimestepRecord) -> None:
        self.timesteps.append(ts)
        self.metadata.num_timesteps = len(self.timesteps)
        if self.timesteps:
            self.metadata.duration_s = self.timesteps[-1].t - self.timesteps[0].t

    def finalize(self, outcome: EpisodeOutcome, notes: str = "") -> None:
        self.metadata.outcome = outcome
        self.metadata.notes = notes

    def to_dict(self) -> dict:
        return {
            "metadata": asdict(self.metadata),
            "timesteps": [asdict(t) for t in self.timesteps],
        }

    def save(self, root: Path) -> Path:
        """Save as <root>/<episode_id>/episode.json.

        Frame images referenced by path are NOT copied here — the recorder
        is responsible for writing them to the same episode directory before
        calling save(), so paths in timesteps should already be relative to
        this directory.
        """
        episode_dir = Path(root) / self.metadata.episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        out_path = episode_dir / "episode.json"
        with open(out_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out_path

    @staticmethod
    def load(episode_json_path: Path) -> "Episode":
        with open(episode_json_path) as f:
            raw = json.load(f)
        meta = EpisodeMetadata(**raw["metadata"])
        episode = Episode(metadata=meta)
        episode.timesteps = [TimestepRecord(**t) for t in raw["timesteps"]]
        return episode

    def validate_time_alignment(self, max_gap_s: float = 0.2) -> list[str]:
        """Cheap sanity check for the failure mode the plan calls out
        explicitly: silently misaligned (image, state, action) tuples.

        Returns a list of human-readable problems found (empty = clean).
        """
        problems = []
        if not self.timesteps:
            problems.append("episode has zero timesteps")
            return problems

        prev_t = None
        for i, ts in enumerate(self.timesteps):
            if prev_t is not None and ts.t <= prev_t:
                problems.append(f"timestep {i}: non-increasing timestamp ({ts.t} <= {prev_t})")
            if prev_t is not None and (ts.t - prev_t) > max_gap_s:
                problems.append(f"timestep {i}: gap of {ts.t - prev_t:.3f}s exceeds {max_gap_s}s")
            if len(ts.joint_state) != len(ts.action):
                problems.append(
                    f"timestep {i}: joint_state len {len(ts.joint_state)} != action len {len(ts.action)}"
                )
            prev_t = ts.t
        return problems
