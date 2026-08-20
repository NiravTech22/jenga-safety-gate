"""
Invariant event schema.

This is the general execution-observability schema from the broader project
(not Jenga-specific), applied here to the Jenga safety-gate experiment. Every
future feature — sim-to-real divergence, spatially-embedded DAGs, the
proprioceptive veto, fleet aggregation — is an extension of this schema, so
it's deliberately kept minimal and stable rather than growing organically.

Implemented, not stubbed.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Optional


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    BLOCKED = "blocked"          # gated: proposed action was vetoed
    FLAGGED = "flagged"          # anomaly noted but not blocking (advisory)


class StageType(str, Enum):
    TASK_CONFIG_PARSE = "task_config_parse"
    PERCEPTION = "perception"           # tower-state extraction
    POLICY_INFERENCE = "policy_inference"
    ANOMALY_SCORE = "anomaly_score"
    GATE_DECISION = "gate_decision"
    ACTUATION = "actuation"


@dataclass
class InvariantEvent:
    node_id: str
    stage: StageType
    status: NodeStatus
    timestamp: float
    latency_ms: float
    input_ref: Optional[str] = None       # e.g. episode_id:timestep_index
    output_ref: Optional[str] = None
    failure_reason: Optional[str] = None
    # Extension point for future work (kept optional so old logs stay valid):
    robot_id: str = "so101-default"
    spatial_pose: Optional[list[float]] = None   # [x,y,z] end-effector, when available

    @staticmethod
    def start(node_id: str, stage: StageType, input_ref: Optional[str] = None) -> "InvariantEvent":
        return InvariantEvent(
            node_id=node_id,
            stage=stage,
            status=NodeStatus.RUNNING,
            timestamp=time.time(),
            latency_ms=0.0,
            input_ref=input_ref,
        )


class InvariantLogger:
    """Minimal append-only event log. SQLite/JSONL is intentionally the
    whole implementation for now, per the plan's own advice: don't build
    infrastructure you don't need yet.
    """

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: InvariantEvent) -> None:
        with open(self.log_path, "a") as f:
            f.write(json.dumps(asdict(event), default=str) + "\n")

    def read_all(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        with open(self.log_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def failures(self) -> list[dict]:
        return [e for e in self.read_all() if e["status"] == NodeStatus.FAILURE.value]

    def blocks(self) -> list[dict]:
        return [e for e in self.read_all() if e["status"] == NodeStatus.BLOCKED.value]


def new_node_id(stage: StageType) -> str:
    return f"{stage.value}-{uuid.uuid4().hex[:8]}"
