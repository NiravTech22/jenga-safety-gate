"""
Step 7: the Invariant safety gate.

Input: proposed action + anomaly score. Output: execute or block, logged
through Invariant. This is the one piece of the safety stack that can and
should be fully built and tested before any hardware is involved — it's
pure decision logic over a score.

Implemented and tested (tests/test_gate.py). No stubs in this file.

IMPORTANT (per PLAN.md Step 7): the threshold must be locked before Step 8
begins and tuned only against Step 6's validation set. This module enforces
that at the API level — you cannot construct a SafetyGate with an unlocked
threshold and also run it in "recording treatment-group results" mode
without explicitly acknowledging that. See `SafetyGate.lock_threshold`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from src.safety.anomaly_detector import AnomalyScore, AnomalyScorer
from src.safety.invariant_schema import (
    InvariantEvent,
    InvariantLogger,
    NodeStatus,
    StageType,
    new_node_id,
)


class GateDecision:
    EXECUTE = "execute"
    BLOCK = "block"


@dataclass
class GateResult:
    decision: str                  # GateDecision.EXECUTE | GateDecision.BLOCK
    anomaly_score: AnomalyScore
    threshold: float
    latency_ms: float
    node_id: str


class ThresholdNotLockedError(RuntimeError):
    pass


class SafetyGate:
    """Fail-closed by default: any exception in scoring is treated as an
    anomaly-detector failure and blocks the action, per config.yaml's
    safety_gate.fallback_on_gate_failure. A gate that fails open defeats
    its own purpose.
    """

    def __init__(
        self,
        scorer: AnomalyScorer,
        invariant_log_path,
        threshold: Optional[float] = None,
        fail_closed: bool = True,
    ):
        self.scorer = scorer
        self.invariant = InvariantLogger(invariant_log_path)
        self._threshold = threshold
        self._threshold_locked = False
        self.fail_closed = fail_closed

    def lock_threshold(self, threshold: float) -> None:
        """Call exactly once, after Step 6 validation, before Step 8. The
        plan is explicit that changing the threshold mid-comparison
        invalidates the experiment — locking makes that a code-level
        guarantee, not just a documentation note.
        """
        if self._threshold_locked:
            raise RuntimeError(
                "Threshold is already locked. If you genuinely need to "
                "change it, you are starting a new experiment — instantiate "
                "a new SafetyGate rather than mutating this one, so Step 9's "
                "comparison table can't accidentally mix two thresholds."
            )
        self._threshold = threshold
        self._threshold_locked = True

    def evaluate(self, frame, action_ref: str) -> GateResult:
        if self._threshold is None:
            raise ThresholdNotLockedError(
                "No threshold set. Call lock_threshold() with a value tuned "
                "against Step 6's 75-frame validation set before evaluating "
                "any real actions."
            )

        node_id = new_node_id(StageType.GATE_DECISION)
        start = time.time()

        try:
            score = self.scorer.score(frame)
            decision = GateDecision.BLOCK if score.value >= self._threshold else GateDecision.EXECUTE
            status = NodeStatus.BLOCKED if decision == GateDecision.BLOCK else NodeStatus.SUCCESS
            failure_reason = None
        except Exception as e:  # noqa: BLE001 — deliberately broad: any scorer failure must fail closed
            if not self.fail_closed:
                raise
            score = AnomalyScore(value=float("inf"), latency_ms=0.0, source="gate_failure_fallback")
            decision = GateDecision.BLOCK
            status = NodeStatus.BLOCKED
            failure_reason = f"anomaly scorer raised {type(e).__name__}: {e}"

        latency_ms = (time.time() - start) * 1000

        self.invariant.log(InvariantEvent(
            node_id=node_id,
            stage=StageType.GATE_DECISION,
            status=status,
            timestamp=time.time(),
            latency_ms=latency_ms,
            input_ref=action_ref,
            output_ref=decision,
            failure_reason=failure_reason,
        ))

        return GateResult(
            decision=decision,
            anomaly_score=score,
            threshold=self._threshold,
            latency_ms=latency_ms,
            node_id=node_id,
        )

    def is_locked(self) -> bool:
        return self._threshold_locked
