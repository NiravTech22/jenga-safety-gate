import tempfile
from pathlib import Path

import pytest

from src.safety.anomaly_detector import AnomalyScore
from src.safety.gate import GateDecision, SafetyGate, ThresholdNotLockedError


class FakeScorer:
    """Deterministic fake for testing gate logic in isolation, per the
    module's own design intent (AnomalyScorer is a Protocol precisely so
    this works)."""

    def __init__(self, fixed_value: float):
        self.fixed_value = fixed_value

    def score(self, frame):
        return AnomalyScore(value=self.fixed_value, latency_ms=1.0, source="fake_for_testing")


class RaisingScorer:
    def score(self, frame):
        raise RuntimeError("simulated scorer crash")


def make_gate(scorer, tmp_dir):
    return SafetyGate(scorer=scorer, invariant_log_path=Path(tmp_dir) / "invariant.jsonl")


def test_evaluate_without_locked_threshold_raises():
    with tempfile.TemporaryDirectory() as tmp:
        gate = make_gate(FakeScorer(0.9), tmp)
        with pytest.raises(ThresholdNotLockedError):
            gate.evaluate(frame=None, action_ref="ep-1:0")


def test_low_score_executes():
    with tempfile.TemporaryDirectory() as tmp:
        gate = make_gate(FakeScorer(0.1), tmp)
        gate.lock_threshold(0.5)
        result = gate.evaluate(frame=None, action_ref="ep-1:0")
        assert result.decision == GateDecision.EXECUTE


def test_high_score_blocks():
    with tempfile.TemporaryDirectory() as tmp:
        gate = make_gate(FakeScorer(0.9), tmp)
        gate.lock_threshold(0.5)
        result = gate.evaluate(frame=None, action_ref="ep-1:0")
        assert result.decision == GateDecision.BLOCK


def test_score_equal_to_threshold_blocks():
    """Boundary should fail closed, not open."""
    with tempfile.TemporaryDirectory() as tmp:
        gate = make_gate(FakeScorer(0.5), tmp)
        gate.lock_threshold(0.5)
        result = gate.evaluate(frame=None, action_ref="ep-1:0")
        assert result.decision == GateDecision.BLOCK


def test_scorer_failure_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        gate = make_gate(RaisingScorer(), tmp)
        gate.lock_threshold(0.5)
        result = gate.evaluate(frame=None, action_ref="ep-1:0")
        assert result.decision == GateDecision.BLOCK
        assert result.anomaly_score.source == "gate_failure_fallback"


def test_scorer_failure_raises_when_fail_closed_disabled():
    with tempfile.TemporaryDirectory() as tmp:
        gate = SafetyGate(
            scorer=RaisingScorer(),
            invariant_log_path=Path(tmp) / "invariant.jsonl",
            fail_closed=False,
        )
        gate.lock_threshold(0.5)
        with pytest.raises(RuntimeError):
            gate.evaluate(frame=None, action_ref="ep-1:0")


def test_threshold_can_only_be_locked_once():
    with tempfile.TemporaryDirectory() as tmp:
        gate = make_gate(FakeScorer(0.1), tmp)
        gate.lock_threshold(0.5)
        with pytest.raises(RuntimeError):
            gate.lock_threshold(0.7)


def test_evaluate_logs_to_invariant():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "invariant.jsonl"
        gate = SafetyGate(scorer=FakeScorer(0.9), invariant_log_path=log_path)
        gate.lock_threshold(0.5)
        gate.evaluate(frame=None, action_ref="ep-1:0")

        from src.safety.invariant_schema import InvariantLogger
        events = InvariantLogger(log_path).read_all()
        assert len(events) == 1
        assert events[0]["status"] == "blocked"
