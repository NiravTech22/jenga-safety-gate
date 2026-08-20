"""
Metrics for Step 5 (baseline) and Step 8 (gated) runs.

Fully implemented — this is pure arithmetic over trial records, no hardware
or ML dependency. See tests/test_metrics.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrialResult(str, Enum):
    SUCCESS = "success"
    CATASTROPHIC_FAILURE = "catastrophic_failure"   # full collapse
    NEAR_MISS = "near_miss"                          # instability, no collapse


@dataclass
class Trial:
    result: TrialResult
    gate_present: bool = False
    gate_blocked: bool = False          # did the gate block at least one proposed action this trial?
    gate_incorrectly_blocked: bool = False  # human-reviewed: was that block actually unnecessary?
    added_latency_ms: float = 0.0       # only meaningful if gate_present


@dataclass
class MetricsSummary:
    n_trials: int
    success_rate: float
    catastrophic_failure_rate: float
    near_miss_rate: float
    gate_false_positive_rate: float | None   # None if gate not present
    mean_added_latency_ms: float | None      # None if gate not present


def summarize(trials: list[Trial]) -> MetricsSummary:
    if not trials:
        raise ValueError("Cannot summarize zero trials.")

    n = len(trials)
    success = sum(1 for t in trials if t.result == TrialResult.SUCCESS)
    catastrophic = sum(1 for t in trials if t.result == TrialResult.CATASTROPHIC_FAILURE)
    near_miss = sum(1 for t in trials if t.result == TrialResult.NEAR_MISS)

    gated_trials = [t for t in trials if t.gate_present]
    if gated_trials:
        blocked_trials = [t for t in gated_trials if t.gate_blocked]
        fp_rate = (
            sum(1 for t in blocked_trials if t.gate_incorrectly_blocked) / len(blocked_trials)
            if blocked_trials else 0.0
        )
        mean_latency = sum(t.added_latency_ms for t in gated_trials) / len(gated_trials)
    else:
        fp_rate = None
        mean_latency = None

    return MetricsSummary(
        n_trials=n,
        success_rate=success / n,
        catastrophic_failure_rate=catastrophic / n,
        near_miss_rate=near_miss / n,
        gate_false_positive_rate=fp_rate,
        mean_added_latency_ms=mean_latency,
    )
