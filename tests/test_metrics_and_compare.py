import pytest

from src.eval.compare import build_comparison_table, evaluate_thesis, render_markdown
from src.eval.metrics import Trial, TrialResult, summarize


def test_summarize_basic_rates():
    trials = [
        Trial(TrialResult.SUCCESS),
        Trial(TrialResult.SUCCESS),
        Trial(TrialResult.CATASTROPHIC_FAILURE),
        Trial(TrialResult.NEAR_MISS),
    ]
    s = summarize(trials)
    assert s.n_trials == 4
    assert s.success_rate == 0.5
    assert s.catastrophic_failure_rate == 0.25
    assert s.near_miss_rate == 0.25
    assert s.gate_false_positive_rate is None
    assert s.mean_added_latency_ms is None


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        summarize([])


def test_summarize_gated_trials_compute_fp_rate_and_latency():
    trials = [
        Trial(TrialResult.SUCCESS, gate_present=True, gate_blocked=False, added_latency_ms=5.0),
        Trial(TrialResult.SUCCESS, gate_present=True, gate_blocked=True,
              gate_incorrectly_blocked=True, added_latency_ms=7.0),
        Trial(TrialResult.CATASTROPHIC_FAILURE, gate_present=True, gate_blocked=True,
              gate_incorrectly_blocked=False, added_latency_ms=6.0),
    ]
    s = summarize(trials)
    assert s.gate_false_positive_rate == pytest.approx(0.5)  # 1 of 2 blocks was incorrect
    assert s.mean_added_latency_ms == pytest.approx(6.0)


def test_thesis_supported_when_catastrophic_drops_and_success_holds():
    ungated = summarize([
        Trial(TrialResult.SUCCESS), Trial(TrialResult.SUCCESS),
        Trial(TrialResult.CATASTROPHIC_FAILURE), Trial(TrialResult.CATASTROPHIC_FAILURE),
    ])
    gated = summarize([
        Trial(TrialResult.SUCCESS, gate_present=True),
        Trial(TrialResult.SUCCESS, gate_present=True),
        Trial(TrialResult.SUCCESS, gate_present=True, gate_blocked=True, gate_incorrectly_blocked=False),
        Trial(TrialResult.NEAR_MISS, gate_present=True),
    ])
    verdict = evaluate_thesis(ungated, gated)
    assert verdict.supported is True


def test_thesis_not_supported_when_gate_overblocks():
    ungated = summarize([
        Trial(TrialResult.SUCCESS), Trial(TrialResult.CATASTROPHIC_FAILURE),
    ])
    gated = summarize([
        Trial(TrialResult.SUCCESS, gate_present=True, gate_blocked=True, gate_incorrectly_blocked=True),
        Trial(TrialResult.NEAR_MISS, gate_present=True, gate_blocked=True, gate_incorrectly_blocked=True),
    ])
    verdict = evaluate_thesis(ungated, gated)
    assert verdict.supported is False
    assert "false-positive" in verdict.reasoning


def test_thesis_not_supported_when_success_collapses():
    ungated = summarize([Trial(TrialResult.SUCCESS)] * 4)
    gated = summarize([
        Trial(TrialResult.CATASTROPHIC_FAILURE, gate_present=True),
        Trial(TrialResult.CATASTROPHIC_FAILURE, gate_present=True),
        Trial(TrialResult.NEAR_MISS, gate_present=True),
        Trial(TrialResult.NEAR_MISS, gate_present=True),
    ])
    verdict = evaluate_thesis(ungated, gated)
    assert verdict.supported is False


def test_render_markdown_contains_verdict():
    ungated = summarize([Trial(TrialResult.SUCCESS), Trial(TrialResult.CATASTROPHIC_FAILURE)])
    gated = summarize([
        Trial(TrialResult.SUCCESS, gate_present=True),
        Trial(TrialResult.SUCCESS, gate_present=True, gate_blocked=True, gate_incorrectly_blocked=False),
    ])
    rows = build_comparison_table(ungated, gated)
    verdict = evaluate_thesis(ungated, gated)
    md = render_markdown(rows, verdict)
    assert "Success rate" in md
    assert "SUPPORTED" in md
