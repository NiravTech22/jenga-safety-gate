import numpy as np

from src.safety.anomaly_detector import (
    AnomalyScore,
    clears_random_baseline,
    validate_against_labels,
)


class PerfectScorer:
    """Scores exactly the risky/normal labels it's given, via a lookup
    keyed by array identity — a way to test the validation harness with a
    known-perfect detector."""

    def __init__(self, label_map):
        self.label_map = label_map  # id(frame) -> is_risky

    def score(self, frame):
        is_risky = self.label_map[id(frame)]
        return AnomalyScore(value=1.0 if is_risky else 0.0, latency_ms=0.1, source="fake_for_testing")


class AlwaysNormalScorer:
    def score(self, frame):
        return AnomalyScore(value=0.0, latency_ms=0.1, source="fake_for_testing")


def make_labeled_frames(n_normal=10, n_risky=10):
    frames = []
    label_map = {}
    for _ in range(n_normal):
        f = np.zeros((4, 4))
        frames.append((f, False))
        label_map[id(f)] = False
    for _ in range(n_risky):
        f = np.ones((4, 4))
        frames.append((f, True))
        label_map[id(f)] = True
    return frames, label_map


def test_perfect_scorer_yields_perfect_precision_recall():
    frames, label_map = make_labeled_frames()
    scorer = PerfectScorer(label_map)
    result = validate_against_labels(scorer, frames, threshold=0.5)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.n_normal == 10
    assert result.n_risky == 10


def test_always_normal_scorer_yields_zero_recall():
    frames, _ = make_labeled_frames()
    scorer = AlwaysNormalScorer()
    result = validate_against_labels(scorer, frames, threshold=0.5)
    assert result.recall == 0.0


def test_clears_random_baseline_true_for_perfect():
    frames, label_map = make_labeled_frames()
    scorer = PerfectScorer(label_map)
    result = validate_against_labels(scorer, frames, threshold=0.5)
    assert clears_random_baseline(result, floor=0.6) is True


def test_clears_random_baseline_false_for_useless_scorer():
    frames, _ = make_labeled_frames()
    scorer = AlwaysNormalScorer()
    result = validate_against_labels(scorer, frames, threshold=0.5)
    assert clears_random_baseline(result, floor=0.6) is False
