import threading
import time

import pytest

from src.parallel.runtime import Observation, ParallelPipeline
from src.safety.anomaly_detector import AnomalyScore
from src.safety.gate import SafetyGate


class Scorer:
    def __init__(self, value=0):
        self.value = value

    def score(self, frame):
        return AnomalyScore(self.value, 0, "fake_for_testing")


def test_workers_overlap(tmp_path):
    barrier = threading.Barrier(2)
    def perceive(clip):
        barrier.wait(timeout=1)
        return {"ok": True}
    class PairedScorer:
        def score(self, clip):
            barrier.wait(timeout=1)
            return AnomalyScore(0, 0, "fake_for_testing")
    pipe = ParallelPipeline(perceive, PairedScorer(), 0.5, tmp_path / "log", 2)
    try:
        result = pipe.evaluate(Observation(0, time.monotonic(), None))
        assert result.decision == "execute"
        assert result.perception == {"ok": True}
    finally:
        pipe.close()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 0.5, 1])
def test_bad_scores_block(tmp_path, value):
    pipe = ParallelPipeline(lambda clip: {}, Scorer(value), 0.5, tmp_path / "log", 2)
    try:
        assert pipe.evaluate(Observation(0, time.monotonic(), None)).decision == "block"
    finally:
        pipe.close()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_invalid_threshold(tmp_path, value):
    gate = SafetyGate(Scorer(), tmp_path / "log")
    with pytest.raises(ValueError):
        gate.lock_threshold(value)


def test_stale_and_replayed_observations(tmp_path):
    pipe = ParallelPipeline(lambda clip: {}, Scorer(), 0.5, tmp_path / "log", 2)
    try:
        assert pipe.evaluate(Observation(0, time.monotonic() - 10, None)).decision == "block"
        assert pipe.evaluate(Observation(1, time.monotonic(), None)).decision == "execute"
        assert pipe.evaluate(Observation(1, time.monotonic(), None)).decision == "block"
    finally:
        pipe.close()


def test_timeout_latches_without_resubmitting(tmp_path):
    release = threading.Event()
    calls = []
    def hung(clip):
        calls.append(1)
        release.wait(2)
    pipe = ParallelPipeline(hung, Scorer(), 0.5, tmp_path / "log", 0.05)
    try:
        start = time.monotonic()
        assert pipe.evaluate(Observation(0, start, None)).decision == "block"
        assert time.monotonic() - start < 0.5
        assert pipe.evaluate(Observation(1, time.monotonic(), None)).decision == "block"
        assert len(calls) == 1
    finally:
        release.set()
        pipe.close()


def test_perception_failure_blocks(tmp_path):
    def fail(clip):
        raise RuntimeError("injected")
    pipe = ParallelPipeline(fail, Scorer(), 0.5, tmp_path / "log", 2)
    try:
        assert pipe.evaluate(Observation(0, time.monotonic(), None)).decision == "block"
    finally:
        pipe.close()


def test_warmup_uses_workers_and_cannot_authorize_or_reset(tmp_path):
    names = []
    def perceive(clip):
        names.append(threading.current_thread().name)
        return {}
    pipe = ParallelPipeline(perceive, Scorer(), .5, tmp_path / "log", 2)
    try:
        pipe.warmup(None)
        assert len(names) == 3
        assert all(name.startswith("inference") for name in names)
        assert pipe.gate.invariant.read_all() == []
        assert pipe.evaluate(Observation(0, time.monotonic(), None)).decision == "execute"
        with pytest.raises(RuntimeError):
            pipe.warmup(None)
    finally:
        pipe.close()
