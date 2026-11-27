import pytest

from src.hardware.passive import TelemetryStale, TelemetryWatch, wait_for_live


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt


class Source:
    path = "logs/arm_latest.json"

    def __init__(self, clock, live):
        self.clock, self.live, self.seq = clock, live, 7

    def read(self):
        if self.live:
            self.seq += 1
            return dict(sequence_id=self.seq, state_timestamp=self.clock())
        return dict(sequence_id=self.seq, state_timestamp=self.clock() - 120)   # frozen file


def test_live_telemetry_passes():
    c = Clock()
    assert wait_for_live(Source(c, live=True), clock=c, sleep=c.sleep)["sequence_id"] > 7


def test_frozen_telemetry_refuses_to_record():
    c = Clock()
    with pytest.raises(TelemetryStale, match="not updating"):
        wait_for_live(Source(c, live=False), timeout=1.0, clock=c, sleep=c.sleep)


def test_watch_warns_once_then_stops():
    c = Clock()
    w = TelemetryWatch(warn_s=1.0, stop_s=5.0, clock=c)
    assert w.check(dict(sequence_id=1)) is None
    c.t += 0.5; assert w.check(dict(sequence_id=1)) is None
    c.t += 1.0; assert "frozen" in w.check(dict(sequence_id=1))
    c.t += 1.0; assert w.check(dict(sequence_id=1)) is None          # warned once
    assert w.check(dict(sequence_id=2)) is None                      # recovered
    c.t += 6.0
    with pytest.raises(TelemetryStale, match="stopped updating"):
        w.check(dict(sequence_id=2))
