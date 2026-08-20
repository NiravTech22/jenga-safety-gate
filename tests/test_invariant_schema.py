import tempfile
from pathlib import Path

from src.safety.invariant_schema import (
    InvariantEvent,
    InvariantLogger,
    NodeStatus,
    StageType,
    new_node_id,
)


def test_new_node_id_includes_stage():
    node_id = new_node_id(StageType.GATE_DECISION)
    assert node_id.startswith("gate_decision-")


def test_logger_appends_and_reads_events():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "invariant.jsonl"
        logger = InvariantLogger(log_path)

        event = InvariantEvent.start("node-1", StageType.PERCEPTION, input_ref="ep-1:0")
        logger.log(event)

        events = logger.read_all()
        assert len(events) == 1
        assert events[0]["node_id"] == "node-1"
        assert events[0]["stage"] == "perception"


def test_logger_filters_failures_and_blocks():
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "invariant.jsonl"
        logger = InvariantLogger(log_path)

        ok = InvariantEvent(node_id="a", stage=StageType.ACTUATION, status=NodeStatus.SUCCESS,
                             timestamp=0.0, latency_ms=1.0)
        fail = InvariantEvent(node_id="b", stage=StageType.ACTUATION, status=NodeStatus.FAILURE,
                               timestamp=0.0, latency_ms=1.0, failure_reason="stall")
        blocked = InvariantEvent(node_id="c", stage=StageType.GATE_DECISION, status=NodeStatus.BLOCKED,
                                  timestamp=0.0, latency_ms=1.0)

        for e in (ok, fail, blocked):
            logger.log(e)

        assert len(logger.failures()) == 1
        assert logger.failures()[0]["node_id"] == "b"
        assert len(logger.blocks()) == 1
        assert logger.blocks()[0]["node_id"] == "c"


def test_empty_log_returns_empty_list():
    with tempfile.TemporaryDirectory() as tmp:
        logger = InvariantLogger(Path(tmp) / "nonexistent.jsonl")
        assert logger.read_all() == []
