"""Bounded, same-observation parallel inference. Never sends motor commands."""
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
import math
import threading
import time

from src.safety.gate import SafetyGate, GateDecision
from src.safety.invariant_schema import InvariantEvent, NodeStatus, StageType, new_node_id


@dataclass(frozen=True)
class Observation:
    sequence: int
    captured_at: float  # time.monotonic(), in this process
    clip: object       # RGB uint8 [T,H,W,3], owned by caller until evaluate returns


@dataclass(frozen=True)
class Result:
    sequence: int
    decision: str
    reason: str
    perception: object = None


class ParallelPipeline:
    """Two concurrent workers; one observation in flight, no growing work queue.

    Timeout or worker failure latches a fault: restart the pipeline after fixing
    the cause. Python threads cannot forcibly cancel a blocked CUDA call.
    This is a decision interface, not a real-time motor watchdog.
    """

    def __init__(self, perceive, scorer, threshold, log_path, max_age_s=0.5):
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("max_age_s must be finite and positive")
        self.perceive = perceive
        self.gate = SafetyGate(scorer, log_path)
        self.gate.lock_threshold(threshold)
        self.max_age_s = max_age_s
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="inference")
        self.lock = threading.Lock()
        self.fault = None
        self.last_sequence = -1

    def warmup(self, clip, passes=3, timeout_s=30):
        """Initialize CUDA/JIT on the actual workers; never authorize actions.

        Call before observations. The longer startup timeout must not be used
        for live decisions. Startup failure latches the same runtime fault.
        """
        if passes <= 0 or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Invalid warmup limits")
        with self.lock:
            if self.fault or self.last_sequence >= 0:
                raise RuntimeError("Warmup is only allowed before evaluation")
            for _ in range(passes):
                tasks = (self.pool.submit(self.perceive, clip),
                         self.pool.submit(self.gate.scorer.score, clip))
                _, pending = wait(tasks, timeout=timeout_s)
                if pending:
                    self.fault = "warmup timeout; restart required"
                    for task in pending:
                        task.cancel()
                    raise TimeoutError(self.fault)
                try:
                    for task in tasks:
                        task.result()
                except Exception:
                    self.fault = "warmup failure; restart required"
                    raise

    def _block(self, obs, reason):
        self.gate.invariant.log(InvariantEvent(
            node_id=new_node_id(StageType.GATE_DECISION),
            stage=StageType.GATE_DECISION, status=NodeStatus.BLOCKED,
            timestamp=time.time(), latency_ms=0,
            input_ref=str(obs.sequence), output_ref=GateDecision.BLOCK,
            failure_reason=reason))
        return Result(obs.sequence, GateDecision.BLOCK, reason)

    def evaluate(self, obs):
        if not self.lock.acquire(blocking=False):
            return self._block(obs, "pipeline busy")
        try:
            if self.fault:
                return self._block(obs, self.fault)
            age = time.monotonic() - obs.captured_at
            if not math.isfinite(age) or age < 0 or age >= self.max_age_s:
                return self._block(obs, "observation stale or timestamp invalid")
            if obs.sequence <= self.last_sequence:
                return self._block(obs, "observation sequence replayed")
            self.last_sequence = obs.sequence
            perception = self.pool.submit(self.perceive, obs.clip)
            gate = self.pool.submit(self.gate.evaluate, obs.clip, str(obs.sequence))
            _, pending = wait((perception, gate), timeout=self.max_age_s - age)
            if pending:
                for task in pending:
                    task.cancel()
                self.fault = "inference timeout; restart required"
                return self._block(obs, self.fault)
            try:
                output, verdict = perception.result(), gate.result()
            except Exception as exc:
                self.fault = f"inference failure: {type(exc).__name__}: {exc}"
                return self._block(obs, self.fault)
            if time.monotonic() - obs.captured_at >= self.max_age_s:
                return self._block(obs, "result expired")
            if verdict.decision != GateDecision.EXECUTE:
                return self._block(obs, "anomaly gate blocked")
            return Result(obs.sequence, GateDecision.EXECUTE, "both workers completed", output)
        finally:
            self.lock.release()

    def close(self):
        self.fault = "pipeline closed"
        self.pool.shutdown(wait=False, cancel_futures=True)
