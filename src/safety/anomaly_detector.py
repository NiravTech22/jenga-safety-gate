"""
Step 6: V-JEPA-distilled anomaly detector.

Two-part design per the plan:
  Part A (GPU machine, offline): distill a small student model from V-JEPA
    embedding-space prediction error.
  Part B (manual + this module): validate the student against 75 hand-labeled
    held-out frames.

Implemented: the AnomalyScore data structure, the student-model inference
interface, and the precision/recall validation harness (pure numpy, testable
without any model — see tests/test_anomaly_detector.py using a fake scorer).

Stubbed: actual V-JEPA embedding extraction and student-model training.
These need a GPU, a real V-JEPA checkpoint, and Step 4's dataset — none of
which exist yet. The interfaces below are the contract that Step 7 (the
gate) is written against, so building the gate doesn't have to wait on this.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class AnomalyScore:
    value: float          # higher = more anomalous, NOT bounded to [0,1] by contract
    latency_ms: float
    source: str            # "student_model" | "vjepa_direct" | "fake_for_testing"


class AnomalyScorer(Protocol):
    """Interface the gate (Step 7) depends on. Anything satisfying this
    protocol can be swapped in — the distilled student model, a raw V-JEPA
    model (too slow for the Pi but fine for offline eval), or a fake scorer
    for testing the gate logic in isolation.
    """

    def score(self, frame: np.ndarray) -> AnomalyScore: ...


class StudentAnomalyDetector:
    """The real Step 6 deliverable. Not implemented yet — needs:
      1. A pretrained V-JEPA checkpoint (Part A, item 1)
      2. Embeddings extracted over Step 4 data (Part A, item 2)
      3. A trained shallow CNN/MLP student (Part A, item 3)
    Interface is final; internals are not.
    """

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"No student model checkpoint at {checkpoint_path}. "
                "Run Step 6 Part A (distillation) first — see docs/PLAN.md."
            )
        # TODO: load the actual student model (shallow CNN / MLP) here once
        # it exists. Framework choice (torch vs onnx for Pi deployment) is
        # an open decision — bias toward onnxruntime for the edge deploy
        # given the Orange Pi Zero 2W's limited compute.
        raise NotImplementedError("Model loading not implemented — no trained checkpoint yet.")

    def score(self, frame: np.ndarray) -> AnomalyScore:
        raise NotImplementedError


@dataclass
class ValidationResult:
    precision: float
    recall: float
    f1: float
    n_normal: int
    n_risky: int
    threshold_used: float


def validate_against_labels(
    scorer: AnomalyScorer,
    labeled_frames: list[tuple[np.ndarray, bool]],  # (frame, is_risky)
    threshold: float,
) -> ValidationResult:
    """Part B of Step 6: score the 75-frame hand-labeled validation set and
    report precision/recall against the given threshold.

    This function is fully implemented and usable today with a fake scorer
    for testing, or with the real student model once it exists — nothing
    here needs to change when Part A is done.
    """
    tp = fp = fn = tn = 0
    n_normal = n_risky = 0

    for frame, is_risky in labeled_frames:
        predicted_risky = scorer.score(frame).value >= threshold
        n_risky += int(is_risky)
        n_normal += int(not is_risky)

        if is_risky and predicted_risky:
            tp += 1
        elif is_risky and not predicted_risky:
            fn += 1
        elif not is_risky and predicted_risky:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return ValidationResult(
        precision=precision,
        recall=recall,
        f1=f1,
        n_normal=n_normal,
        n_risky=n_risky,
        threshold_used=threshold,
    )


def clears_random_baseline(result: ValidationResult, floor: float = 0.6) -> bool:
    """Step 6 exit condition check: does the detector clear a precision/
    recall bar meaningfully above chance? `floor` should come from
    config.yaml's anomaly_detector.precision_recall_floor.
    """
    return result.precision >= floor and result.recall >= floor
