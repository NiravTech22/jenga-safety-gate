#!/usr/bin/env python3
"""Step 6: V-JEPA-distilled anomaly detector (Part A: GPU distillation,
Part B: validation against 75 hand-labeled frames).

BLOCKED on Step 4 (needs data to distill from) for Part A. Part B (manual
labeling) CAN start as soon as any Step 4 frames exist, in parallel with
Part A per the plan.

The validation harness itself (src/safety/anomaly_detector.py:
validate_against_labels, clears_random_baseline) is already implemented and
testable today — see tests/test_anomaly_detector.py, which exercises it
with a fake scorer.

Not implemented yet: the actual distillation training loop.
"""
raise NotImplementedError("Blocked on Step 4 (data). See docs/PLAN.md.")
