#!/usr/bin/env python3
"""Step 7: wire the Invariant safety gate to the Step 5 policy + Step 6
detector, lock the threshold, and dry-run against an obviously-bad and an
obviously-fine proposed action.

The gate logic itself (src/safety/gate.py) is fully implemented and tested
today with a fake scorer — see tests/test_gate.py. This script is what
connects it to the REAL Step 6 model and locks the threshold for real,
which is blocked until Step 6 produces a validated student model.
"""
raise NotImplementedError("Blocked on Step 6 (trained + validated anomaly detector). See docs/PLAN.md.")
