#!/usr/bin/env python3
"""Step 5: train ACT via LeRobot on the Step 4 dataset, evaluate on held-out
scenarios, and freeze the resulting metrics table (src/eval/metrics.py).

BLOCKED until Step 4 produces a real dataset. Once unblocked:
  1. pip install the lerobot lines in requirements.txt
  2. Convert src/data/schema.py Episodes into LeRobot's dataset format
  3. Train ACT (config.yaml policy.* settings)
  4. Run N held-out trials, build a list[Trial] (src/eval/metrics.py),
     call summarize(), and save the result — this table does not change
     once recorded (see PLAN.md Step 5 exit condition).

Not implemented yet — intentionally, per the plan's sequencing.
"""
raise NotImplementedError("Blocked on Step 4 (data collection campaign). See docs/PLAN.md.")
