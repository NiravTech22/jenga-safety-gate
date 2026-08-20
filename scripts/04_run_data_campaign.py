#!/usr/bin/env python3
"""Step 4: full data collection campaign (22+ sessions across 6 scenario
categories — see PLAN.md for the matrix).

BLOCKED until Step 1 (camera rig) and Step 2 (tower-state CV, 20/20
validated) are physically done. This script's job, once unblocked, is to
loop scripts/03_record_episode.py (real, non-dry-run) across the scenario
matrix in config.yaml's data.min_sessions_per_category, track progress
against docs/PROGRESS.md, and refuse to let you under-collect a category.

Not implemented yet — intentionally, per the plan's sequencing.
"""
raise NotImplementedError(
    "Blocked on Step 1 (camera rig) and Step 2 (tower-state CV, validated "
    "20/20). See docs/PLAN.md."
)
