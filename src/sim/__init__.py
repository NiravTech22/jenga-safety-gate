"""Deterministic Jenga-tower simulation backend.

Exists so the full pipeline (Steps 1-9) can be run, tested, and regression-
checked end to end before the physical rig from PLAN.md Steps 1-2 exists.
It is a *substitute for hardware*, never a substitute for results: every
artifact produced through this backend is stamped `backend: sim` and any
comparison table generated from it carries a provenance banner saying the
numbers are simulated. See docs/SIMULATION.md.
"""
from src.sim.tower import (  # noqa: F401
    BLOCK_H_M,
    SLOTS_PER_LAYER,
    ExtractionOutcome,
    TowerSim,
    TowerSimConfig,
)
