#!/usr/bin/env python3
"""Step 2: validate tower-state detection against known configurations.

PLAN.md's exit condition is 20 correct reads out of 20 manual trials, and it
is load-bearing rather than decorative: the anomaly detector and the policy
are both scored against this pipeline's output, so an untrustworthy reader
silently corrupts every number produced from Step 5 onward. This script
exits non-zero on anything below 20/20.

"Correct" means the occupancy grid matches exactly, every slot, and the lean
estimate is within perception.validation.lean_tolerance_deg. Approximately
right occupancy is not right: a missed block is a missed hazard.

Usage:
  python scripts/02_validate_tower_state.py
  python scripts/02_validate_tower_state.py --save-failures   # dump frames
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

from src.perception.tower_state import (
    ArucoFrameCalibrator,
    OccupancyModel,
    get_tower_state,
)
from src.sim.render import default_rig, render_frame_pair, save_frame_pair
from src.sim.tower import TowerSim, TowerSimConfig

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


@dataclass
class TrialSpec:
    """One deliberately-built tower. The hardware equivalent is a person
    building this exact configuration and writing it on a sheet of paper."""
    name: str
    category: str
    pre_removed: tuple
    lean_deg: float


def build_trials(n_layers: int) -> list:
    """Twenty known configurations spanning the Step 4 scenario categories.

    Weighted toward the awkward end on purpose. A validation set of pristine
    towers proves the reader works in the case that never matters.
    """
    trials = [
        TrialSpec("full_00", "stable_easy", (), 0.0),
        TrialSpec("full_01", "stable_easy", (), 0.4),
        TrialSpec("single_gap_low", "stable_easy", ((2, 1),), 0.0),
        TrialSpec("single_gap_high", "stable_easy", ((9, 0),), 0.0),
        TrialSpec("two_gaps", "stable_tight", ((3, 0), (6, 2)), 0.3),
        TrialSpec("three_gaps", "stable_tight", ((2, 2), (5, 1), (8, 0)), 0.0),
        TrialSpec("adjacent_gaps", "stable_tight", ((4, 0), (4, 2)), 0.0),
        TrialSpec("stacked_gaps", "stable_tight", ((5, 1), (6, 1)), 0.2),

        TrialSpec("lean_2deg", "leaning", ((3, 1),), 2.0),
        TrialSpec("lean_4deg", "leaning", ((4, 0), (7, 2)), 4.0),
        TrialSpec("lean_6deg", "leaning", ((2, 1), (5, 0)), 6.0),
        TrialSpec("lean_negative", "leaning", ((6, 1),), -3.0),

        TrialSpec("near_collapse_a", "near_collapse",
                  ((2, 0), (2, 2), (4, 1), (5, 0), (7, 1)), 1.5),
        TrialSpec("near_collapse_b", "near_collapse",
                  ((1, 1), (3, 0), (3, 2), (6, 1), (8, 2)), 3.0),
        TrialSpec("near_collapse_c", "near_collapse",
                  ((2, 1), (4, 0), (4, 2), (6, 0), (7, 1), (9, 2)), 2.0),
        TrialSpec("hollow_layer", "near_collapse", ((5, 0), (5, 1), (5, 2)), 0.5),

        TrialSpec("off_center_a", "off_center", ((3, 0), (4, 0), (5, 0)), 1.0),
        TrialSpec("off_center_b", "off_center", ((2, 2), (3, 2), (6, 2)), 1.0),
        TrialSpec("off_center_c", "off_center", ((1, 0), (4, 2), (7, 0), (8, 2)), 2.5),
        TrialSpec("recovery_state", "recovery", ((3, 1), (6, 0)), 1.2),
    ]
    for t in trials:
        for layer, slot in t.pre_removed:
            assert 0 <= layer < n_layers, f"{t.name}: layer {layer} out of range"
    return trials


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-failures", action="store_true",
                        help="Write frames for any trial that fails, for inspection.")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    n_layers = int(cfg["perception"]["num_layers"])
    tol = float(cfg["perception"]["validation"]["lean_tolerance_deg"])
    required = int(cfg["perception"]["validation"]["trials"])
    calib_dir = Path(cfg["perception"]["calibration_dir"])

    if not (calib_dir / "occupancy_model.json").exists():
        print("No calibration found. Run scripts/01_capture_reference.py first.")
        sys.exit(1)

    cal_side = ArucoFrameCalibrator.load(calib_dir / "side_calibration.json")
    cal_over = ArucoFrameCalibrator.load(calib_dir / "overhead_calibration.json")
    model = OccupancyModel.load(calib_dir / "occupancy_model.json")

    rig = default_rig()
    trials = build_trials(n_layers)
    if len(trials) < required:
        print(f"Only {len(trials)} trials defined; Step 2 requires {required}.")
        sys.exit(1)

    print(f"Validating tower-state detection over {len(trials)} known "
          f"configurations (lean tolerance {tol} deg)\n")
    print(f"{'trial':18s} {'category':14s} {'slots':>7s} {'lean err':>9s}  result")
    print("-" * 62)

    results, n_pass = [], 0
    for i, spec in enumerate(trials):
        sim = TowerSim(TowerSimConfig(seed=4000 + i, num_layers=n_layers,
                                      pre_removed=spec.pre_removed,
                                      initial_lean_deg=spec.lean_deg))
        over, side = render_frame_pair(sim.state, rig, np.random.default_rng(5000 + i))
        state = get_tower_state(over, side, cal_side, cal_over, n_layers, model)

        detected = state.occupancy_array()
        truth = sim.state.occupancy
        wrong = int((detected != truth).sum())
        lean_err = abs(state.lean_angle_deg - sim.state.lean_deg)
        ok = (wrong == 0) and (lean_err <= tol)
        n_pass += int(ok)

        print(f"{spec.name:18s} {spec.category:14s} {truth.size - wrong:3d}/{truth.size:<3d} "
              f"{lean_err:8.2f}  {'PASS' if ok else 'FAIL'}")

        results.append({
            "trial": spec.name, "category": spec.category,
            "slots_correct": int(truth.size - wrong), "slots_total": int(truth.size),
            "lean_true_deg": float(sim.state.lean_deg),
            "lean_detected_deg": float(state.lean_angle_deg),
            "lean_error_deg": float(lean_err),
            "confidence": float(state.confidence),
            "passed": bool(ok),
        })

        if not ok and args.save_failures:
            save_frame_pair(over, side, Path("./data/validation_failures"), spec.name)

    total_slots = sum(r["slots_total"] for r in results)
    correct_slots = sum(r["slots_correct"] for r in results)
    mean_lean_err = float(np.mean([r["lean_error_deg"] for r in results]))

    print("-" * 62)
    print(f"Trials passed:   {n_pass}/{len(trials)}")
    print(f"Slot accuracy:   {correct_slots}/{total_slots} "
          f"({100.0 * correct_slots / total_slots:.2f}%)")
    print(f"Mean lean error: {mean_lean_err:.2f} deg")

    report_path = Path("./data/validation/step02_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "backend": cfg["runtime"]["backend"],
        "num_layers": n_layers,
        "lean_tolerance_deg": tol,
        "trials_passed": n_pass,
        "trials_total": len(trials),
        "slot_accuracy": correct_slots / total_slots,
        "mean_lean_error_deg": mean_lean_err,
        "results": results,
    }, indent=2))
    print(f"Report written to {report_path}")

    if n_pass < len(trials):
        print("\nStep 2 exit condition NOT met. Do not proceed to Step 3 — "
              "everything downstream assumes this pipeline is trustworthy.\n"
              "Re-run with --save-failures to inspect the frames that failed.")
        sys.exit(1)

    print("\nStep 2 exit condition met: 20/20 on known configurations.")


if __name__ == "__main__":
    main()
