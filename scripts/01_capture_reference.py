#!/usr/bin/env python3
"""Step 1: camera rig reference capture and calibration.

Produces the artifacts PLAN.md Step 1 asks for (reference photos of the empty
table and of a built tower) and the two things Step 2 needs from them: a
per-camera pose recovered from the base markers, and the occupancy brightness
model fitted from known-full-tower frames.

Calibration runs against the EMPTY table on purpose. A frame containing a
tower has markers occluded by it, and a pose fitted to whatever markers happen
to survive occlusion is a pose that changes with the thing you are measuring.

Usage:
  python scripts/01_capture_reference.py                 # uses config backend
  python scripts/01_capture_reference.py --backend sim
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

from src.perception.tower_state import ArucoFrameCalibrator, fit_occupancy_model
from src.sim.render import default_rig, render_empty_table, render_frame_pair, save_frame_pair
from src.sim.tower import TowerSim, TowerSimConfig

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
MAX_ACCEPTABLE_REPROJ_PX = 3.0


def capture_sim(cfg, n_pairs: int, seed: int):
    """Render the reference set. On hardware this is the function that gets
    replaced with real VideoCapture reads; everything after it is identical."""
    rig = default_rig()
    empty = render_empty_table(rig, np.random.default_rng(seed))

    towers = []
    for i in range(n_pairs):
        # Full towers, varied only by block tightness and sensor noise: these
        # frames define what "occupied" looks like, so nothing may be missing.
        sim = TowerSim(TowerSimConfig(seed=seed + 1000 + i,
                                      num_layers=cfg["perception"]["num_layers"]))
        towers.append(render_frame_pair(sim.state, rig, np.random.default_rng(seed + 2000 + i)))
    return rig, empty, towers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sim", "hardware"], default=None)
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    backend = args.backend or cfg["runtime"]["backend"]
    seed = int(cfg["runtime"]["seed"])
    n_layers = int(cfg["perception"]["num_layers"])
    n_pairs = int(cfg["perception"]["reference_pairs"])

    if backend == "hardware":
        raise NotImplementedError(
            "Hardware capture needs cameras.overhead.device_index and "
            "cameras.side.device_index set in config.yaml, the rig physically "
            "locked down, and cameras.*.fixed set true. Replace capture_sim "
            "with cv2.VideoCapture reads and re-run. Everything downstream of "
            "capture is backend-independent."
        )

    print(f"Backend: {backend}  (artifacts will be stamped accordingly)")
    rig, (empty_over, empty_side), towers = capture_sim(cfg, n_pairs, seed)

    photo_dir = Path(cfg["data"]["reference_photos_dir"])
    save_frame_pair(empty_over, empty_side, photo_dir, "empty_table")
    for i, (o, s) in enumerate(towers):
        save_frame_pair(o, s, photo_dir, f"tower_{i:02d}")
    print(f"Saved {n_pairs + 1} reference pairs to {photo_dir}")

    # -- calibrate both cameras from the empty table -----------------------
    K = rig.side.intrinsics.matrix()
    calibrators = {}
    for name, frame in (("overhead", empty_over), ("side", empty_side)):
        cal = ArucoFrameCalibrator(K, layout=rig.markers)
        if not cal.calibrate(frame):
            print(f"FAILED: {name} camera saw only {cal.n_markers_seen} markers; "
                  f"need at least 3. Check marker placement and framing.")
            sys.exit(1)
        err = cal.reprojection_error_px()
        print(f"  {name:9s} pose from {cal.n_markers_seen} markers, "
              f"reprojection RMS {err:.2f} px")
        if err > MAX_ACCEPTABLE_REPROJ_PX:
            print(f"FAILED: {name} reprojection error exceeds "
                  f"{MAX_ACCEPTABLE_REPROJ_PX} px. Pose is not trustworthy.")
            sys.exit(1)
        calibrators[name] = cal

    # -- fit the occupancy model from full-tower frames --------------------
    model = fit_occupancy_model([s for _, s in towers], calibrators["side"], n_layers,
                                sigma=float(cfg["perception"]["occupancy_sigma"]))
    print(f"  occupancy model: threshold {model.threshold:.1f} "
          f"(occupied mean {model.occupied_mean:.1f} +/- {model.occupied_std:.1f}, "
          f"n={model.n_samples})")

    calib_dir = Path(cfg["perception"]["calibration_dir"])
    for name, cal in calibrators.items():
        cal.save(calib_dir / f"{name}_calibration.json")
    model.save(calib_dir / "occupancy_model.json")
    print(f"Calibration written to {calib_dir}")

    print("\nStep 1 exit condition met: rig fixed, reference photos saved, "
          "both cameras calibrated.\nProceed to scripts/02_validate_tower_state.py.")


if __name__ == "__main__":
    main()
