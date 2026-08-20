#!/usr/bin/env python3
"""Step 3: record one episode and validate time alignment.

This is a thin CLI around src/data/recorder.py. Wire your actual VR teleop
loop's per-tick callback to call `recorder.record_timestep(...)` — this
script just shows the shape of that integration with a dry-run mode that
uses synthetic frames so you can test the recorder/schema without hardware.

Usage:
  python scripts/03_record_episode.py --scenario stable_easy --dry-run
  python scripts/03_record_episode.py --scenario stable_easy   # real hardware
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

from src.data.recorder import EpisodeRecorder
from src.data.schema import EpisodeOutcome, ScenarioCategory

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def dry_run(recorder: EpisodeRecorder, n_steps: int = 20):
    """Synthetic frames/actions so the recorder + schema path is exercised
    without any camera or arm connected. This is what CI / a fresh clone
    should run to prove the pipeline is wired correctly.
    """
    for i in range(n_steps):
        fake_overhead = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_side = np.zeros((480, 640, 3), dtype=np.uint8)
        joint_state = [0.0] * 6
        action = [0.0] * 6
        recorder.record_timestep(fake_overhead, fake_side, joint_state, action)
        time.sleep(0.01)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=[c.value for c in ScenarioCategory])
    parser.add_argument("--dry-run", action="store_true",
                         help="Use synthetic frames instead of real cameras/arm.")
    parser.add_argument("--operator", default="unknown")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg["data"]["dataset_root"])
    invariant_log = Path("./logs/invariant.jsonl")

    recorder = EpisodeRecorder(
        dataset_root=dataset_root,
        scenario_category=ScenarioCategory(args.scenario),
        invariant_log_path=invariant_log,
        operator=args.operator,
    )

    if args.dry_run:
        print("Dry run: recording synthetic timesteps...")
        dry_run(recorder)
        outcome = EpisodeOutcome.UNLABELED
    else:
        raise NotImplementedError(
            "Real recording needs your VR teleop loop wired to call "
            "recorder.record_timestep() per tick, and FrameCapture pointed "
            "at real camera devices (Step 1). Use --dry-run until then."
        )

    saved_path = recorder.close(outcome, notes="recorded via scripts/03_record_episode.py")
    print(f"Episode saved to: {saved_path}")


if __name__ == "__main__":
    main()
