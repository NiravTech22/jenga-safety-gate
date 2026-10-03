"""Replay held-out MuJoCo RGB clips through both trained CUDA branches."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.parallel.models import MotionPerception, VjepaRiskScorer
from src.parallel.runtime import Observation, ParallelPipeline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", default="checkpoints/mujoco_vjepa")
    p.add_argument("--dataset", default="data/mujoco_jenga")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--max-age", type=float, default=.5)
    p.add_argument("--video", help="Optional MP4 of the same seeded physics rollouts and gate decisions")
    args = p.parse_args()
    if args.steps <= 0:
        p.error("steps must be positive")
    torch.set_num_threads(4)
    directory, dataset = Path(args.directory), Path(args.dataset)
    motion = MotionPerception(directory / "motion.pt")
    scorer = VjepaRiskScorer(directory)
    records = [r for r in json.loads((dataset / "manifest.json").read_text())["records"] if r["split"] == "test"]
    warmup = np.zeros((4, 224, 224, 3), dtype=np.uint8)
    pipe = ParallelPipeline(motion, scorer, scorer.threshold, "logs/mujoco_parallel.jsonl", args.max_age)
    try:
        pipe.warmup(warmup)
    except Exception:
        pipe.close()
        raise
    writer = None
    if args.video:
        import cv2
        from src.sim.mujoco_jenga import JengaPhysics
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), 20, (480, 480))
        if not writer.isOpened():
            pipe.close()
            raise RuntimeError("Cannot open video output")
    try:
        for i, record in enumerate(records[:args.steps]):
            with np.load(dataset / record["file"], allow_pickle=False) as data:
                clip = data["clip"]
            start = time.monotonic()
            result = pipe.evaluate(Observation(i, start, clip))
            print(json.dumps(dict(seed=record["seed"], backend="mujoco_replay", decision=result.decision,
                reason=result.reason, actual_future_failure=record["risky"],
                latency_ms=(time.monotonic()-start)*1000, motor_commands_sent=0)), flush=True)
            if writer is not None:
                env = JengaPhysics(record["seed"], render_size=480)
                try:
                    rng = np.random.default_rng(record["seed"])
                    force = float(rng.uniform(0, .75))
                    direction = float(rng.uniform(-np.pi, np.pi))
                    env.observe_disturbance(force, direction)
                    for _ in range(20):
                        frame = cv2.cvtColor(env.frame("overview"), cv2.COLOR_RGB2BGR)
                        text = f"Seed {record['seed']} | {result.decision.upper()}"
                        cv2.putText(frame, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                        cv2.putText(frame, "Counterfactual physics continues; arm parked", (12, 465), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
                        writer.write(frame)
                        env.step(.05)
                    if env.failed() != record["risky"]:
                        raise RuntimeError("Replay physics no longer matches recorded outcome")
                finally:
                    env.close()
    finally:
        pipe.close()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    main()
