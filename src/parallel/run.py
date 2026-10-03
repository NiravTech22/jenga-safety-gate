"""Run from repository root: python -m src.parallel.run --smoke."""
import argparse
from collections import deque
import json
import time

import numpy as np
import torch

from src.parallel.models import MotionPerception, LocalStudent
from src.parallel.runtime import Observation, ParallelPipeline
from src.safety.anomaly_detector import AnomalyScore


class SmokeScorer:
    def score(self, clip):
        # Deliberately synthetic. Never presented as a V-JEPA detector.
        return AnomalyScore(float(clip.mean() / 255), 0, "fake_for_testing")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--perception-checkpoint")
    p.add_argument("--student")
    p.add_argument("--student-metadata")
    p.add_argument("--video", help="Video filename or integer camera index")
    p.add_argument("--threshold", type=float)
    p.add_argument("--max-age", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=20)
    args = p.parse_args()
    if args.steps <= 0:
        p.error("--steps must be positive")
    if not args.smoke and not all((args.perception_checkpoint, args.student,
                                  args.student_metadata, args.video,
                                  args.threshold is not None)):
        p.error("Real inference requires both checkpoints, student metadata, video, and validated threshold")
    motion = MotionPerception(args.perception_checkpoint, args.device, args.smoke)
    scorer = SmokeScorer() if args.smoke else LocalStudent(
        args.student, args.student_metadata, args.device)
    frames = motion.config["frames"]
    if not args.smoke and scorer.config["frames"] != frames:
        p.error("Both checkpoint contracts must accept the same clip length")
    threshold = 0.5 if args.smoke else args.threshold
    pipeline = ParallelPipeline(motion, scorer, threshold, "logs/parallel.jsonl", args.max_age)
    capture = None
    try:
        # Warm up before timing; these outputs never authorize an action.
        warmup = np.zeros((frames, 128, 128, 3), dtype=np.uint8)
        pipeline.warmup(warmup)
        if not args.smoke:
            import cv2
            source = int(args.video) if args.video.isdecimal() else args.video
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                raise RuntimeError("Cannot open video source")
        buffer = deque(maxlen=frames)
        sequence = 0
        while sequence < args.steps:
            if args.smoke:
                clip = np.full((frames, 128, 128, 3), 255 * (sequence % 2), dtype=np.uint8)
                captured = time.monotonic()
            else:
                ok, frame = capture.read()
                captured = time.monotonic()
                if not ok:
                    break
                buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if len(buffer) < frames:
                    continue
                clip = np.stack(buffer)
            start = time.perf_counter()
            result = pipeline.evaluate(Observation(sequence, captured, clip))
            print(json.dumps(dict(sequence=sequence, decision=result.decision,
                                  reason=result.reason,
                                  latency_ms=round((time.perf_counter() - start) * 1000, 2),
                                  device=str(motion.device), smoke=args.smoke,
                                  motor_commands_sent=0)), flush=True)
            sequence += 1
    finally:
        pipeline.close()
        if capture is not None:
            capture.release()


if __name__ == "__main__":
    main()
