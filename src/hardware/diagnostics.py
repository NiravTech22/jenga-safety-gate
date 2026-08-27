"""Measure what is actually wrong with a teleoperated arm.

"The teleop is not quite accurate" is not one fault. At least five things
produce that sensation and they are indistinguishable from inside a headset:

  offset        a constant bias; wrong home or zero
  scale         travel error that grows with distance from zero; a calibration
                table whose span does not match the real mechanical range
  lost motion   the joint stops short and stays short: backlash, servo
                deadband, or both. From joint position alone these two are
                not separable, because both make the joint settle short of
                where it was sent. Separating them needs torque or current
                sensing, or an encoder on the output side rather than the
                motor. By hand the discriminator is that backlash appears
                only after a reversal, while a deadband also swallows small
                commands issued from rest in the same direction.
  latency       command transport delay, which an operator closing the loop by
                eye experiences as overshoot and hunting

Only offset is genuinely correctable in the VR layer. Scale needs the
calibration table fixed. Backlash needs hardware attention or a controller
that compensates for it. So guessing wrong here means tuning the wrong layer
for a long time.

Each routine drives the arm through a specific probe motion and reports one
number. Everything works against ArmInterface, so the same code characterises
the simulated arm in tests and the real one on the bench.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.hardware.arm import ArmInterface, N_JOINTS
from src.hardware.kinematics import JOINT_NAMES

SETTLE_S = 1.0


def _hold(arm: ArmInterface, target: np.ndarray, settle_s: float = SETTLE_S) -> np.ndarray:
    arm.command_joints(target)
    arm.wait(settle_s)
    return np.asarray(arm.read_joints(), float)


def _vector(joint: int, value: float, base: Optional[np.ndarray] = None) -> np.ndarray:
    v = np.zeros(N_JOINTS) if base is None else np.array(base, float)
    v[joint] = value
    return v


@dataclass
class JointDiagnostic:
    joint: str
    offset_deg: float
    scale_error_pct: float
    lost_motion_deg: float
    repeatability_deg: float
    residual_deg: float          # fit residual; high means the errors are not linear

    def worst_fault(self) -> str:
        """Which single fault dominates for this joint."""
        candidates = {
            "lost motion": abs(self.lost_motion_deg) / 0.5,
            "scale": abs(self.scale_error_pct) / 2.0,
            "offset": abs(self.offset_deg) / 0.5,
            "repeatability": abs(self.repeatability_deg) / 0.3,
        }
        ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
        name, score = ranked[0]
        return name if score > 1.0 else "within tolerance"


@dataclass
class TeleopReport:
    joints: list = field(default_factory=list)
    latency_ms: float = 0.0
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"latency_ms": self.latency_ms,
                "joints": [asdict(j) for j in self.joints],
                "notes": self.notes}

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def render(self) -> str:
        head = (f"{'joint':15s} {'offset':>8s} {'scale':>8s} {'lost motion':>12s} "
                f"{'repeat':>8s}  dominant")
        lines = [head, "-" * len(head)]
        for j in self.joints:
            lines.append(f"{j.joint:15s} {j.offset_deg:+7.2f}° {j.scale_error_pct:+7.2f}% "
                         f"{j.lost_motion_deg:11.2f}° {j.repeatability_deg:7.2f}°  "
                         f"{j.worst_fault()}")
        lines.append("-" * len(head))
        lines.append(f"command latency: {self.latency_ms:.0f} ms")
        for n in self.notes:
            lines.append(f"note: {n}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Individual measurements
# --------------------------------------------------------------------------- #
def measure_calibration(arm: ArmInterface, joint: int, span_rad: float = 0.6,
                        points: int = 7, settle_s: float = SETTLE_S) -> tuple:
    """Fit achieved = scale * commanded + offset over a sweep.

    Every point is approached from below, from a common start well outside the
    sweep. Mixing approach directions would fold backlash into the fit and
    show up as inflated residual rather than as the hysteresis it is.
    """
    targets = np.linspace(-span_rad, span_rad, points)

    def sweep(approach_from: float):
        achieved = []
        for t in targets:
            _hold(arm, _vector(joint, approach_from), settle_s * 0.6)
            achieved.append(_hold(arm, _vector(joint, t), settle_s)[joint])
        return np.array(achieved)

    # Two sweeps, each approached consistently from one side. A single
    # upward-only sweep sits half a backlash band low at every point, and a
    # straight-line fit reports that as an offset, which sends you looking for
    # a calibration bug that is not there. Averaging the two cancels it.
    up = sweep(-span_rad - 0.25)
    down = sweep(span_rad + 0.25)
    achieved = (up + down) / 2.0

    A = np.vstack([targets, np.ones_like(targets)]).T
    (scale, offset), *_ = np.linalg.lstsq(A, achieved, rcond=None)
    residual = float(np.sqrt(np.mean((A @ [scale, offset] - achieved) ** 2)))
    return float(scale), float(offset), residual


def measure_lost_motion(arm: ArmInterface, joint: int, midpoint: float = 0.0,
                        excursion: float = 0.4, settle_s: float = SETTLE_S) -> float:
    """Approach one point from both sides; the gap between where it settles is
    the lost motion, from backlash and servo deadband together."""
    _hold(arm, _vector(joint, midpoint - excursion), settle_s)
    from_below = _hold(arm, _vector(joint, midpoint), settle_s)[joint]
    _hold(arm, _vector(joint, midpoint + excursion), settle_s)
    from_above = _hold(arm, _vector(joint, midpoint), settle_s)[joint]
    return float(abs(from_above - from_below))


def measure_repeatability(arm: ArmInterface, joint: int, home: float = 0.0,
                          trials: int = 6, settle_s: float = SETTLE_S,
                          seed: int = 0) -> float:
    """Spread of settled position returning to the same target from random
    starts. Approach direction is randomised on purpose: this number is meant
    to include hysteresis, because that is what the operator experiences."""
    rng = np.random.default_rng(seed)
    settled = []
    for _ in range(trials):
        _hold(arm, _vector(joint, float(rng.uniform(-0.5, 0.5))), settle_s * 0.8)
        settled.append(_hold(arm, _vector(joint, home), settle_s)[joint])
    return float(np.std(settled))


def measure_latency(arm: ArmInterface, joint: int = 0, amplitude: float = 0.35,
                    poll_s: float = 0.004, timeout_s: float = 1.5) -> float:
    """Time from issuing a command to the first detectable motion.

    Includes transport delay plus however long the joint takes to break its own
    stiction, so treat it as an upper bound on transport delay rather than a
    pure measurement of it.
    """
    _hold(arm, _vector(joint, 0.0))
    start_pos = np.asarray(arm.read_joints(), float)[joint]
    t0 = arm.now()
    arm.command_joints(_vector(joint, amplitude))
    threshold = abs(amplitude) * 0.02
    while arm.now() - t0 < timeout_s:
        arm.wait(poll_s)
        if abs(np.asarray(arm.read_joints(), float)[joint] - start_pos) > threshold:
            return float(arm.now() - t0)
    return float(timeout_s)


# --------------------------------------------------------------------------- #
# Full sweep
# --------------------------------------------------------------------------- #
def run_diagnostics(arm: ArmInterface, joints: Optional[list] = None,
                    settle_s: float = SETTLE_S) -> TeleopReport:
    indices = list(range(N_JOINTS)) if joints is None else joints
    report = TeleopReport()

    for i in indices:
        scale, offset, residual = measure_calibration(arm, i, settle_s=settle_s)
        report.joints.append(JointDiagnostic(
            joint=JOINT_NAMES[i],
            offset_deg=float(np.degrees(offset)),
            scale_error_pct=float((scale - 1.0) * 100.0),
            lost_motion_deg=float(np.degrees(measure_lost_motion(arm, i, settle_s=settle_s))),
            repeatability_deg=float(np.degrees(measure_repeatability(arm, i, settle_s=settle_s))),
            residual_deg=float(np.degrees(residual)),
        ))

    report.latency_ms = float(measure_latency(arm) * 1000.0)

    worst = [j for j in report.joints if j.worst_fault() != "within tolerance"]
    if worst:
        report.notes.append(
            "Joints outside tolerance: " + ", ".join(f"{j.joint} ({j.worst_fault()})" for j in worst))
    if report.latency_ms > 120:
        report.notes.append(
            f"{report.latency_ms:.0f} ms of command latency is enough for an operator to "
            "overshoot and hunt. Fix this before judging any of the position errors, "
            "because chasing a lagging arm produces position errors of its own.")
    return report
