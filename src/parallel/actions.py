"""Dry-run action checks; this module never connects to a motor bus.

Workspace clearance must come from a separately validated monitor. Unknown
clearance blocks. Model completion alone is not an authorization to move.
"""
from dataclasses import dataclass
import time

import numpy as np

from src.hardware.kinematics import JOINT_NAMES


@dataclass(frozen=True)
class ActionProposal:
    sequence: int
    target_rad: tuple


class ActionChecks:
    def __init__(self, limits, max_step_rad, max_age_s=0.5):
        self.limits = np.asarray([limits[n] for n in JOINT_NAMES], dtype=float)
        self.max_step = np.broadcast_to(np.asarray(max_step_rad, float), (6,)).copy()
        self.max_age = float(max_age_s)
        if (self.limits.shape != (6, 2) or not np.isfinite(self.limits).all()
                or np.any(self.limits[:, 0] >= self.limits[:, 1])
                or not np.isfinite(self.max_step).all() or np.any(self.max_step <= 0)
                or not np.isfinite(self.max_age) or self.max_age <= 0):
            raise ValueError("Invalid action limits")
        self.last_sequence = -1

    def evaluate(self, proposal, observation, result, current_rad, *,
                 robot_turn=False, workspace_clear=None, emergency_stop=True):
        """Single-controller-thread API returning (allowed, reason).

All inputs must describe the same observation. Clearance and turn state are
external inputs, not inferred by AccelSight. Call immediately before dispatch;
this dry-run checker supplies no trajectory or collision guarantee.
"""
        if emergency_stop is not False:
            return False, "emergency stop or unknown stop state"
        if robot_turn is not True or workspace_clear is not True:
            return False, "human turn or workspace clearance unknown/occupied"
        if proposal.sequence != observation.sequence or result.sequence != observation.sequence:
            return False, "action/observation mismatch"
        if proposal.sequence <= self.last_sequence:
            return False, "action replay"
        age = time.monotonic() - observation.captured_at
        if not np.isfinite(age) or age < 0 or age >= self.max_age:
            return False, "observation expired"
        if result.decision != "execute":
            return False, "parallel inference blocked"
        target, current = np.asarray(proposal.target_rad, float), np.asarray(current_rad, float)
        if (target.shape != (6,) or current.shape != (6,)
                or not np.isfinite(target).all() or not np.isfinite(current).all()):
            return False, "invalid six-joint state/action"
        if (np.any(target < self.limits[:, 0]) or np.any(target > self.limits[:, 1])
                or np.any(current < self.limits[:, 0]) or np.any(current > self.limits[:, 1])):
            return False, "joint limit"
        if np.any(np.abs(target - current) > self.max_step):
            return False, "joint step limit"
        self.last_sequence = proposal.sequence
        return True, "dry-run action checks passed"
