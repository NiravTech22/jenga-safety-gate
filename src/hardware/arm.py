"""Arm interface, plus a simulated arm that can be given specific defects.

Two reasons this exists.

First, there is currently no seam between the VR teleop loop and the recorder,
so scripts/03_record_episode.py can only dry-run. Anything satisfying
ArmInterface can be recorded from.

Second, and more useful right now: teleop that "feels inaccurate" can be
inaccurate in at least five different ways that are indistinguishable from
inside a headset. A constant offset, a span error, backlash, latency and a
deadband all read as "it does not go quite where I put it". They have
different fixes, and three of them are not fixable in the VR layer at all.
SimulatedArm can be given each defect at a known magnitude, which is how the
diagnostics in src/hardware/diagnostics.py are themselves tested: inject 4
degrees of backlash, confirm the measurement reports 4 degrees.

Time is part of the interface (now/wait) rather than called through the time
module directly, so the simulated arm can run on a virtual clock. Tests then
execute instantly and deterministically instead of sleeping through a
multi-second probe sequence.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

from src.hardware.kinematics import JOINT_NAMES

N_JOINTS = len(JOINT_NAMES)


class ArmInterface(Protocol):
    """What the recorder and the diagnostics need from an arm."""

    def read_joints(self) -> np.ndarray: ...
    def command_joints(self, target: np.ndarray) -> None: ...
    def now(self) -> float: ...
    def wait(self, seconds: float) -> None: ...


@dataclass
class ArmDefects:
    """Error sources, each independently switchable.

    Units are radians and seconds. Defaults are a clean arm; give a field a
    value to simulate that specific fault.
    """
    # Constant per-joint bias. This is what a wrong home/zero produces, and it
    # is the one a VR-side offset can genuinely cancel.
    offset_rad: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS))
    # Multiplicative error on travel. A calibration table whose min/max span
    # does not match the real mechanical range gives exactly this, and it
    # cannot be cancelled by an offset: error grows with distance from zero.
    scale: np.ndarray = field(default_factory=lambda: np.ones(N_JOINTS))
    # Lost motion on direction reversal. Mechanical, not fixable in software.
    backlash_rad: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS))
    # Commands smaller than this produce no motion at all.
    deadband_rad: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINTS))
    # Command transport delay. Reads as overshoot when the operator closes the
    # loop by eye.
    latency_s: float = 0.0
    # First-order response time constant.
    tau_s: float = 0.06
    # Encoder noise, one sigma.
    noise_rad: float = 0.0
    seed: int = 0


class SimulatedArm:
    """First-order joint tracking with optional defects, on a virtual clock."""

    def __init__(self, defects: Optional[ArmDefects] = None,
                 initial: Optional[np.ndarray] = None, dt: float = 0.002):
        self.defects = defects or ArmDefects()
        self.dt = dt
        self._t = 0.0
        self._pos = np.zeros(N_JOINTS) if initial is None else np.array(initial, float)
        self._commanded = self._pos.copy()
        self._pending: list = []          # (apply_at, target)
        self._target = self._pos.copy()
        self._rng = np.random.default_rng(self.defects.seed)

    # -- interface ---------------------------------------------------------
    def now(self) -> float:
        return self._t

    def read_joints(self) -> np.ndarray:
        noise = (self._rng.normal(0, self.defects.noise_rad, N_JOINTS)
                 if self.defects.noise_rad else 0.0)
        return self._pos + noise

    def command_joints(self, target) -> None:
        target = np.asarray(target, float).reshape(N_JOINTS)
        self._commanded = target
        self._pending.append((self._t + self.defects.latency_s, target.copy()))

    def wait(self, seconds: float) -> None:
        steps = max(int(round(seconds / self.dt)), 1)
        for _ in range(steps):
            self._advance(self.dt)

    # -- internals ---------------------------------------------------------
    def _advance(self, dt: float) -> None:
        self._t += dt
        while self._pending and self._pending[0][0] <= self._t:
            _, target = self._pending.pop(0)
            self._target = target

        d = self.defects
        # Where the drive side is asking the joint to be, after the calibration
        # errors the servo layer applies.
        ideal = d.scale * self._target + d.offset_rad

        # Backlash as a dead zone the output sits inside: the joint only moves
        # once the drive has taken up the slack, and it then settles half a
        # band short. Approaching the same point from opposite directions
        # therefore settles a full band apart, which is what makes it
        # measurable and what distinguishes it from a fixed offset. Modelling
        # it as a debt that gets paid off would leave no steady-state error at
        # all, and backlash you can settle out of is not backlash.
        half = d.backlash_rad / 2.0
        goal = np.clip(self._pos, ideal - half, ideal + half)

        error = goal - self._pos
        moving = np.abs(error) > d.deadband_rad
        step = (dt / max(d.tau_s, 1e-6)) * error
        self._pos = self._pos + np.where(moving, step, 0.0)

    # -- convenience -------------------------------------------------------
    def move_to(self, target, settle_s: float = 1.2) -> np.ndarray:
        self.command_joints(target)
        self.wait(settle_s)
        return self.read_joints()


class RealTimeClockMixin:
    """Wall-clock timing for any driver talking to physical hardware."""

    def now(self) -> float:
        return time.time()

    def wait(self, seconds: float) -> None:
        time.sleep(seconds)


class VRTeleopArm(RealTimeClockMixin):
    """Adapter seam for the existing VR teleop server.

    Deliberately unimplemented. The server at vr_control.server_url is outside
    this repository and its API is not documented here, so guessing at it would
    produce code that looks finished and fails on contact. Fill in the two
    methods below against whatever the teleop stack already exposes: a joint
    state read, and a joint position command in radians in JOINT_NAMES order.

    Everything downstream, the recorder and the diagnostics both, works
    against ArmInterface and needs no further change once these return real
    numbers.
    """

    def __init__(self, server_url: str, timeout_s: float = 2.0):
        self.server_url = server_url
        self.timeout_s = timeout_s

    def read_joints(self) -> np.ndarray:
        raise NotImplementedError(
            "Return the current joint positions as radians in JOINT_NAMES "
            f"order: {JOINT_NAMES}. Source them from the teleop server at "
            f"{self.server_url} or directly from the servo bus."
        )

    def command_joints(self, target) -> None:
        raise NotImplementedError(
            "Send a joint position command, radians in JOINT_NAMES order. "
            "This must be the same path the VR loop drives, or the diagnostics "
            "will characterise a path the operator never uses."
        )
