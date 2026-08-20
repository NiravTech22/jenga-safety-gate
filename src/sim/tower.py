"""Physics-lite Jenga tower model.

Deliberately not a rigid-body simulator. It models the one thing the
experiment actually turns on: whether removing a specific block leaves the
structure with enough static margin to stay up. Everything is deterministic
given a seed, including counterfactuals — `preview_extract(layer, slot)`
returns exactly the outcome `extract(layer, slot)` would produce, which is
what makes an honest gate false-positive rate computable (Step 8).

Units are SI (metres, degrees). Geometry matches a standard Jenga set.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

# Standard Jenga block: 75 x 25 x 15 mm.
BLOCK_L_M = 0.075
BLOCK_W_M = 0.025
BLOCK_H_M = 0.015
SLOTS_PER_LAYER = 3
DEFAULT_LAYERS = 12          # shorter than a full 18-layer tower: faster episodes, same physics


class ExtractionOutcome(str, Enum):
    """Outcome of one attempted block extraction.

    Maps 1:1 onto src.eval.metrics.TrialResult for trial-level scoring.
    """
    SUCCESS = "success"                  # block out, tower steady
    NEAR_MISS = "near_miss"              # visible wobble / partial shift, no collapse
    TOPPLE = "topple"                    # catastrophic: structure comes down
    ILLEGAL = "illegal"                  # slot already empty, or top-layer rule violation


@dataclass
class TowerSimConfig:
    num_layers: int = DEFAULT_LAYERS
    seed: int = 0
    # Per-block friction ("tightness"): higher means the block resists
    # extraction and yanks the structure more when it finally moves.
    tightness_mean: float = 0.35
    tightness_std: float = 0.15
    initial_lean_deg: float = 0.0
    # Blocks pre-removed at reset, as (layer, slot) pairs — used to build
    # near-collapse and off-center starting states.
    pre_removed: tuple = ()
    # How much a single extraction perturbs the lean, in degrees.
    lean_perturbation_deg: float = 0.35


@dataclass
class TowerState:
    """Ground-truth state. Perception (Step 2) must reconstruct this from
    images alone; nothing downstream is allowed to read it directly except
    the sim's own scoring and the labelling of validation frames."""
    occupancy: np.ndarray                # (num_layers, 3) bool
    lean_deg: float
    tightness: np.ndarray                # (num_layers, 3) float
    toppled: bool = False
    steps_taken: int = 0

    def copy(self) -> "TowerState":
        return TowerState(
            occupancy=self.occupancy.copy(),
            lean_deg=self.lean_deg,
            tightness=self.tightness.copy(),
            toppled=self.toppled,
            steps_taken=self.steps_taken,
        )


def _slot_offsets_m() -> np.ndarray:
    """Lateral centre offset of each slot within a layer."""
    return np.array([-BLOCK_W_M, 0.0, BLOCK_W_M])


def top_complete_layer(occupancy: np.ndarray) -> int:
    """Index of the highest fully-occupied layer.

    The real-game rule is that you may not pull from the top complete layer
    (it is holding the tower's crown up). Returns -1 if none is complete.
    """
    complete = np.where(occupancy.all(axis=1))[0]
    return int(complete[-1]) if complete.size else -1


def _layer_axis(layer: int) -> int:
    """0 = blocks run along x (slots vary in y), 1 = the alternating layer."""
    return layer % 2


def centre_of_mass(occupancy: np.ndarray, lean_deg: float) -> np.ndarray:
    """(x, y) centre of mass in metres, including the shear from lean.

    Lean tilts the whole stack about the base along +x, so a block at
    height z is displaced by z * tan(lean).
    """
    offsets = _slot_offsets_m()
    tan_lean = np.tan(np.radians(lean_deg))
    total = occupancy.sum()
    if total == 0:
        return np.zeros(2)

    acc = np.zeros(2)
    for layer in range(occupancy.shape[0]):
        z = (layer + 0.5) * BLOCK_H_M
        axis = _layer_axis(layer)
        for slot in range(SLOTS_PER_LAYER):
            if not occupancy[layer, slot]:
                continue
            pos = np.zeros(2)
            pos[1 - axis] = offsets[slot]     # blocks are offset across their layer's axis
            pos[0] += z * tan_lean            # lean shear, applied in +x
            acc += pos
    return acc / total


def support_margin(occupancy: np.ndarray, lean_deg: float) -> float:
    """Static stability margin in metres: how far the centre of mass is
    from the edge of the supporting footprint. Negative means the tower is
    statically unstable and comes down.

    The footprint is the lowest layer that still has at least one block —
    a hollowed-out bottom layer is exactly what makes a tower fragile.
    """
    if occupancy.sum() == 0:
        return -1.0

    occupied_layers = np.where(occupancy.any(axis=1))[0]
    base_layer = int(occupied_layers[0])
    base = occupancy[base_layer]
    offsets = _slot_offsets_m()
    axis = _layer_axis(base_layer)

    present = offsets[base]
    across_min, across_max = present.min() - BLOCK_W_M / 2, present.max() + BLOCK_W_M / 2
    along_min, along_max = -BLOCK_L_M / 2, BLOCK_L_M / 2

    if axis == 0:                     # blocks along x, offsets in y
        x_min, x_max, y_min, y_max = along_min, along_max, across_min, across_max
    else:
        x_min, x_max, y_min, y_max = across_min, across_max, along_min, along_max

    com = centre_of_mass(occupancy, lean_deg)
    return float(min(com[0] - x_min, x_max - com[0], com[1] - y_min, y_max - com[1]))


def structural_penalty(occupancy: np.ndarray) -> float:
    """Extra fragility that a pure centre-of-mass check misses.

    A layer resting on a single off-centre block, or a layer with a
    hollowed-out neighbour directly beneath it, is far weaker than its
    centre of mass suggests.
    """
    penalty = 0.0
    n_layers = occupancy.shape[0]
    for layer in range(n_layers):
        count = int(occupancy[layer].sum())
        if count == 0:
            continue
        if count == 1:
            # A lone centre block is a legal (if precarious) pivot; a lone
            # edge block is much worse.
            penalty += 0.004 if occupancy[layer, 1] else 0.010
        elif count == 2 and not occupancy[layer, 1]:
            penalty += 0.003          # two edges, nothing in the middle
        if layer + 1 < n_layers and occupancy[layer + 1].sum() > count:
            penalty += 0.002          # top-heavy relative to its own support
    return penalty


def stability_score(state: TowerState) -> float:
    """Single scalar the whole sim is scored on. Positive = standing with
    margin, near zero = one bad pull from collapse, negative = down."""
    return support_margin(state.occupancy, state.lean_deg) - structural_penalty(state.occupancy)


class TowerSim:
    """Deterministic tower with a counterfactual-safe extraction model."""

    def __init__(self, config: Optional[TowerSimConfig] = None):
        self.config = config or TowerSimConfig()
        self.state = self._reset()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _reset(self) -> TowerState:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        occupancy = np.ones((cfg.num_layers, SLOTS_PER_LAYER), dtype=bool)
        tightness = np.clip(
            rng.normal(cfg.tightness_mean, cfg.tightness_std, occupancy.shape), 0.02, 0.95
        )
        for layer, slot in cfg.pre_removed:
            occupancy[layer, slot] = False
        return TowerState(occupancy=occupancy, lean_deg=cfg.initial_lean_deg, tightness=tightness)

    def reset(self) -> TowerState:
        self.state = self._reset()
        return self.state

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def legal_actions(self) -> list[tuple[int, int]]:
        """Every (layer, slot) that is occupied and below the top complete
        layer — the real game's rule, and also what keeps the action space
        from containing trivially-illegal moves."""
        if self.state.toppled:
            return []
        top = top_complete_layer(self.state.occupancy)
        return [
            (layer, slot)
            for layer in range(self.state.occupancy.shape[0])
            for slot in range(SLOTS_PER_LAYER)
            if self.state.occupancy[layer, slot] and layer < top
        ]

    def stability(self) -> float:
        return stability_score(self.state)

    def _action_rng(self, layer: int, slot: int) -> np.random.Generator:
        """Per-(state, action) RNG stream.

        Keyed by the *current* occupancy and step count, so previewing an
        action and then taking it draw the same numbers, and previewing
        action A does not perturb the outcome of action B. This is what
        makes the counterfactual ground truth in Step 8 trustworthy.
        """
        key = (
            f"{self.config.seed}|{self.state.steps_taken}|{layer},{slot}|"
            f"{self.state.occupancy.astype(np.uint8).tobytes().hex()}"
        )
        digest = hashlib.sha256(key.encode()).digest()[:8]
        return np.random.default_rng(int.from_bytes(digest, "big"))

    def _outcome_probabilities(self, layer: int, slot: int) -> tuple[float, float, float, float]:
        """(p_topple, p_near_miss, margin_after, lean_after) for a candidate
        extraction, without mutating state."""
        occ = self.state.occupancy.copy()
        occ[layer, slot] = False

        tight = float(self.state.tightness[layer, slot])
        # A tight block drags the structure sideways as it comes free; the
        # higher up it is, the more leverage that drag has.
        height_frac = (layer + 0.5) * BLOCK_H_M / (self.config.num_layers * BLOCK_H_M)
        lean_after = self.state.lean_deg + self.config.lean_perturbation_deg * tight * (
            0.5 + height_frac
        )

        margin_after = support_margin(occ, lean_after) - structural_penalty(occ)

        # Map static margin to collapse probability. ~4 mm of margin is the
        # knee: below it the tower is one perturbation from going over.
        knee = 0.004
        if margin_after <= 0:
            p_topple = 0.97
        else:
            p_topple = float(np.clip(np.exp(-margin_after / knee) * (0.55 + 0.45 * tight), 0.0, 0.97))
        p_near_miss = float(np.clip((1.0 - p_topple) * (0.25 + 0.55 * tight) * np.exp(-margin_after / 0.010), 0.0, 1.0 - p_topple))
        return p_topple, p_near_miss, margin_after, lean_after

    def risk(self, layer: int, slot: int) -> float:
        """Ground-truth probability that this extraction topples the tower.
        Used to label validation frames and to score the gate — never
        visible to the policy or the detector."""
        return self._outcome_probabilities(layer, slot)[0]

    def preview_extract(self, layer: int, slot: int) -> ExtractionOutcome:
        """What `extract(layer, slot)` would return, with no side effects."""
        return self._resolve(layer, slot, commit=False)

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #
    def extract(self, layer: int, slot: int) -> ExtractionOutcome:
        return self._resolve(layer, slot, commit=True)

    def _resolve(self, layer: int, slot: int, commit: bool) -> ExtractionOutcome:
        if self.state.toppled:
            return ExtractionOutcome.ILLEGAL
        if not (0 <= layer < self.state.occupancy.shape[0] and 0 <= slot < SLOTS_PER_LAYER):
            return ExtractionOutcome.ILLEGAL
        if not self.state.occupancy[layer, slot]:
            return ExtractionOutcome.ILLEGAL
        if layer >= top_complete_layer(self.state.occupancy):
            return ExtractionOutcome.ILLEGAL

        p_topple, p_near_miss, _, lean_after = self._outcome_probabilities(layer, slot)
        draw = float(self._action_rng(layer, slot).random())

        if draw < p_topple:
            outcome = ExtractionOutcome.TOPPLE
        elif draw < p_topple + p_near_miss:
            outcome = ExtractionOutcome.NEAR_MISS
        else:
            outcome = ExtractionOutcome.SUCCESS

        if commit:
            self.state.occupancy[layer, slot] = False
            self.state.lean_deg = lean_after
            self.state.steps_taken += 1
            if outcome is ExtractionOutcome.TOPPLE:
                self.state.toppled = True
            elif outcome is ExtractionOutcome.NEAR_MISS:
                # A wobble leaves the tower measurably worse off than a
                # clean pull — that residual lean is what the next
                # extraction has to live with.
                self.state.lean_deg += self.config.lean_perturbation_deg
        return outcome
