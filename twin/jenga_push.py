#!/usr/bin/env python3
"""Phase 2: the SO-101 twin pushes Jenga blocks, gated by a MuJoCo world model.

Two simulations run side by side:

  TRUTH   the "real" world: the twin's custom-claw arm and a tower whose blocks
          have hidden per-block thickness and friction (from the episode seed).
  MODEL   the gate's world model: the same tower built from NOMINAL block sizes,
          no arm, a kinematic pusher instead. It is given the block poses the
          truth world reports (plus noise) and samples friction it cannot see.

Before each push the gate runs the push N times in MODEL and estimates
P(fall). Above the threshold the push is blocked and another block is tried.
Counterfactuals (does this push topple the TRUTH tower?) are scored on a copy
of the truth state, so every gate decision gets a ground-truth label.

Usage (from the repository root):
  python twin/jenga_push.py                      # viewer: one gated episode
  python twin/jenga_push.py --headless --episodes 1
  python twin/jenga_push.py --compare --episodes 20     # gated vs ungated table
  python twin/jenga_push.py --record push.mp4           # offscreen video

Pushable blocks: with the claw horizontal along +x, only blocks lying along x
(even layers) can be pushed out, and only where one fingertip can sit on the
block's end while the other finger stays outside the tower (see DIGITAL_TWIN.md).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

TWIN = Path(__file__).resolve().parent
sys.path.insert(0, str(TWIN))
import so101_pickplace as PP  # noqa: E402  (IK, joint helpers, scene source)

MODELS = TWIN / "models"
LOG_DIR = TWIN.parent / "logs/twin"

# ------------------------------------------------------------------ tower geometry
# Same block convention as src/sim/mujoco_jenga.py (75 x 25 x 15 mm, 0.2 mm layer gap).
BLOCK_HALF = np.array([.0375, .0125, .0075])
LAYER_PITCH = .0152
SLOT_PITCH = .0255
LAYERS = 15
TOWER_XY = np.array([.335, -.045])       # chosen so two of three slots are reachable (DIGITAL_TWIN.md)
PUSH_LAYERS = tuple(range(4, LAYERS - 2, 2))   # even layers (lying along x); >=4 keeps the claw off the table
PUSH_DEPTH = .012                        # fingertip travel into the tower: the hook only fits ~12 mm
REFLEX_STOP_MM = 3.0                     # stop the test push once anything else moves this much
PULL_TRAVEL, PULL_SECS, PULL_MAX_N = .085, 1.2, 3.0
FAIL_DROP, FAIL_DRIFT = .03, .04          # same failure rule as src/sim/mujoco_jenga.py
TIP_LEN = .012                            # fingertip length used as the push contact
SETTLE_S = .15
POSE_NOISE_M = .0005                      # tower-pose error the gate is given (jenga_vision accuracy)
GATE_SAMPLES, GATE_THRESHOLD, GATE_ESS_MIN = 16, .25, 2.5   # locked on calibration towers 1-2, before any comparison


def block_name(layer, slot):
    return f"block_{layer}_{slot}"


def block_index(layer, slot):
    return layer * 3 + slot


def nominal_block_xy(layer, slot):
    off = (slot - 1) * SLOT_PITCH
    x, y = TOWER_XY
    return (np.array([x, y + off]), 0.0) if layer % 2 == 0 else (np.array([x + off, y]), math.pi / 2)


@dataclass
class TowerSpec:
    """Hidden properties of one physical tower: the gate never sees these."""
    seed: int
    half_thickness: np.ndarray   # per block, metres (+/-0.3 mm => some blocks carry no load)
    friction: np.ndarray         # per block sliding friction
    jitter: np.ndarray           # per block xy placement error, metres

    @staticmethod
    def sample(seed, layers=LAYERS):
        rng = np.random.default_rng(seed)
        n = layers * 3
        return TowerSpec(seed, BLOCK_HALF[2] + rng.uniform(-.0003, .0003, n),
                         rng.uniform(.30, .70, n), rng.uniform(-.0008, .0008, (n, 2)))


def _add_tower(world, spec: TowerSpec | None, layers=LAYERS, frozen_below=0):
    """frozen_below > 0 makes those bottom layers static (world-model speed-up: they never move)."""
    z = 0.0
    for layer in range(layers):
        hs = [spec.half_thickness[block_index(layer, s)] if spec else BLOCK_HALF[2] for s in range(3)]
        for s in range(3):
            i = block_index(layer, s)
            xy, yaw = nominal_block_xy(layer, s)
            if spec is not None:
                xy = xy + spec.jitter[i]
            b = ET.SubElement(world, "body", name=block_name(layer, s),
                              pos=PP.fmt([xy[0], xy[1], z + hs[s] + .0001]),
                              quat=PP.fmt([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]))
            if layer >= frozen_below:
                ET.SubElement(b, "freejoint", name=block_name(layer, s) + "_joint")
            mu = spec.friction[i] if spec is not None else .5
            shade = .80 - .08 * (layer % 2)
            ET.SubElement(b, "geom", name=block_name(layer, s) + "_geom", type="box",
                          size=PP.fmt([BLOCK_HALF[0], BLOCK_HALF[1], hs[s]]), mass="0.012",
                          friction=f"{mu:.4f} .005 .0001", condim="4", solref=".006 1",
                          rgba=f"{shade:.2f} {shade * .72:.2f} {shade * .43:.2f} 1")
        z += 2 * max(hs) + .0002


def _physics_options(root):
    opt = root.find("option")
    if opt is None:
        opt = ET.SubElement(root, "option")
    # Elliptic friction cones + impratio keep a 45-block stack from creeping (measured: 19 mm -> 2 mm / 4 s).
    for k, v in dict(timestep="0.002", integrator="implicitfast", iterations="60",
                     cone="elliptic", impratio="10").items():
        opt.set(k, v)


# ------------------------------------------------------------------ fingertip contact geometry
def _tip_boxes():
    """Box matching the last TIP_LEN of each finger mesh, in that finger's body frame.

    The claw's fingers are hooks: about 12 mm of tip, then the finger rises
    into the layer above. The pick-and-place pads are taller than the real tip,
    so for pushing they are disabled and these boxes are the contact surface.
    """
    m = mujoco.MjModel.from_xml_path(str(PP.SOURCE))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    pads = [m.geom(n).id for n in PP.PAD_NAMES]
    axis_w = .5 * (d.geom_xpos[pads[0]] + d.geom_xpos[pads[1]]) - d.xpos[m.body("custom_gripper_base").id]
    axis_w /= np.linalg.norm(axis_w)
    out = {}
    for side, joint in (("left", "gripper"), ("right", "gripper_mimic")):
        b = m.body(f"finger_{side}").id
        Rb = d.xmat[b].reshape(3, 3)
        a = Rb.T @ axis_w
        jaw = m.jnt_axis[m.joint(joint).id].astype(float)
        jaw -= (jaw @ a) * a; jaw /= np.linalg.norm(jaw)
        up = np.cross(a, jaw)
        V = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                k = m.geom_dataid[g]
                mv = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
                Rg = np.zeros(9); mujoco.mju_quat2Mat(Rg, m.geom_quat[g])
                V.append(m.geom_pos[g] + mv @ Rg.reshape(3, 3).T)
        V = np.vstack(V)
        P = V @ np.column_stack([a, jaw, up])
        T = P[P[:, 0] > P[:, 0].max() - TIP_LEN]
        lo, hi = T.min(0), T.max(0)
        R = np.column_stack([a, jaw, up])
        q = np.zeros(4); mujoco.mju_mat2Quat(q, R.ravel())
        out[side] = dict(pos=R @ ((lo + hi) / 2), quat=q, half=(hi - lo) / 2)
    return out


def claw_variant_xml() -> Path:
    """Copy of the claw model with fingertip boxes added and the pick-and-place pads disabled."""
    include = ET.parse(PP.SOURCE).getroot().find("include").get("file")
    src = MODELS / include
    tree = ET.parse(src)
    tips = _tip_boxes()
    for side, t in tips.items():
        body = tree.getroot().find(f".//body[@name='finger_{side}']")
        pad = body.find(f"geom[@name='pad_finger_{side}']")
        pad.set("contype", "0"); pad.set("conaffinity", "0"); pad.set("rgba", "0.2 0.8 0.2 0.15")
        ET.SubElement(body, "geom", name=f"tip_finger_{side}", type="box", pos=PP.fmt(t["pos"]),
                      quat=PP.fmt(t["quat"]), size=PP.fmt(t["half"]), contype="1", conaffinity="1",
                      group="0", friction=".6 .005 .0001", condim="4", solref=".006 1", rgba="1 .45 .1 1")
    out = MODELS / "so101_custom_claw_jenga.generated.xml"
    ET.indent(tree, space="  "); tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def truth_xml(spec: TowerSpec) -> Path:
    tree = ET.parse(PP.SOURCE)
    root = tree.getroot()
    root.find("include").set("file", claw_variant_xml().name)
    world = root.find("worldbody")
    for node in list(world):                      # pick-and-place leftovers in the source scene
        name = node.get("name", "")
        if name in ("demo_table", "grasp_cube") or name.startswith("workcell_leg_"):
            world.remove(node)
    _physics_options(root)
    vis = root.find("visual") if root.find("visual") is not None else ET.SubElement(root, "visual")
    g = vis.find("global") if vis.find("global") is not None else ET.SubElement(vis, "global")
    g.set("offwidth", "1280"); g.set("offheight", "960")
    _add_tower(root.find("worldbody"), spec)
    out = MODELS / "scene_jenga.generated.xml"
    ET.indent(tree, space="  "); tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


def world_model_xml(frozen_below=0) -> str:
    """The gate's world model: nominal tower, a kinematic fingertip, no arm."""
    root = ET.Element("mujoco", model="jenga_world_model")
    ET.SubElement(root, "compiler", angle="radian")
    _physics_options(root)
    w = ET.SubElement(root, "worldbody")
    ET.SubElement(w, "geom", name="floor", type="plane", size="1 1 .01", friction=".7 .005 .0001")
    t = _tip_boxes()["right"]["half"]
    b = ET.SubElement(w, "body", name="pusher", mocap="true", pos="0 0 -1")
    ET.SubElement(b, "geom", name="pusher_geom", type="box", size=PP.fmt(t),
                  friction=".6 .005 .0001", condim="4", solref=".006 1")
    _add_tower(w, None, frozen_below=frozen_below)
    return ET.tostring(root, encoding="unicode")


# ------------------------------------------------------------------ shared physics helpers
def block_ids(model, layers=LAYERS):
    return np.array([model.body(block_name(l, s)).id for l in range(layers) for s in range(3)])


def block_qadr(model, layers=LAYERS):
    return np.array([model.jnt_qposadr[model.joint(block_name(l, s) + "_joint").id]
                     for l in range(layers) for s in range(3)])


def block_dadr(model, layers=LAYERS):
    return np.array([model.jnt_dofadr[model.joint(block_name(l, s) + "_joint").id]
                     for l in range(layers) for s in range(3)])


def run(model, data, seconds, cb=None):
    for _ in range(max(1, round(seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)
        if cb is not None:
            cb()


def tower_failed(pos0, pos, exclude=()):
    keep = np.ones(len(pos0), bool)
    keep[list(exclude)] = False
    drop = pos0[keep, 2] - pos[keep, 2]
    drift = np.linalg.norm(pos[keep, :2] - pos0[keep, :2], axis=1)
    return bool(np.any(drop > FAIL_DROP) or np.any(drift > FAIL_DRIFT))


def park_far(data, qadr, dadr, i):
    """Removed blocks are taken off the table (out of both simulations)."""
    data.qpos[qadr[i]:qadr[i] + 7] = [.7, -.4 + .03 * i, .0076, 1, 0, 0, 0]
    data.qvel[dadr[i]:dadr[i] + 6] = 0


def pull_out(model, data, bid, dadr_i, x_start, cb=None):
    """Idealised extraction from the far side: a capped force drags the block out along +x.

    Stands in for the pull step the arm cannot do yet (it would have to reach
    behind the tower). The force is capped, so a pinned block drags its
    neighbours exactly as a real pull would.
    """
    n = round(PULL_SECS / model.opt.timestep)
    for k in range(n + round(.3 / model.opt.timestep)):
        target = x_start + PULL_TRAVEL * min(1.0, (k + 1) / n)
        f = 300 * (target - data.xpos[bid, 0]) - 4 * data.qvel[dadr_i]
        data.xfrc_applied[bid, 0] = float(np.clip(f, -PULL_MAX_N, PULL_MAX_N))
        mujoco.mj_step(model, data)
        if cb is not None:
            cb()
    data.xfrc_applied[bid] = 0


# ------------------------------------------------------------------ arm: fingertip IK and motion
class Arm:
    """IK on the two fingertip boxes: one tip on the block's end, the other mirrored
    across the claw axis, axis horizontal along +x (the only push direction the
    5-DOF arm can hold without twisting the block: the axis must be radial)."""

    def __init__(self, model):
        self.m = model
        self.k = mujoco.MjData(model)
        self.qadr = [PP.jadr(model, n) for n in PP.ARM]
        self.gadr = [PP.jadr(model, n) for n in ("gripper", "gripper_mimic")]
        self.tips = [model.geom("tip_finger_left").id, model.geom("tip_finger_right").id]
        self.base = model.body("custom_gripper_base").id
        self.lim = np.array([model.jnt_range[model.joint(n).id] for n in PP.ARM]) + [.03, -.03]
        self.tip_half_len = float(model.geom_size[self.tips[1], 0])
        sep0 = np.linalg.norm(np.subtract(*self.fk(np.zeros(5), 0.0)[:2]))
        self.sep_open, self.sep_closed = sep0, sep0 - 2 * STROKE

    def fk(self, q, stroke):
        k = self.k
        k.qpos[self.qadr] = q
        k.qpos[self.gadr] = stroke
        mujoco.mj_kinematics(self.m, k)
        L, R = k.geom_xpos[self.tips[0]].copy(), k.geom_xpos[self.tips[1]].copy()
        a = .5 * (L + R) - k.xpos[self.base]
        return L, R, a / np.linalg.norm(a)

    def stroke_for(self, half_sep):
        s = (self.sep_open - 2 * half_sep) / 2
        return s if 0 <= s <= STROKE else None

    def solve(self, push_tip, half_sep, seeds, w_axis=.05):
        """Right tip at push_tip, left tip mirrored to y = +half_sep, axis +x. -> (q, stroke, err_m)"""
        stroke = self.stroke_for(half_sep)
        if stroke is None:
            return None, None, np.inf
        tR = np.asarray(push_tip, float)
        tL = np.array([tR[0], tR[1] + 2 * half_sep, tR[2]])
        ax = np.array([1.0, 0, 0])

        def resid(q):
            L, R, a = self.fk(q, stroke)
            return np.concatenate([L - tL, R - tR, w_axis * (a - ax)])

        best = (None, np.inf)
        for q0 in seeds:
            q, lam = np.clip(np.array(q0, float), *self.lim.T), 1e-3
            r = resid(q)
            for _ in range(80):
                J = np.empty((9, 5))
                for j in range(5):
                    dq = np.zeros(5); dq[j] = 1e-5
                    J[:, j] = (resid(q + dq) - r) / 1e-5
                step = np.linalg.solve(J.T @ J + lam * np.eye(5), -J.T @ r)
                qn = np.clip(q + step, *self.lim.T); rn = resid(qn)
                if rn @ rn < r @ r:
                    q, r, lam = qn, rn, max(lam / 3, 1e-7)
                else:
                    lam *= 4
                if np.max(np.abs(r[:6])) < 2e-4 or lam > 1e3:
                    break
            err = float(np.max(np.abs(r[:6])))
            if err < best[1]:
                best = (q.copy(), err)
        return best[0], stroke, best[1]


STROKE = .03
SEEDS = [np.array(s) for s in ([0, -.8, 1.2, 0, 0], [0, -.5, .5, 0, 0], [0, -1.2, 1.5, .3, 0], [0, .2, -.2, -.3, 0])]


@dataclass
class PushPlan:
    layer: int
    slot: int
    stroke: float
    waypoints: dict            # name -> 5 joint angles
    ik_err_mm: float


def plan_push(arm: Arm, block_pose_xyz, q_hint=None):
    """Joint waypoints to put a fingertip on the block's near end and push PUSH_DEPTH in."""
    x_face = block_pose_xyz[0] - BLOCK_HALF[0]
    y, z = block_pose_xyz[1], block_pose_xyz[2]
    half_sep = abs(y)                         # claw axis must pass through the pan axis (y = 0)
    seeds = ([q_hint] if q_hint is not None else []) + SEEDS
    wps, worst, stroke, prev = {}, 0.0, None, None
    for name, dx in (("approach", -.030), ("pre", -.008), ("contact", -.001), ("push", PUSH_DEPTH)):
        tip = np.array([x_face + dx - arm.tip_half_len, y, z])
        q, stroke, err = arm.solve(tip, half_sep, ([prev] if prev is not None else []) + seeds)
        if q is None or err > .0015:
            return None
        wps[name], prev, worst = q, q, max(worst, err)
    return PushPlan(-1, -1, stroke, wps, worst * 1000)


# ------------------------------------------------------------------ TRUTH world (arm + hidden tower)
class Truth:
    def __init__(self, spec: TowerSpec):
        self.spec = spec
        self.m = mujoco.MjModel.from_xml_path(str(truth_xml(spec)))
        self.d = mujoco.MjData(self.m)
        self.arm = Arm(self.m)
        self.ids, self.qadr, self.dadr = block_ids(self.m), block_qadr(self.m), block_dadr(self.m)
        self.act = [self.m.actuator(n).id for n in PP.ARM]
        self.grip = self.m.actuator("gripper").id
        q, _, err = self.arm.solve([.27, -.03, .235], .03, SEEDS)
        assert err < .003, f"home pose unreachable ({err * 1000:.1f} mm)"
        self.home = q
        self.d.qpos[self.arm.qadr] = q
        self.d.ctrl[self.act] = q
        mujoco.mj_forward(self.m, self.d)
        run(self.m, self.d, 1.0)                     # let the tower settle under gravity
        self.removed: list[int] = []

    def block_pos(self):
        return self.d.xpos[self.ids].copy()

    def block_qpos(self):
        return np.array([self.d.qpos[a:a + 7] for a in self.qadr])

    def copy(self):
        t = Truth.__new__(Truth)
        t.__dict__.update(self.__dict__)
        t.d = mujoco.MjData(self.m)
        mujoco.mj_copyData(t.d, self.m, self.d)
        t.arm = Arm(self.m)
        t.removed = list(self.removed)
        return t

    def _move(self, q_to, stroke, secs, cb, stop=None):
        """Quintic joint-space move. `stop()` is polled at camera rate (~30 Hz); if it
        returns True the arm freezes where it is. Returns True if stopped early."""
        q0 = self.d.ctrl[self.act].copy(); g0 = float(self.d.ctrl[self.grip])
        n = max(1, round(secs / self.m.opt.timestep))
        for i in range(n):
            u = (i + 1) / n; s = 6 * u**5 - 15 * u**4 + 10 * u**3
            self.d.ctrl[self.act] = q0 + s * (q_to - q0)
            self.d.ctrl[self.grip] = g0 + s * (stroke - g0)
            mujoco.mj_step(self.m, self.d)
            if cb is not None:
                cb()
            if stop is not None and i % 16 == 0 and stop():
                self.d.ctrl[self.act] = self.d.qpos[self.arm.qadr]
                return True
        return False

    def test_push(self, i, plan: PushPlan, rng, cb=None):
        """Arm pushes block i PUSH_DEPTH in and backs off. Returns what a camera would see."""
        p0 = self.block_pos()
        others = [k for k in range(len(p0)) if k != i and k not in self.removed]
        w = plan.waypoints
        for q, secs in ((w["approach"], 1.2), (w["pre"], .5), (w["contact"], .4)):
            self._move(q, plan.stroke, secs, cb)
        ref = self.block_pos()                      # tower at the moment of contact
        noise = lambda: abs(rng.normal(0, 1000 * POSE_NOISE_M))

        def tower_moving():                          # camera, ~30 Hz, during the push only
            moved = np.linalg.norm(self.d.xpos[self.ids[others], :2] - ref[others, :2], axis=1).max()
            return 1000 * moved + noise() > REFLEX_STOP_MM

        stopped = self._move(w["push"], plan.stroke, .8, cb, stop=tower_moving)
        run(self.m, self.d, .2, cb)
        p1 = self.block_pos()
        for q, secs in ((w["pre"], .5), (w["approach"], .4)):
            self._move(q, plan.stroke, secs, cb)
        moved = np.linalg.norm(p1[others, :2] - ref[others, :2], axis=1)
        return dict(p0=p0, reflex_stop=bool(stopped),
                    pushed_mm=float(1000 * (p1[i, 0] - ref[i, 0]) + rng.normal(0, 1000 * POSE_NOISE_M)),
                    response_mm=float(1000 * moved.max() + noise()),
                    fell=tower_failed(p0, self.block_pos(), exclude=self.removed + [i]))

    def finish(self, i, p0, cb=None):
        """Pull block i out (idealised), park it, return home, settle. -> fell?"""
        pull_out(self.m, self.d, self.ids[i], self.dadr[i], self.d.xpos[self.ids[i], 0], cb)
        park_far(self.d, self.qadr, self.dadr, i)
        self.removed.append(i)
        self._move(self.home, 0.0, .8, cb)
        run(self.m, self.d, .6, cb)
        return tower_failed(p0, self.block_pos(), exclude=self.removed)

    def retreat(self, cb=None):
        self._move(self.home, 0.0, .8, cb)
        run(self.m, self.d, .3, cb)


# ------------------------------------------------------------------ MODEL world (the gate)
class _Variant:
    """World model compiled with every layer below `frozen_below` static."""

    def __init__(self, frozen_below):
        self.frozen_below = frozen_below
        self.m = mujoco.MjModel.from_xml_string(world_model_xml(frozen_below))
        self.d = mujoco.MjData(self.m)
        n = LAYERS * 3
        self.ids = np.array([self.m.body(block_name(l, s)).id for l in range(LAYERS) for s in range(3)])
        self.geoms = np.array([self.m.geom(block_name(l, s) + "_geom").id for l in range(LAYERS) for s in range(3)])
        self.free = [k for k in range(n) if k // 3 >= frozen_below]
        self.qadr = {k: self.m.jnt_qposadr[self.m.joint(block_name(k // 3, k % 3) + "_joint").id] for k in self.free}
        self.dadr = {k: self.m.jnt_dofadr[self.m.joint(block_name(k // 3, k % 3) + "_joint").id] for k in self.free}
        self.pusher = self.m.body("pusher").mocapid[0]
        self.tip_half = self.m.geom_size[self.m.geom("pusher_geom").id].copy()
        # nominal top of the frozen part, to seat the copied free blocks on it
        self.base_top = frozen_below * LAYER_PITCH if frozen_below else 0.0
        self.saved = []


class WorldModel:
    """Nominal tower + kinematic fingertip. Hidden block thickness and friction are
    sampled from the same prior the real tower comes from, then weighted by how
    well each sample reproduces the tower's observed reaction to the test push.

    Speed-ups (measured, ~5x): layers more than one below the target are static,
    and only samples that explain the observation are simulated through the pull."""

    def __init__(self):
        self.variants = {}

    def _variant(self, layer):
        fb = max(0, layer - 1)
        if fb not in self.variants:
            self.variants[fb] = _Variant(fb)
        return self.variants[fb]

    @staticmethod
    def _set_hidden(v, rng):
        n = len(v.geoms)
        h = BLOCK_HALF[2] + rng.uniform(-.0003, .0003, n)
        h[: v.frozen_below * 3] = BLOCK_HALF[2]
        v.m.geom_size[v.geoms, 2] = h
        v.m.geom_rbound[v.geoms] = np.linalg.norm(np.column_stack([np.full(n, BLOCK_HALF[0]),
                                                                    np.full(n, BLOCK_HALF[1]), h]), axis=1)
        v.m.geom_aabb[v.geoms, 5] = h
        v.m.geom_friction[v.geoms, 0] = rng.uniform(.30, .70, n)

    def _push_stage(self, v, qpos, removed, i, rng):
        m, d = v.m, v.d
        mujoco.mj_resetData(m, d)
        self._set_hidden(v, rng)
        on = np.ones(len(v.geoms), bool); on[[k for k in removed if k not in v.qadr]] = False
        m.geom_contype[v.geoms] = on; m.geom_conaffinity[v.geoms] = on
        # seat the observed free blocks on the nominal frozen base
        lowest = min(qpos[k][2] for k in v.free if k not in removed) - BLOCK_HALF[2]
        dz = v.base_top - lowest
        for k, a in v.qadr.items():
            d.qpos[a:a + 7] = qpos[k]
            d.qpos[a + 2] += dz
        for k in removed:
            if k in v.qadr:
                park_far(d, v.qadr, v.dadr, k)
        d.mocap_pos[v.pusher] = [0, 0, -1]
        mujoco.mj_forward(m, d)
        run(m, d, SETTLE_S)
        p0 = d.xpos[v.ids].copy()
        face = p0[i] - np.array([BLOCK_HALF[0] + v.tip_half[0] + .0005, 0, 0])
        others = [k for k in v.free if k != i and k not in removed]
        moved = lambda: 1000 * np.linalg.norm(d.xpos[v.ids[others], :2] - p0[others, :2], axis=1).max()
        n = round(.8 / m.opt.timestep)
        for k in range(n):                           # same reflex as the arm: stop when the tower moves
            d.mocap_pos[v.pusher] = face + [(PUSH_DEPTH + .0005) * (k + 1) / n, 0, 0]
            mujoco.mj_step(m, d)
            if k % 16 == 0 and moved() > REFLEX_STOP_MM:
                break
        run(m, d, .2)
        depth = 1000 * (d.xpos[v.ids[i], 0] - p0[i, 0])
        return float(moved()), float(depth), p0

    def _pull_stage(self, v, state, removed, i):
        m, d = v.m, v.d
        mujoco.mj_setState(m, d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        d.mocap_pos[v.pusher] = [0, 0, -1]
        mujoco.mj_forward(m, d)                      # positions of THIS sample, not the last one simulated
        pull_out(m, d, v.ids[i], v.dadr[i], d.xpos[v.ids[i], 0])
        park_far(d, v.qadr, v.dadr, i)
        run(m, d, 1.0)                    # the truth arm takes ~1.4 s to clear; a tipping tower needs that long
        moving = max(np.linalg.norm(d.qvel[v.dadr[k]:v.dadr[k] + 3]) for k in v.free if k != i)
        return d.xpos[v.ids].copy(), moving

    def p_fall(self, qpos, removed, i, obs, rng, samples=GATE_SAMPLES):
        """Observation-weighted P(fall) and effective sample size."""
        v = self._variant(i // 3)
        removed = list(removed)
        resp, depth, starts, states, hidden = [], [], [], [], []
        size = mujoco.mj_stateSize(v.m, mujoco.mjtState.mjSTATE_INTEGRATION)
        for _ in range(samples):
            seed = int(rng.integers(1 << 31))
            r, dep, p0 = self._push_stage(v, qpos, removed, i, np.random.default_rng(seed))
            st = np.empty(size); mujoco.mj_getState(v.m, v.d, st, mujoco.mjtState.mjSTATE_INTEGRATION)
            resp.append(r); depth.append(dep); starts.append(p0); states.append(st); hidden.append(seed)
        resp, depth = np.array(resp), np.array(depth)
        # Likelihood of the observation: how deep the push got before the reflex, and how
        # much the tower moved. Widths = camera noise + arm-vs-model mismatch.
        sig_r = max(1.0, .25 * obs["response_mm"])
        w = np.exp(-.5 * ((resp - obs["response_mm"]) / sig_r) ** 2
                   - .5 * ((depth - obs["pushed_mm"]) / 2.0) ** 2)
        ess = float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12))
        fell = np.zeros(samples)
        for k in np.flatnonzero(w > .02 * w.max()):           # only samples that explain the observation
            self._set_hidden(v, np.random.default_rng(hidden[k]))
            on = np.ones(len(v.geoms), bool); on[[r for r in removed if r not in v.qadr]] = False
            v.m.geom_contype[v.geoms] = on; v.m.geom_conaffinity[v.geoms] = on
            p_end, moving = self._pull_stage(v, states[k], removed, i)
            fell[k] = tower_failed(starts[k], p_end, exclude=removed + [i]) or moving > .05
        p = float((w * fell).sum() / w.sum()) if w.sum() > 1e-9 else float("nan")
        return dict(p_fall=p, ess=ess, model_response_mm=[round(float(r), 2) for r in resp],
                    model_depth_mm=[round(float(x), 1) for x in depth])


def gate_decision(g, threshold=GATE_THRESHOLD, ess_min=GATE_ESS_MIN):
    """Block when the model predicts a fall, or when too few sampled towers explain what was seen."""
    if not np.isfinite(g["p_fall"]) or g["ess"] < ess_min:
        return False, "uncertain"
    return (g["p_fall"] <= threshold), ("predicted fall" if g["p_fall"] > threshold else "ok")


# ------------------------------------------------------------------ episodes and the comparison
MAX_REMOVALS, MAX_ATTEMPTS = 5, 10
CANDIDATES = [block_index(L, s) for L in PUSH_LAYERS for s in (1, 2)]


def _measured(truth, rng):
    q = truth.block_qpos()
    q[:, :3] += rng.normal(0, POSE_NOISE_M, (len(q), 3))
    return q


def episode(seed, gated, cb=None, log=print, wm=None):
    """One game on tower `seed`. The proposer picks blocks at random (same sequence
    for gated and ungated until their choices diverge)."""
    truth = Truth(TowerSpec.sample(seed))
    noise, proposer = np.random.default_rng(seed + 7), np.random.default_rng(seed + 99)
    wm = wm or (WorldModel() if gated else None)
    rejected, decisions, result = set(), [], "attempts used up"
    for attempt in range(MAX_ATTEMPTS):
        if len(truth.removed) >= MAX_REMOVALS:
            result = "max removals"
            break
        pool = [i for i in CANDIDATES if i not in truth.removed and i not in rejected]
        if not pool:
            result = "stopped: no block judged safe"
            break
        i = pool[int(proposer.integers(len(pool)))]
        name = f"L{i // 3}S{i % 3}"
        plan = plan_push(truth.arm, truth.block_pos()[i])
        if plan is None:
            rejected.add(i); continue
        qpos = _measured(truth, noise)
        obs = truth.test_push(i, plan, noise, cb)
        if obs["fell"]:
            log(f"  {name}: tower fell during the test push"); result = "fell (test push)"; break
        if gated:
            g = wm.p_fall(qpos, truth.removed, i, obs, noise)
            ok, why = gate_decision(g)
            if not ok:
                cf = truth.copy()
                cf_fell = cf.finish(i, obs["p0"])     # counterfactual: what the pull would have done
                decisions.append(dict(block=name, p_fall=g["p_fall"], ess=g["ess"], allowed=False,
                                      reason=why, would_fall=cf_fell))
                log(f"  {name}: test push moved the tower {obs['response_mm']:.1f} mm"
                    f"{' (reflex stop at ' + format(obs['pushed_mm'], '.1f') + ' mm)' if obs['reflex_stop'] else ''} -> P(fall)={g['p_fall']:.2f} "
                    f"ESS={g['ess']:.1f} BLOCKED ({why}); counterfactual: {'would fall' if cf_fell else 'would be fine'}")
                rejected.add(i)
                truth.retreat(cb)
                continue
        fell = truth.finish(i, obs["p0"], cb)
        if gated:
            decisions.append(dict(block=name, p_fall=g["p_fall"], ess=g["ess"], allowed=True, reason=why,
                                  would_fall=fell))
        log(f"  {name}: response {obs['response_mm']:.1f} mm{' (reflex stop)' if obs['reflex_stop'] else ''}"
            + (f", P(fall)={g['p_fall']:.2f} ALLOWED" if gated else "") + f" -> {'TOWER FELL' if fell else 'removed'}")
        if fell:
            result = "fell"
            break
    removed_ok = len(truth.removed) - (1 if result == "fell" else 0)
    return dict(seed=seed, gated=gated, result=result, fell=result.startswith("fell"),
                blocks_removed=removed_ok, decisions=decisions)


def _episode_job(args):
    seed, gated = args
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = episode(seed, gated)
    r["log"] = buf.getvalue()
    return r


def compare(seeds, workers=2, out=LOG_DIR / "jenga_compare.json"):
    import multiprocessing as mp
    jobs = [(s, g) for s in seeds for g in (False, True)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        results = []
        for r in pool.imap_unordered(_episode_job, jobs):
            results.append(r)
            print(f"[{time.time() - t0:5.0f}s] tower {r['seed']} {'gated  ' if r['gated'] else 'ungated'}: "
                  f"{r['result']}, {r['blocks_removed']} removed", flush=True)
    summary = summarize(results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(summary=summary, threshold=GATE_THRESHOLD, ess_min=GATE_ESS_MIN,
                                   samples=GATE_SAMPLES, results=sorted(results, key=lambda r: (r["seed"], r["gated"]))),
                              indent=2, default=float) + "\n")
    print(format_summary(summary))
    print(f"saved {out}")
    return summary


def summarize(results):
    s = {}
    for gated in (False, True):
        rs = [r for r in results if r["gated"] == gated]
        s["gated" if gated else "ungated"] = dict(
            episodes=len(rs), towers_fell=sum(r["fell"] for r in rs),
            stopped_safely=sum(r["result"].startswith("stopped") for r in rs),
            blocks_removed=sum(r["blocks_removed"] for r in rs),
            completed=sum(r["result"] == "max removals" for r in rs),
            out_of_attempts=sum(r["result"] == "attempts used up" for r in rs))
    d = [x for r in results if r["gated"] for x in r["decisions"]]
    s["gate"] = dict(decisions=len(d),
                     blocked_would_fall=sum(not x["allowed"] and x["would_fall"] for x in d),
                     blocked_would_be_fine=sum(not x["allowed"] and not x["would_fall"] for x in d),
                     allowed_fell=sum(x["allowed"] and x["would_fall"] for x in d),
                     allowed_fine=sum(x["allowed"] and not x["would_fall"] for x in d))
    return s


def format_summary(s):
    u, g, q = s["ungated"], s["gated"], s["gate"]
    rows = [("Towers that fell", u["towers_fell"], g["towers_fell"]),
            ("Games finished (5 blocks out)", u["completed"], g["completed"]),
            ("Games stopped safely (no block judged safe / attempts used up)", "n/a",
             g["stopped_safely"] + g["out_of_attempts"]),
            ("Blocks removed (total)", u["blocks_removed"], g["blocks_removed"])]
    out = [f"\n| Metric ({u['episodes']} towers each) | Ungated | Gated |", "|---|---|---|"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    falls = q["blocked_would_fall"] + q["allowed_fell"]
    fine = q["blocked_would_be_fine"] + q["allowed_fine"]
    out += ["", f"Gate decisions: {q['decisions']}. Pulls that would have toppled the tower: {falls}, "
                f"blocked {q['blocked_would_fall']}. Safe pulls: {fine}, wrongly blocked {q['blocked_would_be_fine']}."]
    return "\n".join(out)


# ------------------------------------------------------------------ CLI
def _viewer_cb(viewer, model):
    state = dict(n=0, t=time.monotonic())

    def cb():
        state["n"] += 1
        if state["n"] % 20 == 0:
            viewer.sync()
            lag = state["n"] * model.opt.timestep - (time.monotonic() - state["t"])
            if lag > 0:
                time.sleep(min(lag, .04))
            if not viewer.is_running():
                raise KeyboardInterrupt
    return cb, state


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=101, help="tower seed for a single episode")
    ap.add_argument("--ungated", action="store_true", help="single episode without the gate")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--record", metavar="FILE.mp4")
    ap.add_argument("--compare", action="store_true", help="gated vs ungated over --episodes towers")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=101, help="first evaluation tower (1-2 were calibration)")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    if a.compare:
        compare(range(a.seed0, a.seed0 + a.episodes), a.workers)
        return
    gated = not a.ungated
    print(f"Tower {a.seed}, {'gated' if gated else 'ungated'}:")
    if a.headless:
        r = episode(a.seed, gated)
    elif a.record:
        truth_probe = Truth(TowerSpec.sample(a.seed))
        rec = PP.VideoRecorder(truth_probe.m, truth_probe.d, a.record)
        rec.cam.lookat[:] = [.30, -.02, .12]; rec.cam.distance, rec.cam.azimuth, rec.cam.elevation = .62, 215, -14
        holder = {}

        def cb():
            holder["n"] = holder.get("n", 0) + 1
            if holder["n"] % 20 == 0:
                rec.sync()
        # run the episode on the recorder's model/data so the frames show it
        orig = Truth.__init__

        def init(self, spec):
            self.__dict__.update(truth_probe.__dict__)
            rec.data = self.d
        Truth.__init__ = init
        try:
            r = episode(a.seed, gated, cb)
        finally:
            Truth.__init__ = orig
        rec.save()
    else:
        import mujoco.viewer
        truth_probe = Truth(TowerSpec.sample(a.seed))
        orig = Truth.__init__
        Truth.__init__ = lambda self, spec: self.__dict__.update(truth_probe.__dict__)
        with mujoco.viewer.launch_passive(truth_probe.m, truth_probe.d) as viewer:
            viewer.cam.lookat[:] = [.30, -.02, .12]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = .62, 215, -14
            cb, _ = _viewer_cb(viewer, truth_probe.m)
            try:
                r = episode(a.seed, gated, cb)
                print("Close the window to exit.")
                while viewer.is_running():
                    viewer.sync(); time.sleep(.05)
            except KeyboardInterrupt:
                return
            finally:
                Truth.__init__ = orig
    print(f"Result: {r['result']}, {r['blocks_removed']} blocks removed")


if __name__ == "__main__":
    main()
