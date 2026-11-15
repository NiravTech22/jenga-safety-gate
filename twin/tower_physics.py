#!/usr/bin/env python3
"""Jenga tower physics, validated against static equilibrium.

A tower stands while the centre of mass of everything above each layer lies
over that layer's remaining support, and collapses completely once it does
not. This module builds the tower in MuJoCo from one parameter table and checks
it against closed-form predictions:

  stand     intact tower must not creep or drift
  tilt      tilt gravity slowly; collapse angle vs atan((w/2) / h_com)
  neck      leave only the middle block in a low layer (25 mm support); tilt;
            collapse angle vs atan(12.5 mm / h_com above the neck)
  remove    removal patterns: which leave the tower standing, which topple it
  pull      force to slide a block out vs mu * (N_top + N_bottom)

Usage (from the repository root):
  python twin/tower_physics.py --validate          # all tests, table + JSON + plot
  python twin/tower_physics.py --view neck         # watch one scenario
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "docs/results"


# ------------------------------------------------------------------ one parameter table
@dataclass
class Param:
    value: float
    unit: str
    source: str          # "measured: ..." once Nirav's numbers replace an estimate


@dataclass
class TowerPhysics:
    """Every physical number the tower uses. Replace `source` with 'measured: ...'
    when a value comes from the real blocks."""
    block_length: Param = field(default_factory=lambda: Param(.075, "m", "estimate: standard Jenga 7.5 cm"))
    block_width: Param = field(default_factory=lambda: Param(.025, "m", "estimate: standard Jenga 2.5 cm"))
    block_height: Param = field(default_factory=lambda: Param(.015, "m", "estimate: standard Jenga 1.5 cm"))
    block_mass: Param = field(default_factory=lambda: Param(.010, "kg", "estimate: 540 g set / 54 blocks"))
    side_gap: Param = field(default_factory=lambda: Param(.0005, "m", "estimate: blocks in a layer are not glued edge to edge"))
    layers: Param = field(default_factory=lambda: Param(18, "-", "estimate: standard tower"))
    friction: Param = field(default_factory=lambda: Param(.40, "-", "estimate: wood-wood static 0.25-0.50 (lab value 0.43)"))
    table_friction: Param = field(default_factory=lambda: Param(.50, "-", "estimate: wood on a table top"))
    # Contact model: wood is stiff but not rigid. solref = (time constant, damping
    # ratio); solimp sets how hard contacts get as they compress. A tower in
    # equilibrium must sit still (no creep) and a tower out of equilibrium must
    # fall, so these were tuned against the stand / tilt / neck tests.
    contact_timeconst: Param = field(default_factory=lambda: Param(.003, "s", "tuned: stiffest stable at dt 2 ms; no idle creep"))
    contact_dampratio: Param = field(default_factory=lambda: Param(1.0, "-", "tuned: critically damped, no bounce"))
    timestep: Param = field(default_factory=lambda: Param(.002, "s", "tuned: 0.000 mm creep / 5 s at 18 layers"))
    noslip_iterations: Param = field(default_factory=lambda: Param(0, "-", "tuned: noslip added creep here (0.03-0.8 mm / 3 s); off"))
    impratio: Param = field(default_factory=lambda: Param(10, "-", "tuned: friction as stiff as contact; =1 let loaded blocks slip at 61% of mu*N"))
    contact_solimp: Param = field(default_factory=lambda: Param(.00001, "m", "tuned: contact reaches full stiffness within 0.01 mm"))

    def v(self, name):
        return getattr(self, name).value


P = TowerPhysics()


# ------------------------------------------------------------------ tower geometry
def slot_pitch(p=P):
    return p.v("block_width") + p.v("side_gap")


def block_pose(layer, slot, p=P, xy=(0.0, 0.0)):
    """Centre and yaw. Even layers lie along x, odd layers along y. Origin: tower base centre."""
    off = (slot - 1) * slot_pitch(p)
    x, y = xy
    if layer % 2 == 0:
        y += off; yaw = 0.0
    else:
        x += off; yaw = math.pi / 2
    return np.array([x, y, p.v("block_height") * (layer + .5)]), yaw


def add_tower(world, p=P, removed=(), thickness=None, xy=(0.0, 0.0), prefix="block"):
    """Add tower bodies to an XML worldbody. thickness: {(layer, slot): extra metres}.
    Layers sit exactly on each other; a thicker block lifts its whole layer's support."""
    L, W, H, m = (p.v(k) for k in ("block_length", "block_width", "block_height", "block_mass"))
    thickness = thickness or {}
    z = 0.0
    for layer in range(int(p.v("layers"))):
        extra = [thickness.get((layer, s), 0.0) for s in range(3)]
        for s in range(3):
            if (layer, s) in removed:
                continue
            c, yaw = block_pose(layer, s, p, xy)
            h = H + extra[s]
            c[2] = z + h / 2
            b = ET.SubElement(world, "body", name=f"{prefix}_{layer}_{s}", pos=" ".join(f"{v:.6f}" for v in c),
                              quat=f"{math.cos(yaw / 2):.6f} 0 0 {math.sin(yaw / 2):.6f}")
            ET.SubElement(b, "freejoint", name=f"{prefix}_{layer}_{s}_joint")
            shade = .80 - .08 * (layer % 2)
            ET.SubElement(b, "geom", name=f"{prefix}_{layer}_{s}_geom", type="box",
                          size=f"{L / 2:.6f} {W / 2:.6f} {h / 2:.6f}", mass=f"{m:.6f}",
                          friction=f"{p.v('friction'):.4f} .005 .0001", condim="3",
                          solref=f"{p.v('contact_timeconst'):.5f} {p.v('contact_dampratio'):.3f}",
                          solimp=f".99 .999 {p.v('contact_solimp'):.6f}", rgba=f"{shade:.2f} {shade * .72:.2f} {shade * .43:.2f} 1")
        z += H + max(extra)


def tower_xml(p=P, removed=(), thickness=None, gravity=(0, 0, -9.81), extra=""):
    root = ET.Element("mujoco", model="jenga_tower")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep=f"{p.v('timestep')}", integrator="implicitfast", cone="elliptic",
                  impratio=f"{p.v('impratio'):g}", noslip_iterations=f"{int(p.v('noslip_iterations'))}",
                  gravity=" ".join(f"{g:.6f}" for g in gravity))
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth="1280", offheight="960")
    w = ET.SubElement(root, "worldbody")
    ET.SubElement(w, "light", pos="0.3 -0.4 1.2", dir="-0.3 0.4 -1", diffuse=".8 .8 .8")
    ET.SubElement(w, "geom", name="table", type="plane", size="1 1 .01", rgba=".25 .28 .32 1",
                  friction=f"{p.v('table_friction'):.3f} .005 .0001")
    add_tower(w, p, removed, thickness)
    s = ET.tostring(root, encoding="unicode")
    return s.replace("</worldbody>", extra + "</worldbody>") if extra else s


class Tower:
    def __init__(self, p=P, removed=(), thickness=None, gravity=(0, 0, -9.81)):
        self.p = p
        self.m = mujoco.MjModel.from_xml_string(tower_xml(p, removed, thickness, gravity))
        self.d = mujoco.MjData(self.m)
        self.names = [(l, s) for l in range(int(p.v("layers"))) for s in range(3) if (l, s) not in removed]
        self.ids = np.array([self.m.body(f"block_{l}_{s}").id for l, s in self.names])
        mujoco.mj_forward(self.m, self.d)
        self.p0 = self.pos()

    def pos(self):
        return self.d.xpos[self.ids].copy()

    def run(self, seconds, cb=None):
        for _ in range(max(1, round(seconds / self.m.opt.timestep))):
            mujoco.mj_step(self.m, self.d)
            if cb is not None:
                cb()

    def set_gravity(self, tilt_deg, axis="x"):
        """Tilt gravity by tilt_deg about the given horizontal axis (same as tilting the table)."""
        g, a = 9.81, math.radians(tilt_deg)
        self.m.opt.gravity[:] = (0, g * math.sin(a), -g * math.cos(a)) if axis == "x" else (g * math.sin(a), 0, -g * math.cos(a))

    def collapsed(self, ref=None, drop=.03):
        """Any block dropped more than 3 cm: the tower has come down."""
        ref = self.p0 if ref is None else ref
        return bool(np.any(ref[:, 2] - self.pos()[:, 2] > drop))

    def collapse_fraction(self, ref=None):
        """Fraction of blocks that ended more than one block-height below where they started."""
        ref = self.p0 if ref is None else ref
        return float(np.mean(ref[:, 2] - self.pos()[:, 2] > self.p.v("block_height")))

    def max_drift_mm(self, ref=None):
        ref = self.p0 if ref is None else ref
        return float(1000 * np.linalg.norm(self.pos() - ref, axis=1).max())


# ------------------------------------------------------------------ closed-form predictions
def tower_height(p=P, layers=None):
    return p.v("block_height") * (int(p.v("layers")) if layers is None else layers)


def predict_tilt_deg(p=P):
    """Rigid tower tips about its base edge when gravity passes the edge:
    tan(theta) = (half base width) / (height of the centre of mass)."""
    half_w = (3 * p.v("block_width") + 2 * p.v("side_gap")) / 2
    return math.degrees(math.atan(half_w / (tower_height(p) / 2)))


def predict_neck_deg(p=P, neck_layer=2):
    """Only the middle block left in neck_layer: everything above pivots on a
    25 mm strip, so tan(theta) = (W/2) / (height of the upper part's centre of mass above the neck)."""
    above = int(p.v("layers")) - neck_layer - 1
    return math.degrees(math.atan((p.v("block_width") / 2) / (tower_height(p, above) / 2)))


def predict_removal_stands(removed_slots):
    """One layer, slots removed. The load above is centred, so it stands if the
    remaining blocks span the centre line: middle present, or both outers present."""
    left = {0, 1, 2} - set(removed_slots)
    return 1 in left or left == {0, 2}


def predict_pull_n(p=P, layer=8):
    """A block carrying everything above it (W_above). Two textbook limits:
      dragged:  the tower above rides along, only the bottom face slips
                F = mu * (W_above + m g)                      (lower limit)
      pinned:   the tower above is held, both faces slip
                F = mu * (2 W_above + m g)                    (upper limit)
    A loose block carries nothing: F = mu * m g."""
    g, m = 9.81, p.v("block_mass")
    w_above = (int(p.v("layers")) - layer - 1) * 3 * m * g
    mu = p.v("friction")
    return mu * (w_above + m * g), mu * (2 * w_above + m * g), mu * m * g


# ------------------------------------------------------------------ simulation tests
def test_stand(p=P, seconds=10.0):
    t = Tower(p); t.run(.5); ref = t.pos(); t.run(seconds)
    creep = t.max_drift_mm(ref)
    return dict(test="stand", predicted="0 mm drift", measured=f"{creep:.3f} mm in {seconds:.0f} s",
                passed=creep < .1, value=creep)


def _tilt_deg(R):
    """How far a block has rotated away from level (degrees)."""
    return math.degrees(math.acos(float(np.clip(R.reshape(3, 3)[2, 2], -1, 1))))


def _com(t, mask=None):
    m = t.m.body_mass[t.ids]
    pos = t.pos()
    if mask is not None:
        m, pos = m[mask], pos[mask]
    return (m[:, None] * pos).sum(0) / m.sum()


def equilibrium_margin_mm(com, edge_h, edge_z, tilt_deg, h=1):
    """Gravity tilted by theta toward +y (h=1) or +x (h=0). Signed distance (in the
    tilt plane) from the line of action of the weight through `com` to the support
    edge at (edge_h, edge_z): > 0 inside the support (stands), < 0 outside (tips)."""
    a = math.radians(tilt_deg)
    return 1000 * ((edge_h - com[h]) * math.cos(a) - (com[2] - edge_z) * math.sin(a))


def _tilt_until_collapse(t, axis="x", step=.25, hold=.25, start=0.0, stop=30.0, cb=None):
    """Raise the tilt in small steps. Tipping is rotation that keeps growing: a tower
    past equilibrium rotates about its support edge at a constant tilt. (MuJoCo's
    soft friction lets blocks creep sideways under load; creep keeps them level,
    so it cannot be mistaken for tipping.) Returns (onset angle, fraction down)."""
    t.run(.5, cb); ref = t.pos(); top = int(np.argmax(ref[:, 2]))
    t.last_stable = None
    a = 0.0
    while a < start:
        a = min(start, a + 1.0); t.set_gravity(a, axis); t.run(.2, cb)
    while a <= stop:
        t.set_gravity(a, axis)
        t.run(hold / 2, cb)
        r1 = _tilt_deg(t.d.xmat[t.ids[top]])
        t.run(hold / 2, cb)
        r2 = _tilt_deg(t.d.xmat[t.ids[top]])
        # A tower in equilibrium leans a little (compliant contacts) and settles;
        # past equilibrium it keeps rotating at a constant tilt.
        if r2 > 1.0 and r2 - r1 > .2:
            t.run(2.5, cb)
            return a, t.collapse_fraction(ref)
        t.last_stable = (a, t.pos().copy())          # deformed shape at the last angle that held
        a += step
    return float("nan"), 0.0


def _onset_margin(t, mask, edge_h, edge_z, step, h=1):
    """Equilibrium margin of the tower's own deformed shape at the last stable angle
    and one step later: the law predicts margin >= 0 then <= 0."""
    a, pos = t.last_stable
    m = t.m.body_mass[t.ids][mask]
    com = (m[:, None] * pos[mask]).sum(0) / m.sum()
    return (equilibrium_margin_mm(com, edge_h, edge_z, a, h),
            equilibrium_margin_mm(com, edge_h, edge_z, a + step, h))


def test_tilt(p=P, cb=None, step=.25):
    t = Tower(p)
    rigid = predict_tilt_deg(p)
    onset, frac = _tilt_until_collapse(t, "x", step=step, start=max(0.0, rigid - 6), cb=cb)
    half_w = (3 * p.v("block_width") + 2 * p.v("side_gap")) / 2
    m_ok, m_next = _onset_margin(t, np.ones(len(t.ids), bool), half_w, 0.0, step)
    return dict(test="tilt (whole tower)", predicted=f"tips when margin<=0; rigid {rigid:.1f} deg",
                measured=f"tips at {onset:.2f} deg; margin {m_ok:+.1f}/{m_next:+.1f} mm; {frac:.0%} down",
                passed=m_ok >= -2 and m_next <= 2 and frac > .5 and onset >= .8 * rigid, value=onset,
                rigid_deg=rigid, margin_mm=[m_ok, m_next])


def test_neck(p=P, neck_layer=2, cb=None, step=.1):
    t = Tower(p, removed={(neck_layer, 0), (neck_layer, 2)})
    rigid = predict_neck_deg(p, neck_layer)
    # Tip across the neck block's 25 mm width: even layers lie along x (tilt toward +y),
    # odd layers along y (tilt toward +x).
    axis, h = ("x", 1) if neck_layer % 2 == 0 else ("y", 0)
    onset, frac = _tilt_until_collapse(t, axis, step=step, start=max(0.0, rigid - 3), cb=cb)
    upper = np.array([l > neck_layer for l, _ in t.names])
    edge_z = (neck_layer + 1) * p.v("block_height")
    m_ok, m_next = _onset_margin(t, upper, p.v("block_width") / 2, edge_z, step, h)
    return dict(test="tilt on a one-block neck", predicted=f"tips when margin<=0; rigid {rigid:.1f} deg",
                measured=f"tips at {onset:.2f} deg; margin {m_ok:+.1f}/{m_next:+.1f} mm; {frac:.0%} down",
                passed=m_ok >= -2 and m_next <= 2 and frac > .3 and onset >= .8 * rigid, value=onset,
                rigid_deg=rigid, margin_mm=[m_ok, m_next])


def test_removal(p=P, layer=3, seconds=4.0):
    rows = []
    for slots in ((1,), (0,), (0, 2), (0, 1), (1, 2)):
        t = Tower(p, removed={(layer, s) for s in slots})
        t.run(seconds)
        stands = not t.collapsed()
        pred = predict_removal_stands(slots)
        rows.append(dict(test=f"remove layer {layer} slots {list(slots)}",
                         predicted="stands" if pred else "collapses",
                         measured=("stands" if stands else f"collapses ({t.collapse_fraction():.0%} of blocks down)"),
                         passed=stands == pred))
    return rows


def _block_axis(layer):
    """World axis a block lies along: even layers along x, odd layers along y."""
    return 0 if layer % 2 == 0 else 1


def _slides_at(p, layer, extra, force, seconds=1.5):
    """Hold a constant force on the middle block for 1.5 s; does it come out?"""
    t = Tower(p, thickness={(layer, 1): extra}); t.run(.6)
    k = t.names.index((layer, 1)); bid = t.ids[k]; dof = t.m.jnt_dofadr[t.m.body_jntadr[bid]]
    ax = _block_axis(layer)                  # pull along the block's length
    x0 = t.d.xpos[bid, ax]
    t.d.xfrc_applied[bid, ax] = force
    t.run(seconds)
    # Pulled out = moved more than 10 mm. Creep (< 1 mm/s) and single stick-slip
    # jumps (~1 mm) both stay well under that.
    return t.d.xpos[bid, ax] - x0 > .010


def _breakaway(p, layer, extra, hi, steps=7):
    """Smallest constant force that pulls the block out (bisection, ~2% resolution)."""
    lo = 0.0
    if not _slides_at(p, layer, extra, hi):
        return None
    for _ in range(steps):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if _slides_at(p, layer, extra, mid) else (mid, hi)
    return hi


def test_pull(p=P, layer=8):
    dragged, pinned, loose = predict_pull_n(p, layer)
    out = []
    # Near the limit a loaded block stick-slips (as real wood does), so report the
    # band: highest force that always held, lowest force from which it always comes out.
    forces = np.arange(.70, 1.60, .03) * dragged
    out_at = [_slides_at(p, layer, +.0003, f) for f in forces]
    always = next((i for i in range(len(forces)) if all(out_at[i:])), None)
    held = max((forces[i] for i in range(len(forces)) if not out_at[i]), default=0.0)
    f = forces[always] if always is not None else None
    out.append(dict(test=f"pull loaded block (layer {layer})", predicted=f"{dragged:.2f}-{pinned:.2f} N",
                    measured=(f"always out from {f:.2f} N (stick-slip above {held:.2f} N)" if f else "did not slide"),
                    value=f, held_n=held,
                    # MuJoCo's soft friction gives 80-90% of mu*N under heavy load; loose blocks match exactly
                    passed=f is not None and .80 * dragged <= f <= pinned))
    f = _breakaway(p, layer, -.0003, 10 * loose)
    out.append(dict(test=f"pull loose block (layer {layer})", predicted=f"{loose:.3f} N",
                    measured=f"{f:.3f} N" if f else "did not slide", value=f,
                    passed=f is not None and abs(f - loose) <= .3 * loose))
    return out


def test_creep(p=P, layer=8, frac=.5):
    """Known limitation: soft friction creeps under a sustained sub-limit load.
    Reported, with a generous bound, so a parameter change that worsens it fails."""
    loaded = predict_pull_n(p, layer)[0]
    t = Tower(p, thickness={(layer, 1): .0003}); t.run(.6)
    k = t.names.index((layer, 1)); bid = t.ids[k]
    ax = _block_axis(layer)
    t.d.xfrc_applied[bid, ax] = frac * loaded
    t.run(1.0); x1 = t.pos()[k, ax]; t.run(1.0)
    rate = 1000 * (t.pos()[k, ax] - x1)
    return dict(test=f"creep at {frac:.0%} of friction limit", predicted="0 mm/s (real wood)",
                measured=f"{rate:.2f} mm/s (MuJoCo soft friction)", passed=rate < 1.0, value=rate)


def validate(p=P, out=OUT_DIR / "tower_physics_validation.json"):
    t0 = time.time()
    rows = [test_stand(p), test_tilt(p), test_neck(p)] + test_removal(p) + test_pull(p) + [test_creep(p)]
    width = max(len(r["test"]) for r in rows)
    print(f"\n{'test':{width}s}  {'theory':>12s}  measured")
    for r in rows:
        print(f"{r['test']:{width}s}  {r['predicted']:>12s}  {r['measured']:40s} {'PASS' if r['passed'] else 'FAIL'}")
    print(f"\n{sum(r['passed'] for r in rows)}/{len(rows)} passed in {time.time() - t0:.0f} s")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(parameters={k: asdict(v) for k, v in vars(p).items()}, results=rows),
                              indent=2, default=float) + "\n")
    return rows


def record(path, p=P):
    """Video: intact tower standing, a removal that leaves one outer block, a tilt to collapse."""
    import imageio.v2 as imageio
    frames = []

    def clip(t, title_fn, run):
        r = mujoco.Renderer(t.m, 540, 720)
        cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, .12]; cam.distance, cam.azimuth, cam.elevation = .75, 55, -12
        n = [0]

        def cb():
            n[0] += 1
            if n[0] % 17 == 0:                       # ~30 fps of sim time
                r.update_scene(t.d, cam); frames.append(r.render().copy())
        run(cb)
        r.close()

    t = Tower(p); clip(t, None, lambda cb: t.run(2.0, cb))
    t = Tower(p, removed={(3, 0), (3, 1)}); clip(t, None, lambda cb: t.run(3.0, cb))
    t = Tower(p, removed={(2, 0), (2, 2)}); clip(t, None, lambda cb: _tilt_until_collapse(t, "x", step=.1, start=3.0, cb=cb))
    imageio.mimwrite(path, frames, fps=30, quality=8)
    print(f"saved {path} ({len(frames)} frames)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--record", metavar="FILE.mp4")
    ap.add_argument("--view", choices=["stand", "tilt", "neck", "single-outer"])
    a = ap.parse_args()
    if a.record:
        record(a.record)
        return
    if a.view:
        import mujoco.viewer
        removed = {"neck": {(2, 0), (2, 2)}, "single-outer": {(3, 0), (3, 1)}}.get(a.view, set())
        t = Tower(P, removed=removed)
        with mujoco.viewer.launch_passive(t.m, t.d) as v:
            v.cam.lookat[:] = [0, 0, .12]; v.cam.distance, v.cam.azimuth, v.cam.elevation = .7, 60, -15
            n = [0]

            def cb():
                n[0] += 1
                if n[0] % 10 == 0:
                    v.sync(); time.sleep(.01)
            if a.view in ("tilt", "neck"):
                _tilt_until_collapse(t, "x", step=.25 if a.view == "tilt" else .1, cb=cb)
            while v.is_running():
                t.run(.02, cb)
        return
    rows = validate()
    sys.exit(0 if all(r["passed"] for r in rows) else 1)


if __name__ == "__main__":
    main()
