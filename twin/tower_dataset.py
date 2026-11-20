#!/usr/bin/env python3
"""Database of simulated Jenga tower runs: many ways a tower stands or falls.

Every run builds a tower with randomised physics (friction, block mass, block
thickness tolerance, height), applies one scenario, records whether and when it
collapsed, compares the outcome with a static-equilibrium prediction where one
exists, and stores the block trajectories. Runs are deterministic from their id,
so the database can be extended or regenerated exactly.

Scenarios
  removal  blocks vanish one at a time (perfect extraction)    -> pure equilibrium
  pull     a block is pulled along its length with a force ramp  -> friction + load
  poke     a short sideways impulse on a block in the upper half
  tilt     the table tilts slowly to an angle, in a random direction
  shake    the table shakes sideways (sinusoid)
  game     random extractions in sequence until it falls or 15 are out

Usage (from the repository root)
  python twin/tower_dataset.py generate --runs 300 --workers 4    # adds runs 0..299 (skips existing)
  python twin/tower_dataset.py summary                            # tables by scenario
  python twin/tower_dataset.py show 17                            # one run's record
  python twin/tower_dataset.py replay 17                          # watch a stored run in MuJoCo
  python twin/tower_dataset.py export runs.csv                    # flat CSV of every run
  python twin/tower_dataset.py check [--fix]                      # find (and redo) numerically unstable runs

Output: data/tower_runs/runs.sqlite and data/tower_runs/traj/run_XXXXXX.npz
(positions and orientations of every block at 25 Hz).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tower_physics as TP  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "data/tower_runs"
SCENARIOS = ("removal", "pull", "poke", "tilt", "shake", "game")
# Per-scenario definition version: `generate` redoes stored runs whose version is out of date.
# The "-m1" suffix is the physics version: runs made before the measured values are redone.
VERSIONS = dict(removal="1-m1", pull="1-m1", poke="3-m1", tilt="1-m1", shake="1-m1", game="1-m1")
FINGER = ('<body name="finger" mocap="true" pos="3 3 -1"><geom name="finger_geom" type="sphere" '
          'size=".006" friction=".5 .005 .0001" condim="3" rgba=".9 .3 .2 1"/></body>')
FRAME_EVERY_S = .04                # trajectory sample period (25 Hz)
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    seed INTEGER, scenario TEXT, version TEXT,
    layers INTEGER, friction REAL, block_mass_g REAL, thickness_tol_mm REAL,
    scenario_params TEXT,
    outcome TEXT, blocks_fell INTEGER,
    collapsed INTEGER, onset_s REAL, fraction_down REAL, max_top_disp_mm REAL,
    predicted_stable INTEGER, min_margin_mm REAL,
    blocks_removed INTEGER, sim_seconds REAL, wall_seconds REAL,
    timestep REAL, unstable INTEGER,
    traj_file TEXT, created TEXT
);
CREATE INDEX IF NOT EXISTS idx_scenario ON runs(scenario);
CREATE INDEX IF NOT EXISTS idx_outcome ON runs(outcome);
"""


# ------------------------------------------------------------------ static equilibrium prediction
def _split(load, x_res, supports):
    """Reactions on a rigid beam's supports (positions `supports`) for a total load
    `load` acting at x_res: force balance + moment balance; with three supports
    (statically indeterminate) take the least-squares solution, as for equal
    supports of equal stiffness. Negative reactions are dropped and re-solved."""
    xs = list(supports)
    while True:
        A = np.array([np.ones(len(xs)), xs])
        R = np.linalg.lstsq(A, np.array([load, load * x_res]), rcond=None)[0]
        if len(xs) == 1 or np.all(R >= -1e-12):
            return dict(zip(xs, np.clip(R, 0, None)))
        xs.pop(int(np.argmin(R)))


def equilibrium(p, removed):
    """Predict whether a tower with `removed` blocks stands, from statics alone.

    Load path, top down: every block is a rigid beam lying across the three
    crossing blocks of the layer below. Its load (own weight + what the blocks
    above rest on it, applied where they cross it) has a resultant along its
    length; the block stands if that resultant lies over its supports (each
    crossing block is 25 mm wide), and its load is passed down to them by
    force and moment balance. Margin = distance from the resultant to the
    nearer support edge, minimum over all blocks (negative = tips).
    Rigid blocks, no friction, no thickness variation. -> (stands, min_margin_mm)
    """
    L, W, pitch, g = int(p.v("layers")), p.v("block_width"), TP.slot_pitch(p), 9.81
    m = p.v("block_mass")
    off = [(s - 1) * pitch for s in range(3)]
    present = {(l, s) for l in range(L) for s in range(3) if (l, s) not in removed}
    # loads[(l, s)] = list of (force, position along the block's own length)
    loads = {b: [(m * g, 0.0)] for b in present}
    min_margin = math.inf
    for layer in range(L - 1, 0, -1):
        below = [s for s in range(3) if (layer - 1, s) in present]
        for s in range(3):
            b = (layer, s)
            if b not in present:
                continue
            F = sum(f for f, _ in loads[b])
            x = sum(f * x for f, x in loads[b]) / F
            if not below:
                return False, -1e3
            lo, hi = min(off[t] for t in below) - W / 2, max(off[t] for t in below) + W / 2
            margin = min(x - lo, hi - x)
            min_margin = min(min_margin, margin)
            # pass the load down: support t receives R at its crossing with b,
            # i.e. at position off[s] along the support block's length
            for t_pos, R in _split(F, float(np.clip(x, lo, hi)), [off[t] for t in below]).items():
                t = off.index(t_pos)
                loads[(layer - 1, t)].append((R, off[s]))
    return bool(min_margin > 0), float(1000 * min_margin)


# ------------------------------------------------------------------ one run
class Run:
    def __init__(self, run_id, base_seed=0, timestep=None):
        self.id = run_id
        self.seed = base_seed * 1_000_003 + run_id
        rng = np.random.default_rng(self.seed)
        self.rng = rng
        self.scenario = SCENARIOS[run_id % len(SCENARIOS)] if base_seed == 0 else SCENARIOS[rng.integers(len(SCENARIOS))]
        p = TP.TowerPhysics()
        # Ranges around Nirav's measurements (2026-11-20, see TowerPhysics sources).
        p.layers.value = int(rng.integers(10, 17))                 # real set: 16 layers
        p.friction.value = float(rng.uniform(.18, .30))            # measured mu 0.23 +/- 0.04
        p.block_mass.value = float(rng.uniform(.0176, .0195))      # measured 18.5 g
        self.tol = float(rng.choice([0.0, .0002, .0003, .0004]))   # measured thickness sd 0.19 mm
        # Per-block friction (measured trial spread). MuJoCo takes the larger mu of the two blocks
        # at a contact, which raises the mean by ~0.56 sd, so blocks are drawn that much lower.
        mu, sd = p.v("friction"), p.v("friction_sd")
        self.mu_block = {(l, s): float(np.clip(rng.normal(mu - .56 * sd, sd), .08, .6))
                         for l in range(p.layers.value) for s in range(3)} if self.tol else None
        if timestep:
            p.timestep.value = timestep
            p.contact_timeconst.value = max(p.contact_timeconst.value * timestep / .002, 2 * timestep)
        self.p = p
        L = p.layers.value
        thick = {(l, s): float(rng.uniform(-self.tol, self.tol)) for l in range(L) for s in range(3)} if self.tol else None
        self.tower = TP.Tower(p, thickness=thick, extra=FINGER, friction=self.mu_block)
        m = self.tower.m
        self.qadr = np.array([m.jnt_qposadr[m.body_jntadr[b]] for b in self.tower.ids])
        self.dadr = np.array([m.jnt_dofadr[m.body_jntadr[b]] for b in self.tower.ids])
        self.removed: list[tuple[int, int]] = []
        self.frames, self.quats, self.times = [], [], []
        self._every = max(1, round(FRAME_EVERY_S / m.opt.timestep))
        self._n = 0
        self.onset = None
        self.params = {}
        self.worst = None          # (stands, margin_mm) of the least stable state reached

    # -- helpers
    def _cb(self):
        self._n += 1
        if self._n % self._every == 0:
            d = self.tower.d
            self.frames.append(d.xpos[self.tower.ids].astype(np.float32).copy())
            self.quats.append(d.xquat[self.tower.ids].astype(np.float16).copy())
            self.times.append(d.time)
            if self.onset is None and self._fallen_now():
                self.onset = float(d.time)

    def _fallen_now(self):
        keep = [i for i, n in enumerate(self.tower.names) if n not in self.removed]
        return bool(np.any(self.ref[keep, 2] - self.tower.pos()[keep, 2] > .03))

    def run(self, seconds):
        self.tower.run(seconds, self._cb)

    def unstable(self):
        """MuJoCo hit NaN/Inf acceleration and reset the state: the run is not valid."""
        return self.tower.d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0

    def remove(self, name):
        """Perfect extraction: the block leaves the tower instantly. The static
        prediction is re-evaluated after every removal (the worst state counts)."""
        i = self.tower.names.index(name)
        d, m = self.tower.d, self.tower.m
        g = m.body_geomadr[self.tower.ids[i]]
        m.geom_contype[g] = m.geom_conaffinity[g] = 0       # gone: it can no longer touch anything
        m.geom_rgba[g, 3] = 0
        d.qpos[self.qadr[i]:self.qadr[i] + 7] = [2.0 + .1 * len(self.removed), 2.0, -1.0, 1, 0, 0, 0]
        d.qvel[self.dadr[i]:self.dadr[i] + 6] = 0
        m.body_gravcomp[self.tower.ids[i]] = 1.0             # and it stays put out of the way
        self.removed.append(name)
        stands, margin = equilibrium(self.p, set(self.removed))
        if self.worst is None or margin < self.worst[1]:
            self.worst = (stands, margin)

    def pick_block(self, top_margin=1):
        L = int(self.p.v("layers"))
        cand = [(l, s) for l in range(L - top_margin) for s in range(3) if (l, s) not in self.removed]
        return cand[int(self.rng.integers(len(cand)))]

    # -- scenarios
    def removal(self):
        n = int(self.rng.integers(1, 13))
        self.params = dict(n_remove=n)
        self.run(.5)
        for _ in range(n):
            self.remove(self.pick_block()); self.run(.3)
        self.run(2.0)

    def pull(self):
        name = self.pick_block()
        f_max = float(self.rng.uniform(.1, 4.0)); sign = float(self.rng.choice([-1, 1]))
        self.params = dict(block=name, f_max_n=f_max, direction=sign)
        self.run(.5)
        i = self.tower.names.index(name); bid = self.tower.ids[i]
        ax = 0 if name[0] % 2 == 0 else 1
        n = round(1.0 / self.tower.m.opt.timestep)
        for k in range(n):
            self.tower.d.xfrc_applied[bid, ax] = sign * f_max * (k + 1) / n
            self.tower.run(self.tower.m.opt.timestep, self._cb)
        self.run(1.5)
        self.tower.d.xfrc_applied[bid] = 0
        self.run(1.5)

    def poke(self):
        """A fingertip (12 mm sphere) moves horizontally into the tower at hand speed,
        then pulls back. Displacement-driven, like a real poke."""
        L = int(self.p.v("layers"))
        layer = int(self.rng.integers(L // 2, L)); slot = int(self.rng.integers(3))
        speed = float(self.rng.uniform(.02, .40)); depth = float(self.rng.uniform(.005, .040))
        ang = float(self.rng.uniform(0, 2 * math.pi))
        self.params = dict(block=(layer, slot), speed_ms=speed, depth_m=depth, heading_deg=math.degrees(ang))
        self.run(.5)
        d, m = self.tower.d, self.tower.m
        target = d.xpos[self.tower.ids[self.tower.names.index((layer, slot))]].copy()
        u = np.array([math.cos(ang), math.sin(ang), 0.0])
        half = (3 * self.p.v("block_width") + 2 * self.p.v("side_gap")) / 2 + .001
        # Where the approach ray (through the target block, along u) enters the tower's
        # square footprint (slab method), then back off so the 6 mm finger starts clear.
        t_enter = max((-half * math.copysign(1, u[i]) - target[i]) / u[i] for i in (0, 1) if abs(u[i]) > 1e-9)
        clear = .02 + .006
        start = target + u * (t_enter - clear)
        travel = clear + depth
        mid = m.body("finger").mocapid[0]
        n_in = max(1, round(travel / speed / m.opt.timestep))
        for k in range(n_in + 1):
            d.mocap_pos[mid] = start + u * travel * k / n_in
            self.tower.run(m.opt.timestep, self._cb)
        for k in range(n_in, -1, -1):
            d.mocap_pos[mid] = start + u * travel * k / n_in
            self.tower.run(m.opt.timestep, self._cb)
        d.mocap_pos[mid] = [3, 3, -1]
        self.run(2.5)

    def tilt(self):
        target = float(self.rng.uniform(0, 22)); heading = float(self.rng.uniform(0, 2 * math.pi))
        self.params = dict(tilt_deg=target, heading_deg=math.degrees(heading))
        self.run(.5)
        g = 9.81; n = 40
        for k in range(1, n + 1):
            a = math.radians(target * k / n)
            self.tower.m.opt.gravity[:] = (g * math.sin(a) * math.cos(heading), g * math.sin(a) * math.sin(heading), -g * math.cos(a))
            self.run(.05)
        self.run(2.5)

    def shake(self):
        amp = float(self.rng.uniform(0, 4.0)); freq = float(self.rng.uniform(1, 6)); heading = float(self.rng.uniform(0, 2 * math.pi))
        self.params = dict(accel_amp_ms2=amp, freq_hz=freq, heading_deg=math.degrees(heading))
        self.run(.5)
        m, dt = self.tower.m, self.tower.m.opt.timestep
        t0 = self.tower.d.time
        for _ in range(round(2.0 / dt)):
            # shaking the table = an extra horizontal inertial acceleration on every block
            a = amp * math.sin(2 * math.pi * freq * (self.tower.d.time - t0))
            m.opt.gravity[:] = (-a * math.cos(heading), -a * math.sin(heading), -9.81)
            self.tower.run(dt, self._cb)
        m.opt.gravity[:] = (0, 0, -9.81)
        self.run(2.0)

    def game(self):
        self.params = dict(max_extractions=15)
        self.run(.5)
        for _ in range(15):
            self.remove(self.pick_block(top_margin=2)); self.run(.6)
            if self._fallen_now():
                break
        self.run(1.5)

    def execute(self):
        t0 = time.time()
        self.tower.run(.3)                                 # settle before the reference snapshot
        self.ref = self.tower.pos()
        top = int(np.argmax(self.ref[:, 2]))
        getattr(self, self.scenario)()
        pos = self.tower.pos()
        keep = [i for i, n in enumerate(self.tower.names) if n not in self.removed]
        fell = int(np.sum(self.ref[keep, 2] - pos[keep, 2] > .03))
        frac = float(np.mean(self.ref[keep, 2] - pos[keep, 2] > self.p.v("block_height")))
        outcome = "stood" if fell == 0 else ("collapsed" if frac >= .25 else "partial")
        collapsed = outcome == "collapsed"
        disp = max(np.linalg.norm(f[top] - self.ref[top]) for f in self.frames) if self.frames else 0.0
        if self.removed:
            self.params["removed"] = [list(b) for b in self.removed]
        pred, margin = self.worst if (self.scenario in ("removal", "game") and self.worst) else (None, None)
        return dict(id=self.id, seed=self.seed, scenario=self.scenario, version=VERSIONS[self.scenario],
                    layers=int(self.p.v("layers")), friction=self.p.v("friction"),
                    block_mass_g=1000 * self.p.v("block_mass"), thickness_tol_mm=1000 * self.tol,
                    scenario_params=json.dumps(self.params, default=list),
                    outcome=outcome, blocks_fell=fell,
                    collapsed=int(collapsed), onset_s=self.onset, fraction_down=frac,
                    max_top_disp_mm=1000 * float(disp),
                    predicted_stable=None if pred is None else int(pred), min_margin_mm=margin,
                    blocks_removed=len(self.removed), sim_seconds=float(self.tower.d.time),
                    timestep=float(self.tower.m.opt.timestep), unstable=int(self.unstable()),
                    wall_seconds=time.time() - t0)



def _job(args):
    run_id, base_seed, traj_dir = args
    for dt in (None, .001, .0005):           # numerical blow-up: redo with a smaller time step
        r = Run(run_id, base_seed, dt)
        rec = r.execute()
        if not rec["unstable"]:
            break
    if traj_dir is not None:
        f = Path(traj_dir) / f"run_{run_id:06d}.npz"
        np.savez_compressed(f, t=np.array(r.times, np.float32), pos=np.stack(r.frames), quat=np.stack(r.quats),
                            names=np.array(r.tower.names, np.int16), removed=np.array(r.removed, np.int16).reshape(-1, 2))
        rec["traj_file"] = str(f.relative_to(REPO)) if f.is_relative_to(REPO) else str(f)
    else:
        rec["traj_file"] = None
    rec["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return rec


def connect(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(out / "runs.sqlite")
    db.executescript(SCHEMA)
    have = {c[1] for c in db.execute("PRAGMA table_info(runs)")}
    for col, typ in (("timestep", "REAL"), ("unstable", "INTEGER")):   # databases made before these existed
        if col not in have:
            db.execute(f"ALTER TABLE runs ADD COLUMN {col} {typ}")
    db.commit()
    return db


def check(out=DEFAULT_OUT, fix=False):
    """Find runs where MuJoCo silently reset the simulation (databases made before
    the automatic retry): a reset shows up in the trajectory as blocks jumping back
    to their start positions, or as NaN. --fix deletes them so `generate` redoes them."""
    db = connect(out)
    bad = []
    for rid, traj, unstable in db.execute("SELECT id, traj_file, unstable FROM runs ORDER BY id").fetchall():
        if unstable:
            bad.append((rid, "flagged unstable")); continue
        if not traj:
            continue
        f = REPO / traj if not Path(traj).is_absolute() else Path(traj)
        if not f.exists():
            continue
        pos = np.load(f)["pos"].astype(float)
        if not np.isfinite(pos).all():
            bad.append((rid, "NaN in trajectory")); continue
        # A MuJoCo reset puts every block back exactly where the run started.
        # Signature: in one frame, blocks that had moved > 3 cm are back within 0.5 mm of frame 0.
        away = np.linalg.norm(pos - pos[0], axis=2)                # (frames, blocks)
        snapped = (away[:-1] > .03) & (away[1:] < .0005)
        if np.any(snapped.sum(axis=1) >= 2):
            bad.append((rid, "simulation reset (blocks snapped back to their start)"))
    if not bad:
        print("No unstable runs found.")
        return
    for rid, why in bad:
        print(f"  run {rid}: {why}")
    if fix:
        db.executemany("DELETE FROM runs WHERE id=?", [(r,) for r, _ in bad]); db.commit()
        print(f"Deleted {len(bad)} runs. Run `generate` with the same --runs to redo them (with a smaller time step if needed).")
    else:
        print(f"{len(bad)} suspect runs. Add --fix to delete them, then re-run `generate`.")


def generate(n_runs, workers, out=DEFAULT_OUT, start=0, base_seed=0, keep_traj=True):
    import multiprocessing as mp
    db = connect(out)
    have = {rid for rid, sc, v in db.execute("SELECT id, scenario, version FROM runs") if VERSIONS.get(sc) == v}
    todo = [i for i in range(start, start + n_runs) if i not in have]
    traj_dir = None
    if keep_traj:
        traj_dir = Path(out) / "traj"; traj_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(todo)} runs to do ({n_runs - len(todo)} already up to date in the database), {workers} workers")
    cols = [c[1] for c in db.execute("PRAGMA table_info(runs)")]
    t0, done = time.time(), 0
    with mp.get_context("spawn").Pool(workers) as pool:
        for rec in pool.imap_unordered(_job, [(i, base_seed, traj_dir) for i in todo]):
            db.execute(f"INSERT OR REPLACE INTO runs ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                       [rec.get(c) for c in cols])
            db.commit(); done += 1
            if done % 10 == 0 or done == len(todo):
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(todo)}  ({rate:.2f} runs/s, ~{(len(todo) - done) / max(rate, 1e-9) / 60:.0f} min left)", flush=True)
    summary(out)


def summary(out=DEFAULT_OUT):
    db = connect(out)
    total = db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    print(f"\n{total} runs in {Path(out) / 'runs.sqlite'}\n")
    n_bad = db.execute("SELECT COUNT(*) FROM runs WHERE unstable=1").fetchone()[0]
    if n_bad:
        print(f"({n_bad} runs flagged numerically unstable are included below; run `check --fix` to redo them)\n")
    print(f"{'scenario':10s} {'runs':>5s} {'stood':>6s} {'partial':>8s} {'collapsed':>10s} {'mean onset s':>13s}")
    for sc, n, st, pa, co, onset in db.execute(
            "SELECT scenario, COUNT(*), SUM(outcome='stood'), SUM(outcome='partial'), SUM(outcome='collapsed'), "
            "AVG(onset_s) FROM runs GROUP BY scenario ORDER BY scenario"):
        print(f"{sc:10s} {n:5d} {st:6d} {pa:8d} {co:10d} {onset or 0:13.2f}")
    print("\nStatic-equilibrium load-path prediction vs simulation (removal + game runs).")
    print("The prediction assumes identical blocks; real blocks differ by tenths of a mm.")
    for label, cond in (("identical blocks", "thickness_tol_mm = 0"), ("blocks vary +/-0.2-0.4 mm", "thickness_tol_mm > 0")):
        rows = db.execute(f"SELECT predicted_stable, outcome FROM runs WHERE predicted_stable IS NOT NULL AND {cond}").fetchall()
        if not rows:
            continue
        fell = lambda o: o != "stood"
        agree = sum((p == 1) != fell(o) for p, o in rows)
        print(f"  {label:26s} {agree}/{len(rows)} agree ({agree / len(rows):.0%})   "
              f"pred fall+fell {sum(p == 0 and fell(o) for p, o in rows)}, "
              f"pred stand+stood {sum(p == 1 and not fell(o) for p, o in rows)}, "
              f"pred stand but fell {sum(p == 1 and fell(o) for p, o in rows)}, "
              f"pred fall but stood {sum(p == 0 and not fell(o) for p, o in rows)}")


def show(run_id, out=DEFAULT_OUT):
    db = connect(out); db.row_factory = sqlite3.Row
    r = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if r is None:
        sys.exit(f"run {run_id} not in the database")
    for k in r.keys():
        print(f"{k:18s} {r[k]}")


def export(path, out=DEFAULT_OUT):
    db = connect(out)
    cur = db.execute("SELECT * FROM runs ORDER BY id")
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow([c[0] for c in cur.description]); w.writerows(cur)
    print(f"wrote {path}")


def replay(run_id, out=DEFAULT_OUT, speed=1.0):
    """Re-simulate run `run_id` (deterministic from its id) in the MuJoCo viewer."""
    import mujoco.viewer
    db = connect(out)
    row = db.execute("SELECT seed, scenario, timestep FROM runs WHERE id=?", (run_id,)).fetchone()
    base_seed, dt = 0, None
    if row:
        if row[0] != run_id:
            base_seed = (row[0] - run_id) // 1_000_003
        if row[2] and abs(row[2] - .002) > 1e-9:
            dt = row[2]                          # replay a retried run with the step it was stored with
    r = Run(run_id, base_seed, dt)
    print(f"run {run_id}: {r.scenario}, {int(r.p.v('layers'))} layers, friction {r.p.v('friction'):.2f}")
    with mujoco.viewer.launch_passive(r.tower.m, r.tower.d) as v:
        v.cam.lookat[:] = [0, 0, .12]; v.cam.distance, v.cam.azimuth, v.cam.elevation = .8, 55, -15
        wall0, sim0 = time.time(), r.tower.d.time
        orig = r._cb

        def cb():
            orig()
            if r._n % 10 == 0:
                v.sync()
                lag = (r.tower.d.time - sim0) / speed - (time.time() - wall0)
                if lag > 0:
                    time.sleep(min(lag, .05))
                if not v.is_running():
                    raise KeyboardInterrupt
        r._cb = cb
        try:
            print(r.execute() | {})
            while v.is_running():
                v.sync(); time.sleep(.05)
        except KeyboardInterrupt:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="database folder")
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("generate"); g.add_argument("--runs", type=int, default=300)
    g.add_argument("--workers", type=int, default=2); g.add_argument("--start", type=int, default=0)
    g.add_argument("--no-traj", action="store_true", help="store results only, no trajectories")
    sp.add_parser("summary")
    s = sp.add_parser("show"); s.add_argument("run_id", type=int)
    r = sp.add_parser("replay"); r.add_argument("run_id", type=int); r.add_argument("--speed", type=float, default=1.0)
    e = sp.add_parser("export"); e.add_argument("path")
    c = sp.add_parser("check"); c.add_argument("--fix", action="store_true")
    a = ap.parse_args()
    if a.cmd == "generate":
        generate(a.runs, a.workers, a.out, a.start, keep_traj=not a.no_traj)
    elif a.cmd == "summary":
        summary(a.out)
    elif a.cmd == "show":
        show(a.run_id, a.out)
    elif a.cmd == "replay":
        replay(a.run_id, a.out, a.speed)
    elif a.cmd == "export":
        export(a.path, a.out)
    elif a.cmd == "check":
        check(a.out, a.fix)


if __name__ == "__main__":
    main()
