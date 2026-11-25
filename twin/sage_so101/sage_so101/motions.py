"""Excitation motions. One file per joint (only that joint moves, so its actuator dynamics are
identifiable without coupling), plus an all-joint random-waypoint file held out for validation.
Segments: slow ramp up/down (backlash/hysteresis), steps of several sizes (kp/kd/delay/torque limit),
chirp with velocity-limited amplitude (bandwidth/lag)."""
import numpy as np
from .common import save_motion, joint_names
import os

def _ramp(c, a, rate, hz):
    n = int(2 * a / rate * hz)
    up = np.linspace(-a, a, n); return c + np.concatenate([up, up[::-1]])

def _steps(c, a, hz, hold=1.0):
    seq = [0, .25, 0, -.25, 0, .5, 0, -.5, 0, 1.0, 0, -1.0, 0]
    return np.concatenate([np.full(int(hold * hz), c + a * s) for s in seq])

def _chirp(c, a, hz, dur=20.0, f0=0.2, f1=4.0, vlim=3.0):
    t = np.arange(int(dur * hz)) / hz
    ph = 2 * np.pi * (f0 * t + (f1 - f0) * t ** 2 / (2 * dur))
    f = f0 + (f1 - f0) * t / dur
    amp = np.minimum(a, vlim / (2 * np.pi * f))
    return c + amp * np.sin(ph)

def _minjerk(p0, p1, n):
    s = np.linspace(0, 1, n)[:, None]; return p0 + (p1 - p0) * (10 * s**3 - 15 * s**4 + 6 * s**5)

def per_joint(cfg, scale=1.0):
    hz = cfg["control_hz"]; J = cfg["joints"]; out = {}
    cen = np.array([j["center"] for j in J])
    for k, j in enumerate(J):
        a = j["amp"] * scale; c = j["center"]
        sig = np.concatenate([_ramp(c, a, 0.3, hz), _steps(c, a, hz), _chirp(c, a, hz)])
        m = np.tile(cen, (len(sig), 1)); m[:, k] = np.clip(sig, j["lo"], j["hi"])
        out[f"excite_{j['name']}"] = m
    return out

def random_waypoints(cfg, scale=1.0, dur=40.0, seed=0):
    hz = cfg["control_hz"]; J = cfg["joints"]; rng = np.random.default_rng(seed)
    cen = np.array([j["center"] for j in J]); amp = np.array([j["amp"] for j in J]) * scale
    segs, p = [], cen.copy()
    while sum(len(s) for s in segs) < dur * hz:
        q = cen + amp * rng.uniform(-1, 1, len(J)); segs.append(_minjerk(p, q, int(rng.uniform(1.0, 2.5) * hz))); p = q
    return np.vstack(segs)[: int(dur * hz)]

def generate(cfg, out_dir, scale=1.0):
    os.makedirs(out_dir, exist_ok=True); names = joint_names(cfg)
    files = per_joint(cfg, scale); files["random_waypoints"] = random_waypoints(cfg, scale)
    for n, arr in files.items(): save_motion(os.path.join(out_dir, n + ".txt"), names, arr)
    return sorted(files)
