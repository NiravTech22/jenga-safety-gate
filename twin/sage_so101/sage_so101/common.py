"""Shared config + trial I/O. A 'trial' is one motion played on sim OR real, same CSV layout,
so every downstream tool (fit, analyze) treats both identically (the SAGE paired-data idea)."""
import json, os
from dataclasses import dataclass
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config(path=None):
    path = path or os.path.join(ROOT, "configs", "so101.json")
    cfg = json.load(open(path))
    cfg["_dir"] = os.path.dirname(os.path.abspath(path))
    return cfg

def joint_names(cfg):
    return [j["name"] for j in cfg["joints"]]

@dataclass
class Trial:
    t: np.ndarray      # (N,) seconds
    cmd: np.ndarray    # (N,J) commanded position, rad, sim frame
    pos: np.ndarray    # (N,J) measured position, rad, sim frame
    joints: list

    @property
    def dt(self): return float(np.median(np.diff(self.t)))
    def vel(self):
        from scipy.signal import savgol_filter
        return savgol_filter(self.pos, 7, 2, deriv=1, delta=self.dt, axis=0)

def save_trial(path, tr):
    hdr = ",".join(["t"] + [f"cmd_{j}" for j in tr.joints] + [f"pos_{j}" for j in tr.joints])
    np.savetxt(path, np.column_stack([tr.t, tr.cmd, tr.pos]), delimiter=",", header=hdr, comments="")

def load_trial(path):
    with open(path) as f: cols = f.readline().strip().split(",")
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    joints = [c[4:] for c in cols if c.startswith("cmd_")]
    J = len(joints)
    return Trial(d[:, 0], d[:, 1:1 + J], d[:, 1 + J:1 + 2 * J], joints)

def load_motion(path):
    """SAGE motion-file format: header line of joint names, then radians per timestep."""
    with open(path) as f: names = f.readline().strip().lstrip("#").strip().split(",")
    return [n.strip() for n in names], np.loadtxt(path, delimiter=",", skiprows=1)

def save_motion(path, names, arr):
    np.savetxt(path, arr, delimiter=",", header=",".join(names), comments="", fmt="%.6f")
